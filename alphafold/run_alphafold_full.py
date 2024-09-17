# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union, Optional
from omegaconf import OmegaConf
from datetime import datetime

from absl import app
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
import numpy as np

# Internal import (7716).
logging.set_verbosity(logging.INFO)
  
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    is_prokaryote: Optional[bool] = None):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features, loading them from storage if they have already been saved to storage.
  t_0 = time.time()
  features_output_path = os.path.join(output_dir, 'features.pkl')
  if os.path.isfile(features_output_path):
      with open(features_output_path, 'rb') as f:
          feature_dict = pickle.load(f)
  else:
      if is_prokaryote is None:
          feature_dict = data_pipeline.process(
              input_fasta_path=fasta_path,
              msa_output_dir=msa_output_dir)
      else:
          feature_dict = data_pipeline.process(
              input_fasta_path=fasta_path,
              msa_output_dir=msa_output_dir,
              is_prokaryote=is_prokaryote)
      # Write out features as a pickled dictionary.
      with open(features_output_path, 'wb') as f:
          pickle.dump(feature_dict, f, protocol=4)
  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  unrelaxed_pdbs = {}
  relaxed_pdbs = {}
  ranking_confidences = {}

  # Run the models, ignoring FASTA files for which relaxed structures have already been predicted.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
    if not os.path.isfile(relaxed_output_path):
        model_random_seed = model_index + random_seed * num_models
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed)
        timings[f'process_features_{model_name}'] = time.time() - t_0

        t_0 = time.time()
        prediction_result = model_runner.predict(processed_feature_dict,
                                                 random_seed=model_random_seed)
        t_diff = time.time() - t_0
        timings[f'predict_and_compile_{model_name}'] = t_diff
        logging.info(
            'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
            model_name, fasta_name, t_diff)

        if benchmark:
          t_0 = time.time()
          model_runner.predict(processed_feature_dict,
                               random_seed=model_random_seed)
          t_diff = time.time() - t_0
          timings[f'predict_benchmark_{model_name}'] = t_diff
          logging.info(
              'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
              model_name, fasta_name, t_diff)

        plddt = prediction_result['plddt']
        ranking_confidences[model_name] = prediction_result['ranking_confidence']

        # Save the model outputs.
        result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)

        # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1)
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode)

        unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(unrelaxed_pdbs[model_name])

        if amber_relaxer:
          # Relax the prediction.
          t_0 = time.time()
          relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
          timings[f'relax_{model_name}'] = time.time() - t_0

          relaxed_pdbs[model_name] = relaxed_pdb_str

          # Save the relaxed PDB.
          relaxed_output_path = os.path.join(
              output_dir, f'relaxed_{model_name}.pdb')
          with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

  # Rank by model confidence and write out relaxed PDBs in rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if amber_relaxer:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    # Use the multimer flag as a proxy for determining whether the model(s) predicted TM-scores.
    label = 'iptm+ptm' if 'multimer' in model_preset else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))

  if remove_msas_after_use:
      # Remove MSAs' .sto files after using them for structure prediction(s), to significantly free up storage space
      msa_extensions_to_remove = ['.sto']
      for root_dir, dir_names, filenames in os.walk(msa_output_dir):
          for filename in filenames:
              if filename.endswith(msa_extensions_to_remove):
                  try:
                      os.remove(os.path.join(root_dir, filename))
                  except OSError:
                      logging.info(f'Error while deleting MSA file {os.path.join(root_dir, filename)}')

def run(config_file):

  conf = OmegaConf.load(config_file)

  # PRESETS
  db_preset = conf.af2.db_preset
  model_preset = conf.af2.model_preset

  # PATH VARIABLES
  fasta_paths = conf.af2.fasta_paths
  data_dir = conf.af2.data_dir
  output_dir = conf.af2.output_dir

  # BINARY PATHS
  jackhmmer_binary_path = shutil.which('jackhmmer')
  hhblits_binary_path = shutil.which('hhblits')
  hhsearch_binary_path = shutil.which('hhsearch')
  hmmsearch_binary_path = shutil.which('hmmsearch')
  hmmbuild_binary_path = shutil.which('hmmbuild')
  kalign_binary_path = shutil.which('kalign')

  # DATABASE PATHS
  uniref90_database_path = data_dir + conf.af2.uniref90
  mgnify_database_path = data_dir + conf.af2.mgnify
  pdb70_database_path = data_dir + conf.af2.pdb70
  template_mmcif_dir = data_dir + conf.af2.pdb_mmcif
  obsolete_pdbs_path = data_dir + conf.af2.obs_pdbs

  uniclust30_database_path = None
  small_bfd_database_path = None
  bfd_database_path = None
  uniprot_database_path = None
  pdb_seqres_database_path = None

  # MODEL SPECIFICATION
  max_template_date = datetime.today().strftime('%Y-%m-%d')
  is_prokaryote_list = None
  benchmark = False
  random_seed = None
  use_precomputed_msas = False
  remove_msas_after_use = False
  run_relax = True
  use_gpu_relax = None
  run_multimer_system = False

  for tool_name in ('jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not [f'{tool_name}'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                        'sure it is installed on your system.')

  use_small_bfd = db_preset == 'reduced_dbs'

  if use_small_bfd:
     small_bfd_database_path = data_dir + conf.af2.small_bfd
  else:
    bfd_database_path = data_dir + conf.af2.bfd
    uniclust30_database_path = data_dir + conf.af2.uniref30
    uniprot_database_path = data_dir + conf.af2.uniprot
    pdb_seqres_database_path = data_dir + conf.af2.pdb_seqres

  if model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  # Check that is_prokaryote_list has same number of elements as fasta_paths,
  # and convert to bool.
  if is_prokaryote_list:
    if len(is_prokaryote_list) != len(fasta_paths):
      raise ValueError('--is_prokaryote_list must either be omitted or match '
                       'length of --fasta_paths.')
    is_prokaryote_list = []
    for s in is_prokaryote_list:
      if s in ('true', 'false'):
        is_prokaryote_list.append(s == 'true')
      else:
        raise ValueError('--is_prokaryote_list must contain comma separated '
                         'true or false values.')
  else:  # Default is_prokaryote to False.
    is_prokaryote_list = [False] * len(fasta_names)

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        database_path=pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=template_mmcif_dir,
        max_template_date=max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_mmcif_dir,
        max_template_date=max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=jackhmmer_binary_path,
      hhblits_binary_path=hhblits_binary_path,
      uniref90_database_path=uniref90_database_path,
      mgnify_database_path=mgnify_database_path,
      bfd_database_path=bfd_database_path,
      uniclust30_database_path=uniclust30_database_path,
      small_bfd_database_path=small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=use_precomputed_msas)

  if run_multimer_system:
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=jackhmmer_binary_path,
        uniprot_database_path=uniprot_database_path,
        use_precomputed_msas=use_precomputed_msas)
  else:
    data_pipeline = monomer_data_pipeline

  model_runners = {}
  model_names = config.MODEL_PRESETS[model_preset]
  for model_name in model_names:
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  if run_relax:
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=use_gpu_relax)
  else:
    amber_relaxer = None

  random_seed = random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_names))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(fasta_paths):
    is_prokaryote = is_prokaryote_list[i] if run_multimer_system else None
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=benchmark,
        random_seed=random_seed,
        is_prokaryote=is_prokaryote)