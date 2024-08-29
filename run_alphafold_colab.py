import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '4.0'

from IPython.utils import io
import os
import subprocess
import jax
import sys

import enum
from alphafold.notebooks import notebook_utils

### GLOBALS
# Single sequence entered = monomer model
@enum.unique
class ModelType(enum.Enum):
  MONOMER = 0
model_type_to_use = ModelType.MONOMER

test_url_pattern = 'https://storage.googleapis.com/alphafold-colab{:s}/latest/uniref90_2022_01.fasta.1'
JACKHMMER_BINARY_PATH = ""

MIN_PER_SEQUENCE_LENGTH = 16
MAX_PER_SEQUENCE_LENGTH = 4000
MAX_MONOMER_MODEL_LENGTH = 2500
MAX_LENGTH = 4000
MAX_VALIDATED_LENGTH = 3000

### PYTHON IMPORTS
import collections
import copy
from concurrent import futures
import json
import random
import shutil

from urllib import request
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import feature_processing
from alphafold.data import msa_pairing
from alphafold.data import pipeline
#from alphafold.data import pipeline_multimer
from alphafold.data.tools import jackhmmer

from alphafold.common import confidence
from alphafold.common import protein

from alphafold.relax import relax
from alphafold.relax import utils

from IPython import display
from ipywidgets import GridspecLayout
from ipywidgets import Output
from omegaconf import OmegaConf

from Bio import SeqIO

# Color bands for visualizing plddt
PLDDT_BANDS = [(0, 50, '#FF7D45'),
               (50, 70, '#FFDB13'),
               (70, 90, '#65CBF3'),
               (90, 100, '#0053D6')]

def fetch(source):
  request.urlretrieve(test_url_pattern.format(source))
  return source

def get_msa(sequences):
  """Searches for MSA for given sequences using chunked Jackhmmer search.
  
  Args:
    sequences: A list of sequences to search against all databases.

  Returns:
    A dictionary mapping unique sequences to dicionaries mapping each database
    to a list of  results, one for each chunk of the database.
  """
  sequence_to_fasta_path = {}
  # Deduplicate to not do redundant work for multiple copies of the same chain in homomers.
  for sequence_index, sequence in enumerate(sorted(set(sequences)), 1):
    fasta_path = f'target_{sequence_index:02d}.fasta'
    with open(fasta_path, 'wt') as f:
      f.write(f'>query\n{sequence}')
    sequence_to_fasta_path[sequence] = fasta_path

  # Run the search against chunks of genetic databases (since the genetic
  # databases don't fit in Colab disk).
  raw_msa_results = {sequence: {} for sequence in sequence_to_fasta_path.keys()}
  print('\nGetting MSA for all sequences')

  for db_config in MSA_DATABASES:
    db_name = db_config['db_name']
    jackhmmer_runner = jackhmmer.Jackhmmer(
        binary_path=JACKHMMER_BINARY_PATH,
        database_path=db_config['db_path'],
        get_tblout=True,
        num_streamed_chunks=db_config['num_streamed_chunks'],
        z_value=db_config['z_value'])
    # Query all unique sequences against each chunk of the database to prevent
    # redunantly fetching each chunk for each unique sequence.
    results = jackhmmer_runner.query_multiple(list(sequence_to_fasta_path.values()))
    for sequence, result_for_sequence in zip(sequence_to_fasta_path.keys(), results):
      raw_msa_results[sequence][db_name] = result_for_sequence

  return raw_msa_results

def get_sequence(fasta_file):
    input_seqs = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        if "design" not in seq_record.id: # Skip first sequence ("GGGGG...GGG")
            index = seq_record.description.find("score=")
            input_seqs[seq_record.description[index + 6: index + 12]] = seq_record.seq

    sequence = str(input_seqs[max(input_seqs)])

    # Following Alphafold formatting :/
    sequence_2 = ''  #@param {type:"string"}
    sequence_3 = ''  #@param {type:"string"}
    sequence_4 = ''  #@param {type:"string"}
    sequence_5 = ''  #@param {type:"string"}
    sequence_6 = ''  #@param {type:"string"}
    sequence_7 = ''  #@param {type:"string"}
    sequence_8 = ''  #@param {type:"string"}
    sequence_9 = ''  #@param {type:"string"}
    sequence_10 = ''  #@param {type:"string"}
    sequence_11 = ''  #@param {type:"string"}
    sequence_12 = ''  #@param {type:"string"}
    sequence_13 = ''  #@param {type:"string"}
    sequence_14 = ''  #@param {type:"string"}
    sequence_15 = ''  #@param {type:"string"}
    sequence_16 = ''  #@param {type:"string"}
    sequence_17 = ''  #@param {type:"string"}
    sequence_18 = ''  #@param {type:"string"}
    sequence_19 = ''  #@param {type:"string"}
    sequence_20 = ''  #@param {type:"string"}

    input_sequences = (sequence, 
    sequence_2, sequence_3, sequence_4, sequence_5, 
    sequence_6, sequence_7, sequence_8, sequence_9, sequence_10,
    sequence_11, sequence_12, sequence_13, sequence_14, sequence_15, 
    sequence_16, sequence_17, sequence_18, sequence_19, sequence_20)

    return sequence

# Find the closest source
ex = futures.ThreadPoolExecutor(3)
fs = [ex.submit(fetch, source) for source in ['', '-europe', '-asia']]
source = None

for f in futures.as_completed(fs):
    source = f.result()
    ex.shutdown()
    break

DB_ROOT_PATH = f'https://storage.googleapis.com/alphafold-colab{source}/latest/'

# The z_value is the number of sequences in a database.
MSA_DATABASES = [
    {'db_name': 'uniref90',
    'db_path': f'{DB_ROOT_PATH}uniref90_2022_01.fasta',
    'num_streamed_chunks': 62,
    'z_value': 144_113_457},
    {'db_name': 'smallbfd',
    'db_path': f'{DB_ROOT_PATH}bfd-first_non_consensus_sequences.fasta',
    'num_streamed_chunks': 17,
    'z_value': 65_984_053},
    {'db_name': 'mgnify',
    'db_path': f'{DB_ROOT_PATH}mgy_clusters_2022_05.fasta',
    'num_streamed_chunks': 120,
    'z_value': 623_796_864},
]

TOTAL_JACKHMMER_CHUNKS = sum([cfg['num_streamed_chunks'] for cfg in MSA_DATABASES])
MAX_HITS = {
    'uniref90': 10_000,
    'smallbfd': 5_000,
    'mgnify': 501,
    'uniprot': 50_000,
}

def run(config_file):
    conf = OmegaConf.load(config_file)    

    PARAMS_PATH = conf.af2.params
    JACKHMMER_BINARY_PATH = conf.af2.hmmer

    if jax.local_devices()[0].platform == 'tpu':
        raise RuntimeError('Colab TPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
    elif jax.local_devices()[0].platform == 'cpu':
        raise RuntimeError('Colab CPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
    else:
        print(f'Running with {jax.local_devices()[0].device_kind} GPU')

    input_sequences = get_sequence(conf.af2.fasta_paths)

    # Validate the input sequences.
    sequences = input_sequences

    if len(sequences) == 1:
        print('Using the single-chain model.')
    else:
        print('Too many sequences. Grabbing first one.')

    # Check whether total length exceeds limit.
    total_sequence_length = sum([len(seq) for seq in sequences])
    if total_sequence_length > MAX_LENGTH:
        raise ValueError('The total sequence length is too long: '
                    f'{total_sequence_length}, while the maximum is '
                    f'{MAX_LENGTH}.')

    # Check whether we exceed the monomer limit.
    if len(sequences[0]) > MAX_MONOMER_MODEL_LENGTH:
        raise ValueError(
            f'Input sequence is too long: {len(sequences[0])} amino acids, while '
            f'the maximum for the monomer model is {MAX_MONOMER_MODEL_LENGTH}.')
        
    if total_sequence_length > MAX_VALIDATED_LENGTH:
        print('WARNING: The accuracy of the system has not been fully validated '
                'above 3000 residues, and you may experience long running times or '
                f'run out of memory. Total sequence length is {total_sequence_length} '
                'residues.')

    # Takes about 20 minutes using 1 GPU
    features_for_chain = {}
    raw_msa_results_for_sequence = get_msa(sequences)

    for sequence_index, sequence in enumerate(sequences, start=1):
        raw_msa_results = copy.deepcopy(raw_msa_results_for_sequence[sequence])

        # Extract the MSAs from the Stockholm files.
        # NB: deduplication happens later in pipeline.make_msa_features.
        single_chain_msas = []
        uniprot_msa = None
        for db_name, db_results in raw_msa_results.items():
            merged_msa = notebook_utils.merge_chunked_msa(
                results=db_results, max_hits=MAX_HITS.get(db_name))
            if merged_msa.sequences and db_name != 'uniprot':
                single_chain_msas.append(merged_msa)
                msa_size = len(set(merged_msa.sequences))
                print(f'{msa_size} unique sequences found in {db_name} for sequence {sequence_index}')
            elif merged_msa.sequences and db_name == 'uniprot':
                uniprot_msa = merged_msa

        notebook_utils.show_msa_info(single_chain_msas=single_chain_msas, sequence_index=sequence_index)

        # Turn the raw data into model features.
        feature_dict = {}
        feature_dict.update(pipeline.make_sequence_features(
            sequence=sequence, description='query', num_res=len(sequence)))
        feature_dict.update(pipeline.make_msa_features(msas=single_chain_msas))
        # We don't use templates in AlphaFold Colab notebook, add only empty placeholder features.
        feature_dict.update(notebook_utils.empty_placeholder_template_features(
            num_templates=0, num_res=len(sequence)))

        features_for_chain[protein.PDB_CHAIN_IDS[sequence_index - 1]] = feature_dict

    # Do further feature post-processing depending on the model type.
    np_example = features_for_chain[protein.PDB_CHAIN_IDS[0]]

    run_relax = True
    relax_use_gpu = False

    # Run the model
    model_names = config.MODEL_PRESETS['monomer'] + ('model_2_ptm',)

    output_dir = conf.af2.output_dir
    os.makedirs(output_dir, exist_ok=True)

    plddts = {}
    ranking_confidences = {}
    pae_outputs = {}
    unrelaxed_proteins = {}

    for model_name in model_names:

        cfg = config.model_config(model_name)
        cfg.data.eval.num_ensemble = 1

        params = data.get_model_haiku_params(model_name, PARAMS_PATH)
        model_runner = model.RunModel(cfg, params)
        processed_feature_dict = model_runner.process_features(np_example, random_seed=0)
        prediction = model_runner.predict(processed_feature_dict, random_seed=random.randrange(sys.maxsize))

        mean_plddt = prediction['plddt'].mean()

        if 'predicted_aligned_error' in prediction:
            pae_outputs[model_name] = (prediction['predicted_aligned_error'],
                                        prediction['max_predicted_aligned_error'])
        else:
            # Monomer models are sorted by mean pLDDT. Do not put monomer pTM models here as they
            # should never get selected.
            ranking_confidences[model_name] = prediction['ranking_confidence']
            plddts[model_name] = prediction['plddt']
        
        # Set the b-factors to the per-residue plddt.
        final_atom_mask = prediction['structure_module']['final_atom_mask']
        b_factors = prediction['plddt'][:, None] * final_atom_mask
        unrelaxed_protein = protein.from_prediction(
            processed_feature_dict,
            prediction,
            b_factors=b_factors,
            remove_leading_feature_dimension=(
                model_type_to_use == ModelType.MONOMER))
        unrelaxed_proteins[model_name] = unrelaxed_protein

        # Delete unused outputs to save memory.
        del model_runner
        del params
        del prediction

    # Find the best model according to the mean pLDDT.
    best_model_name = max(ranking_confidences.keys(), key=lambda x: ranking_confidences[x])

    if run_relax:
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=0,
            tolerance=2.39,
            stiffness=10.0,
            exclude_residues=[],
            max_outer_iterations=3,
            use_gpu=relax_use_gpu)
        relaxed_pdb, _, _ = amber_relaxer.process(prot=unrelaxed_proteins[best_model_name])
    else:
        print('Warning: Running without the relaxation stage.')
        relaxed_pdb = protein.to_pdb(unrelaxed_proteins[best_model_name])

    # Construct multiclass b-factors to indicate confidence bands
    # 0=very low, 1=low, 2=confident, 3=very high
    banded_b_factors = []
    for plddt in plddts[best_model_name]:
        for idx, (min_val, max_val, _) in enumerate(PLDDT_BANDS):
            if plddt >= min_val and plddt <= max_val:
                banded_b_factors.append(idx)
                break
    banded_b_factors = np.array(banded_b_factors)[:, None] * final_atom_mask
    to_visualize_pdb = utils.overwrite_b_factors(relaxed_pdb, banded_b_factors)

    # Write out the prediction
    pred_output_path = os.path.join(output_dir, 'selected_prediction.pdb')
    with open(pred_output_path, 'w') as f:
        f.write(relaxed_pdb)

    show_sidechains = True
    def plot_plddt_legend():
        """Plots the legend for pLDDT."""
        thresh = ['Very low (pLDDT < 50)',
                    'Low (70 > pLDDT > 50)',
                    'Confident (90 > pLDDT > 70)',
                    'Very high (pLDDT > 90)']

        colors = [x[2] for x in PLDDT_BANDS]

        plt.figure(figsize=(2, 2))
        for c in colors:
            plt.bar(0, 0, color=c)
        plt.legend(thresh, frameon=False, loc='center', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.title('Model Confidence', fontsize=20, pad=20)
        return plt

    # Color the structure by per-residue pLDDT
    color_map = {i: bands[2] for i, bands in enumerate(PLDDT_BANDS)}
    view = py3Dmol.view(width=800, height=600)
    view.addModelsAsFrames(to_visualize_pdb)
    style = {'cartoon': {'colorscheme': {'prop': 'b', 'map': color_map}}}
    if show_sidechains:
        style['stick'] = {}
        view.setStyle({'model': -1}, style)
        view.zoomTo()

    grid = GridspecLayout(1, 2)
    out = Output()
    with out:
        view.show()
    grid[0, 0] = out

    out = Output()
    with out:
        plot_plddt_legend().show()
    grid[0, 1] = out

    display.display(grid)

    # Display pLDDT and predicted aligned error (if output by the model).
    if pae_outputs:
        num_plots = 2
    else:
        num_plots = 1

    plt.figure(figsize=[8 * num_plots, 6])
    plt.subplot(1, num_plots, 1)
    plt.plot(plddts[best_model_name])
    plt.title('Predicted LDDT')
    plt.xlabel('Residue')
    plt.ylabel('pLDDT')

    if num_plots == 2:
        plt.subplot(1, 2, 2)
        pae, max_pae = list(pae_outputs.values())[0]
        plt.imshow(pae, vmin=0., vmax=max_pae, cmap='Greens_r')
        plt.colorbar(fraction=0.046, pad=0.04)

    # Display lines at chain boundaries.
    best_unrelaxed_prot = unrelaxed_proteins[best_model_name]
    total_num_res = best_unrelaxed_prot.residue_index.shape[-1]
    chain_ids = best_unrelaxed_prot.chain_index
    for chain_boundary in np.nonzero(chain_ids[:-1] - chain_ids[1:]):
        if chain_boundary.size:
            plt.plot([0, total_num_res], [chain_boundary, chain_boundary], color='red')
            plt.plot([chain_boundary, chain_boundary], [0, total_num_res], color='red')

    plt.title('Predicted Aligned Error')
    plt.xlabel('Scored residue')
    plt.ylabel('Aligned residue')

    # Save the predicted aligned error (if it exists).
    pae_output_path = os.path.join(output_dir, 'predicted_aligned_error.json')
    if pae_outputs:
        # Save predicted aligned error in the same format as the AF EMBL DB.
        pae_data = confidence.pae_json(pae=pae, max_pae=max_pae.item())
        with open(pae_output_path, 'w') as f:
            f.write(pae_data)
