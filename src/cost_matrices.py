import click
import pandas as pd
import numpy as np
import os
from io import StringIO
from collections import defaultdict
import json
from tqdm import tqdm
import git

import Levenshtein
from Bio import SeqIO, Align

from typing import List
from pdb import set_trace
from skbio import TreeNode # type: ignore
from ete3 import Tree # type: ignore

@click.group()
def cli():
    pass


def load_otus(data_path: str, metadata: pd.DataFrame) -> pd.DataFrame:
    ibd_data = pd.read_csv(data_path, dtype={0: str})
    _otus = ibd_data.set_index(ibd_data.columns[3])
    _otus.drop(columns=['patient', 'visit', 'site'], inplace=True)
    otus = _otus.T.copy()
    
    return otus[metadata['sample']].copy()


def load_metadata(metadata_path: str):
    # dianosis: CD = chrons disease, UC = ulcerative colitis, nonIBD = control
    ibd_metadata = pd.read_csv(metadata_path)

    # Drop the nonIBD label since we only have 1 example of it
    non_ibd_index = ibd_metadata[ibd_metadata.diagnosis == "nonIBD"].index.item()
    ibd_metadata.drop(index=non_ibd_index, inplace=True)
    
    return ibd_metadata


def _create_alignment_cost_matrix(sequences: List, output: str = "alignment_cost_matrix.npy"):
    if os.path.exists(output):
        print("Loading alignment cost matrix from disk")
        
        with open(output, "rb") as file_in:
            return np.load(file_in)
    else:
        print("Calculating alignment cost matrix")
        alignment_cost_matrix = np.zeros((len(sequences), len(sequences)), dtype=int)
        aligner = Align.PairwiseAligner()

        for i in range(len(sequences)):
            print(f"Calculating row {i}")
            for j in range(i + 1, len(sequences)):
                dist = aligner.align(sequences[i].seq, sequences[j].seq).score
                alignment_cost_matrix[i, j] = dist
                
        with open(output, "wb") as file_out:
            np.save(file_out, alignment_cost_matrix)
                
    return alignment_cost_matrix


def _create_levenshtein_cost_matrix(sequences: List, output: str = "levenshtein_cost_matrix.npy"):
    # Get distances
    if os.path.exists(output):
        print("Loading Levenshtein distance based cost matrix from disk")
        
        with open(output, "rb") as file_in:
            levenshtein_cost_matrix = np.load(file_in)
    else:
        print("Calculating Levenshtein distance based cost matrix")
        levenshtein_cost_matrix = np.zeros((len(sequences), len(sequences)), dtype=int)
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                dist = Levenshtein.distance(str(sequences[i].seq), str(sequences[j].seq))
                levenshtein_cost_matrix[i, j] = levenshtein_cost_matrix[j, i] = dist
        
        with open("levenshtein_cost_matrix.npy", "wb") as f:
            np.save(f, levenshtein_cost_matrix)
            
    return levenshtein_cost_matrix


def _create_phylogenetic_cost_matrix(otus: List[int], tree: TreeNode, output: str = "phylogenetic_cost_matrix.npy"):
    if os.path.exists(output):
        print("Loading phylogenetic cost matrix from disk")

        with open(output, "rb") as file_in:
            phylogenetic_cost_matrix = np.load(file_in)
    else:
        subtree = tree.shear(set(otus))
        phylogenetic_cost_matrix = subtree.tip_tip_distances().data
        
        with open(output, "wb") as file_out:
            np.save(file_out, phylogenetic_cost_matrix)
            
    return phylogenetic_cost_matrix
        

def _create_genome_cost_matrix(otus: pd.DataFrame, normalize: bool = False, output: str = "genome_cost_matrix.npy"):
    # FIXME: Clean up the logic and avoid saving temporary files
    blast_search_dir = os.path.join("blast_search_results_tmp")

    # For every OTU, list all taxids corresponding to a complete genome
    otu_to_taxid = defaultdict(set)
    pbar = tqdm(os.listdir(blast_search_dir), total=len(os.listdir(blast_search_dir)))

    for otu in pbar:
        subdir = os.path.join(blast_search_dir, otu)
        if os.path.isdir(subdir):
            files = os.listdir(subdir)
            info = [f for f in files if f.endswith("_1.json")][0]
            info_path = os.path.join(subdir, info)
            
            with open(info_path, "r") as f:
                info_dict = json.load(f)
                
            hits = info_dict['BlastOutput2']['report']['results']['search']['hits']
            for hit in hits:
                if 'title' in hit['description'][0].keys():
                    if "complete genome" in hit['description'][0]['title']:
                        otu_to_taxid[otu].add(hit['description'][0]['taxid'])

    # Convert sets to lists
    otu_to_taxid = {k: list(v) for k,v in otu_to_taxid.items()}

    with open("otu_and_taxon_ids.json", "w") as file_out:
        json.dump(otu_to_taxid, file_out)

    otu_to_taxid_first_hit = {k: v[0] for k,v in otu_to_taxid.items()}

    with open("otu_and_taxon_id_first_hit.json", "w") as file_out:
        json.dump(otu_to_taxid_first_hit, file_out)
        
    df = pd.read_csv("taxid_and_kmer.txt", sep="\t", names=["taxid", "kmer_vector_path"]).drop_duplicates(subset="taxid")
    taxid_to_otu = {v:k for k,v in otu_to_taxid_first_hit.items()}
    df['otu'] = df['taxid'].map(taxid_to_otu)

    df.drop('taxid', axis=1, inplace=True)
    df.set_index('otu', inplace=True)

    otu_to_vector = {
        otu: pd.read_csv(vector_path, sep="\t", index_col=0).to_numpy().flatten()
        for otu, vector_path in df['kmer_vector_path'].to_dict().items()
    }

    index = pd.read_csv(df.iloc[0].item(), sep="\t", index_col=0).index.to_list()

    genome_df = pd.DataFrame(otu_to_vector, index=index)

    # Ensure that genome_df's columns are in the same order as otus index
    genome_df_columns = [col for col in otus.index if col in genome_df.columns]
    genome_df = genome_df[genome_df_columns].copy()
    
    if os.path.exists(output):
        with open(output, "rb") as file_in:
            cost_matrix = np.load(file_in)
    else:
        cost_matrix = np.zeros((len(genome_df_columns), len(genome_df_columns)), dtype=float)

        for i in range(len(genome_df_columns)):
            for j in range(len(genome_df_columns)):
                otu_i = genome_df[genome_df_columns[i]].to_numpy()
                otu_j = genome_df[genome_df_columns[j]].to_numpy()
                
                if normalize:
                    otu_i_norm = otu_i / np.sum(otu_i)
                    otu_j_norm = otu_j / np.sum(otu_j)
                    dist = np.linalg.norm(otu_i_norm - otu_j_norm, 2)
                else:
                    dist = np.linalg.norm(otu_i - otu_j, 2)
                    
                cost_matrix[i, j] = dist
    
    return cost_matrix

    
@cli.command()
@click.option("--data_path", type=str, default="ihmp/ibd_data.csv", show_default=True)
@click.option("--metadata_path", type=str, default="ihmp/ibd_metadata_new.csv", show_default=True)
@click.option("--output", type=str, default="alignment_cost_matrix.npy", show_default=True)
def create_alignment_cost_matrix(data_path, metadata_path, output):
    metadata = load_metadata(metadata_path=metadata_path)
    otus = load_otus(data_path=data_path, metadata=metadata)
    
    # TODO: Add this as CLI option
    fasta_index = SeqIO.index("data/gg_13_5.fasta", "fasta")
    sequences = [fasta_index[otu] for otu in otus.index]
    return _create_alignment_cost_matrix(sequences=sequences, output=output)


@cli.command()
@click.option("--data_path", type=str, default="ihmp/ibd_data.csv", show_default=True)
@click.option("--metadata_path", type=str, default="ihmp/ibd_metadata_new.csv", show_default=True)
@click.option("--output", type=str, default="levenshtein_cost_matrix.npy", show_default=True)
def create_levenshtein_cost_matrix(data_path, metadata_path, output):
    metadata = load_metadata(metadata_path=metadata_path)
    otus = load_otus(data_path=data_path, metadata=metadata)
    
    fasta_index = SeqIO.index("data/gg_13_5.fasta", "fasta")
    sequences = [fasta_index[otu] for otu in otus.index][:100]
    return _create_levenshtein_cost_matrix(sequences=sequences, output=output)


@cli.command()
@click.option("--data_path", type=str, default="ihmp/ibd_data.csv", show_default=True)
@click.option("--metadata_path", type=str, default="ihmp/ibd_metadata_new.csv", show_default=True)
@click.option("--tree_path", type=str, default="data/gg_13_5_otus_99_annotated.tree", show_default=True)
@click.option("--output", type=str, default="phylogenetic_cost_matrix.npy", show_default=True)
def create_phylogenetic_cost_matrix(data_path: str, metadata_path, output):
    metadata = load_metadata(metadata_path=metadata_path)
    otus = load_otus(data_path=data_path, metadata=metadata)
    
    tree = Tree("data/gg_13_5_otus_99_annotated.tree", format=1, quoted_node_names=True)
    skbio_tree = TreeNode.read(StringIO(tree.write(format_root_node=True))) # type: ignore
    
    return _create_phylogenetic_cost_matrix(otus=otus.index.to_list(), tree=skbio_tree, output=output)
    

@cli.command()
@click.option("--data_path", type=str, default="ihmp/ibd_data.csv", show_default=True)
@click.option("--metadata_path", type=str, default="ihmp/ibd_metadata_new.csv", show_default=True)
@click.option("--normalize", is_flag=True, type=bool, default=False, show_default=True)
@click.option("--output", type=str, default="genom_cost_matrix.npy", show_default=True)
def create_genome_cost_matrix(data_path, metadata_path, normalize, output):
    metadata = load_metadata(metadata_path=metadata_path)
    otus = load_otus(data_path=data_path, metadata=metadata)
    
    return _create_genome_cost_matrix(otus=otus, normalize=normalize, output=output)


@cli.command()
def list_cost_matrix_subcommands():
    subcommands = [
        "create-alignment-cost-matrix",
        "create-genome-cost-matrix",
        "create-levenshtein-cost-matrix",
        "create-phylogenetic-cost-matrix"
    ]
    
    for command in subcommands:
        print(command)


@cli.command()
@click.option("--data_path", type=str, default="ihmp/ibd_data.csv", show_default=True)
@click.option("--metadata_path", type=str, default="ihmp/ibd_metadata_new.csv", show_default=True)
@click.option("--tree_path", type=str, default="data/gg_13_5_otus_99_annotated.tree", show_default=True)
def create_all_cost_matrices(data_path: str, metadata_path: str, tree_path: str):
    ROOT = str(git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir)
    output_basedir = os.path.join(ROOT, "data", "ot_cost_matrices")
    
    metadata = load_metadata(metadata_path=metadata_path)
    otus = load_otus(data_path=data_path, metadata=metadata)
    fasta_index = SeqIO.index("data/gg_13_5.fasta", "fasta")
    sequences = [fasta_index[otu] for otu in otus.index]
    
    tree = Tree(tree_path, format=1, quoted_node_names=True)
    skbio_tree = TreeNode.read(StringIO(tree.write(format_root_node=True))) # type: ignore
    
    print("Creating alignment_cost_matrix")
    _create_alignment_cost_matrix(sequences=sequences, output=os.path.join(output_basedir, "alignment_cost_matrix.npy"))
    print("Creating levenshtein_cost_matrix")
    _create_levenshtein_cost_matrix(sequences=sequences, output=os.path.join(output_basedir, "levenshtein_cost_matrix.npy"))
    print("Creating phylogenetic_cost_matrix")
    _create_phylogenetic_cost_matrix(otus=otus.index.to_list(), tree=skbio_tree, output=os.path.join(output_basedir, "phylogenetic_cost_matrix.npy"))
    print("Creating genome_cost_matrix")
    _create_genome_cost_matrix(otus=otus, normalize=False, output=os.path.join(output_basedir, "genome_cost_matrix.npy"))
    print("Creating genome_cost_matrix")
    _create_genome_cost_matrix(otus=otus, normalize=True, output=os.path.join(output_basedir, "genome_normalized_cost_matrix.npy"))


if __name__ == "__main__":
    cli()