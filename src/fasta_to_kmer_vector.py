from pandas import Series
from collections import defaultdict
from typing import List
from Bio.SeqIO.FastaIO import SimpleFastaParser

from itertools import product
import click


def load_fasta_as_string(fasta_path: str) -> str:
    with open(fasta_path, "r") as file_in:
        parsed = list(SimpleFastaParser(file_in))

        # Note: treating all nucleotides as equally confident is a naive way to do this
        # Add options for users (e.g. filtering out, converting, making separate vectors, etc.)
        sequence = parsed[0][1].upper()

    return sequence


def generate_all_nucleotide_combinations(n) -> List[str]:
    nucleotides = ['A', 'C', 'T', 'G']
    combinations = [''.join(combo) for combo in product(nucleotides, repeat=n)]

    return combinations


def generate_kmer_vector(fasta: str, k=5, step=1, start_offset=0) -> Series:
    # Start at the beginning of the string and get windows of size k
    START = start_offset
    END = k
    LENGTH = len(fasta)
    exclude_symbols = ['S', 'K', 'N', 'R', 'M', 'Y', 'W']

    # Use a defaultdict to incrementally add kmers and their counts
    kmer_counts = defaultdict(int)

    # Slide the window by `step` until we reach the end
    while END <= LENGTH:
        kmer = fasta[START:END]
        contains_excluded = any(char in kmer for char in exclude_symbols)

        if not contains_excluded:
            kmer_counts[kmer] += 1

        START += step
        END += step

    # If we didn't observe a k-mer in the genome, it's count will be 0
    all_possible_kmers = generate_all_nucleotide_combinations(k)
    kmers_to_add = set(all_possible_kmers) - set(kmer_counts.keys())

    for kmer in kmers_to_add:
        kmer_counts[kmer] = 0

    # Sort the k-mer keys to ensure we have same order across samples
    sorted_kmer_counts = dict(sorted(kmer_counts.items(), key=lambda x: x[0]))

    sorted_kmer_counts_series = Series(sorted_kmer_counts, name="kmer_counts")

    return sorted_kmer_counts_series


@click.command()
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to FASTA file containing genome")
@click.option("--output", "-o", type=click.Path(writable=True), help="File name for output (suffix determines type)")
@click.option("--normalize", "-n",  help="Normalize k-mer vector to sum to 1 (i.e. vec / sum(vec))", type=bool, is_flag=True, default=False)
@click.option("--verbose", "-v", help="Print when output file is saved", type=bool, is_flag=True, default=False)
def main(input, output, normalize, verbose):
    genome_seq = load_fasta_as_string(input)
    vec = generate_kmer_vector(genome_seq, k=5, step=1)

    if output.endswith("csv"):
        vec.to_csv(output)
    elif output.endswith("tsv"):
        vec.to_csv(output, sep="\t")
    elif output.endswith("parquet"):
        vec.to_parquet(output)

    if verbose:
        print(f"Output saved to: {output}")


if __name__ == "__main__":
    main()
