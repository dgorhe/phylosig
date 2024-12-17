import click
import requests
from Bio import SeqIO
import time
import os


@click.command()
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--database", "-d", type=str, required=False, default="refseq_reference_genomes")
@click.option("--output", "-o", type=click.Path(writable=True), required=True, help="Output file for BLAST results.")
@click.option("--dry-run", "-n", is_flag=True, default=False, help="If present, then no POST request sent to URL + params")
@click.option("--timeout", "-t", default=120, help="Number of seconds to wait. Setting this to -1 waits unlimited time.")
def blast_search(fasta_file, database, output, dry_run, timeout):
    """
    Perform a BLAST search on the sequence in the given FASTA file
    and save the results to the specified output file. The output will
    be a zip file.
    """
    # Define constants for the NCBI BLAST API
    NCBI_BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    NCBI_BLAST_PARAMS = {
        "CMD": "Put",
        "DATABASE": database,
        "PROGRAM": "blastn",
        "FORMAT_TYPE": "JSON2",
    }

    if os.path.exists(output) and not dry_run:
        click.echo(f"{output} already exists, terminating program.")
        return
    elif not os.path.exists(output) and dry_run:
        click.echo(f"Will save output to: {output}")
        return
    elif os.path.exists(output) and dry_run:
        click.echo(f"{output} already exists. It will not be created in a full run.")
        return

    # Read the sequence from the input FASTA file
    with open(fasta_file) as f:
        record = next(SeqIO.parse(f, "fasta"))
        sequence = str(record.seq)

    # Submit the BLAST request to NCBI
    params = NCBI_BLAST_PARAMS.copy()
    params["QUERY"] = sequence

    response = requests.post(NCBI_BLAST_URL, data=params)

    # Check if the request was successful
    if response.status_code != 200:
        click.echo("Error submitting BLAST request.", err=True)
        return

    # Parse the request ID (RID) from the response
    rid = response.text.split("RID = ")[1].split("\n")[0].strip()

    # Poll for results
    click.echo("Waiting for results...", nl=False)
    time_elapsed = 0
    while time_elapsed < timeout:
        # Wait 5 seconds and add that to the time elapsed
        time.sleep(5)
        time_elapsed += 5

        result_response = requests.get(NCBI_BLAST_URL, params={
            "CMD": "Get",
            "RID": rid,
            "FORMAT_TYPE": "JSON2"
        })
        if "Status=WAITING" not in result_response.text:
            break
        click.echo(".", nl=False)

    if "Status=WAITING" in result_response.text:
        click.echo(f"Result timed out for rid: {rid}")
        click.echo(f"URL: {NCBI_BLAST_URL}")
        return

    click.echo("\nResults retrieved.")

    # Save results to output file
    with open(output, "wb") as f:
        f.write(result_response.content)

    click.echo(f"Results saved to {output}")


if __name__ == "__main__":
    blast_search()
