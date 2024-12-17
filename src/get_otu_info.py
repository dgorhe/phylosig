import click
import json
from typing import Dict, Union, List
import requests
import os
from pdb import set_trace
import re


def get_info(path: str, output_dir: str) -> None:
    """
    Get info for a Greengenes OTU from RNACentral API

    Args:
        path (str): Path to file containing Greengenes OTU ids (one per line)
        output_dir (str): Directory to save each JSON returned by RNACentral
    """
    base_url = "https://rnacentral.org/api/v1/rna/"

    with open(path, "r") as file_in:
        for line in file_in.readlines():
            params = {"external_id": line}
            response = requests.post(base_url, params=params)
            info = response.json()

            output_path = os.path.join(output_dir, line)
            with open(output_path, "w") as file_out:
                json.dump(info, file_out)


def get_xrefs(path: str) -> Dict:
    """
    Get the external references (xrefs) from RNACentral

    Args:
        path (str): Path to JSON returned from RNACentral with a UID

    Returns:
        Dict: External references as a dictionary
    """
    with open(path, "r") as file_in:
        info = json.load(file_in)

    xrefs = []

    for entry in info['results']:
        xref = entry['xrefs']
        response = requests.post(xref)
        xrefs.append(response.json())

    return {"xrefs": xrefs}


def is_valid_ena_url(string: str) -> bool:
    pattern = r'https://www\.ebi\.ac\.uk/ena/browser/view/[A-Z0-9]+(?:\.\d+)?'
    match = re.match(pattern=pattern, string=string)
    if match is not None:
        return True

    return False


def extract_ena_urls(xref: Union[Dict, str]) -> List[str]:
    if type(xref) is str:
        with open(xref, "r") as file_in:
            external_references = json.load(file_in)
    elif type(xref) is Dict:
        external_references = xref

    xrefs = external_references["xrefs"]
    urls: List[str] = []

    for x in xrefs:
        results = x["results"]
        for result in results:
            if "accession" in result.keys():
                potential_url = result["accession"]["expert_db_url"]
                accession_id = potential_url.split("/")[-1]

                if is_valid_ena_url(potential_url):
                    urls.append(accession_id)

    return urls


def get_genomes(xref: Union[Dict, str]) -> str:
    if type(xref) is str:
        with open(xref, "r") as file_in:
            external_references = json.load(file_in)
    elif type(xref) is Dict:
        external_references = xref

    # genomes = []
    xrefs = external_references["xrefs"]

    for x in xrefs:
        results = x["results"]
        for result in results:
            if "accession" in result.keys():
                potential_url = result["accession"]["expert_db_url"]
                set_trace()

                if is_valid_ena_url(potential_url):
                    id = potential_url.split("/")[-1]
                    req_url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{id}"
                    set_trace()

                    response = requests.post(req_url)
                    genome = response.json()
                    set_trace()

    return ""


@click.command()
@click.option("--input", "-i", type=click.Path(exists=True), help="Input JSON file from RNA Central")
@click.option("--output", "-o", type=click.Path(), help="Path to output JSON file containing external references (xrefs)")
def main(input, output):
    # get_info(input)
    # if os.path.exists(output):
    #     print(f"{output} already exists.")
    #     return

    # print(f"Processing {input}")
    # xrefs = get_xrefs(input)

    # with open(output, "w") as file_out:
    #     json.dump(xrefs, file_out)

    urls = extract_ena_urls(input)
    with open(output, "w") as file_out:
        for u in urls:
            file_out.write(u + "\n")


if __name__ == "__main__":
    main()
