# ruff: noqa: PLR0915
import json
import pandas as pd
import os
import re
from emb_predict.embeddings.mt_embed import get_smiles_embeddings
from emb_predict.embeddings.llm_embed import compute_disease_embeddings
from emb_predict.embeddings.embeddings import get_embeddings_filename
from emb_predict.utils import (
    log,
    parse_string_to_dict,
    download_file_to_string,
    download_remote_directory,
)
from emb_predict.vectordb.vectordb import QdrantDB, init_vectordb
from emb_predict.application import set_project_paths, load_embeddings
from typing import Optional


def get_latest_ot_version():
    url = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/conf/platform.conf"
    version_str = download_file_to_string(url)
    version_obj = parse_string_to_dict(version_str)
    data_version = version_obj["data_version"]
    return data_version


def download_ot_data(download_dir: str, version: str = "latest"):
    log.info(f"Downloading OpenTargets data version {version} to {download_dir}")
    base_url = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{version}/output/etl/json"
    download_remote_directory(f"{base_url}/indication/", f"{download_dir}")
    download_remote_directory(f"{base_url}/molecule/", f"{download_dir}")
    log.info("Download complete.")


def extract_indications_from_json_entry(entry):
    data = []
    drug_id = re.sub(r"(CHEMBL)(\d+)", r"\1_\2", entry["id"])
    indications_data = entry["indications"]
    approved_indications = entry["approvedIndications"]

    # go through the indication, get the disease name for any approved indication
    if approved_indications:
        for indication in indications_data:
            if indication["disease"] in approved_indications:
                row = {
                    "drug_id": drug_id,
                    "disease_id": indication["disease"],
                    "disease_name": indication["efoName"],
                }
                data.append(row)
    return data


def extract_indications_from_ot_dir(directory: str):
    # initialize an empty dictionary to store the approved indications with their names
    entries = []

    # iterate over all json files in the directory
    for file in os.listdir(directory):
        if file.endswith(".json"):
            with open(f"{directory}/{file}") as f:
                # Load the JSON data
                json_content = f.read()

                # Split the JSON content into separate JSON strings
                # Assuming each JSON object is separated by a newline
                json_strings = json_content.strip().split("\n")

                # Parse each JSON string into a dictionary and store in a list
                json_objects = [json.loads(json_str) for json_str in json_strings]

                for entry in json_objects:
                    data = extract_indications_from_json_entry(entry)
                    entries.extend(data)

    entries_df = pd.DataFrame(entries)
    entries_df = entries_df.drop_duplicates()
    return entries_df


def extract_molecule_from_json_entry(entry):
    data = []
    if "canonicalSmiles" in entry:
        drug_id = re.sub(r"(CHEMBL)(\d+)", r"\1_\2", entry["id"])
        row = {
            "id": drug_id,
            "name": entry["name"] if "CHEMBL" not in entry["name"] else "",
            "smiles": entry["canonicalSmiles"],
        }
        data.append(row)
    return data


def extract_molecule_from_ot_file(directory: str):
    # initialize an empty dictionary to store the approved indications with their names
    entries = []

    # iterate over all json files in the directory
    for file in os.listdir(directory):
        if file.endswith(".json"):
            with open(f"{directory}/{file}") as f:
                # Load the JSON data
                json_content = f.read()

                # Split the JSON content into separate JSON strings
                # Assuming each JSON object is separated by a newline
                json_strings = json_content.strip().split("\n")

                # Parse each JSON string into a dictionary and store in a list
                json_objects = [json.loads(json_str) for json_str in json_strings]

                for entry in json_objects:
                    data = extract_molecule_from_json_entry(entry)
                    entries.extend(data)

    entries_df = pd.DataFrame(entries)
    entries_df = entries_df.drop_duplicates()
    return entries_df


def save_to_vectordb(
    df: pd.DataFrame,
    fields: list[str],
    db: QdrantDB,
    collection_name: Optional[str] = None,
):
    log.info(f"Preparing to save {df.shape[0]} items to {collection_name} in qdrantdb")

    items = []
    for index, row in df.iterrows():
        # now create a point struct
        payload = {}
        embedding = None
        if "embedding" not in row:
            log.error(f"Error: embedding not found in row {index}. Aborting save.")
            break
        embedding = row["embedding"]

        for field in fields:
            if field in row:
                if field == "embedding":
                    row[field] = ",".join([str(f) for f in row["embedding"]])
                payload[field] = row[field]

        item = {}
        item["payload"] = payload
        item["vector"] = embedding
        items.append(item)

    operation_info = db.add(collection_name, items, batch_size=100)
    log.info(f"Operation info: {operation_info}")


def prepare_ot_data(args):
    dataset = args.dataset
    if args.dataset_version == "latest":
        args.dataset_version = get_latest_ot_version()

    paths = set_project_paths(
        base_dir=args.base_dir,
        download_dir=args.download_dir,
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        dataset=args.dataset,
        dataset_version=args.dataset_version,
        drug_embedding_model=args.drug_embedding_model,
        drug_embedding_dim=args.drug_embedding_dim,
        disease_embedding_model=args.disease_embedding_model,
        disease_embedding_dim=args.disease_embedding_dim,
        make_dirs=True,
        config_file=args.config,
    )

    #### make all the filenames
    ot_drug_fp = f"{paths['download_dir']}/molecule"
    ot_disease_fp = f"{paths['download_dir']}/indication"

    if args.all:
        args.download = True
        args.prepare_drugs = True
        args.prepare_diseases = True
        args.store_disease_vectors = True
        args.store_drug_vectors = True

    if args.download:
        download_ot_data(
            download_dir=paths["download_dir"], version=paths["dataset_version"]
        )

    if args.prepare_drugs:
        log.info("Preparing drug embeddings")
        drug_df = extract_molecule_from_ot_file(ot_drug_fp)
        drug_df.to_csv(paths["dataset_drug_processed_fp"], index=False)

        drug_list = drug_df["smiles"].unique()
        embeddings_dict = get_smiles_embeddings(
            drug_list.tolist(), target_folder=paths["mt_dir"]
        )
        embeddings_df = pd.DataFrame(
            embeddings_dict.items(), columns=["item", "embedding"]
        )

        drug_embeddings_df = pd.merge(
            drug_df, embeddings_df, left_on="smiles", right_on="item", how="left"
        )
        drug_embeddings_df.drop("item", axis=1, inplace=True)

        # save the embeddings dataframe
        # columns: id, name, smiles, embedding (string)
        # embeddings_dimension = len(drug_embeddings_df['embedding'][0])
        drug_embeddings_df.to_csv(paths["drug_embedding_fp"], index=False)
        log.info(f"Drug embeddings saved to {paths['drug_embedding_fp']}")

    if args.prepare_diseases:
        log.info("Preparing disease embeddings")
        entries_df = extract_indications_from_ot_dir(ot_disease_fp)
        disease_df = (
            entries_df.groupby(["disease_name"])
            .agg({"disease_id": "first"})
            .reset_index()
        )
        entries_df = entries_df.drop("disease_name", axis=1)
        entries_df.to_csv(paths["dataset_drug_disease_processed_fp"], index=False)

        disease_df = disease_df.rename(
            columns={"disease_id": "id", "disease_name": "name"}
        )
        disease_df.to_csv(paths["dataset_disease_processed_fp"], index=False)

        disease_list = disease_df["name"].unique()
        embeddings_df = compute_disease_embeddings(
            model=paths["disease_embedding_model"], items=disease_list.tolist()
        )
        disease_embeddings_df = pd.merge(
            disease_df, embeddings_df, left_on="name", right_on="item", how="left"
        )
        disease_embeddings_df.drop("item", axis=1, inplace=True)

        a = embeddings_df["embedding"][0]
        disease_embedding_dimension = len(a)

        paths["disease_embedding_fp"] = get_embeddings_filename(
            paths["processed_dir"],
            dataset,
            "disease",
            paths["disease_embedding_model"],
            disease_embedding_dimension,
        )
        disease_embeddings_df.to_csv(paths["disease_embedding_fp"], index=False)
        log.info(f"Disease embeddings saved to {paths['disease_embedding_fp']}")

    if args.store_drug_vectors or args.store_disease_vectors:
        # read the embedding files in and store the results in the vectordb
        log.info("Saving embeddings to vectordb")
        collections = []
        collections.append(
            {
                "name": paths["vectordb_drug_collection"],
                "size": paths["drug_embedding_dim"],
            }
        ) if args.store_drug_vectors else None
        collections.append(
            {
                "name": paths["vectordb_disease_collection"],
                "size": paths["disease_embedding_dim"],
            }
        ) if args.store_disease_vectors else None

        try:
            db = init_vectordb(
                url=args.vectordb_url,
                api_key=args.vectordb_api_key,
                collections=collections,
                recreate=True,
            )
            if args.store_drug_vectors:
                save_to_vectordb(
                    df=load_embeddings(paths["drug_embedding_fp"]),
                    fields=["id", "name", "smiles", "embedding"],
                    db=db,
                    collection_name=paths["vectordb_drug_collection"],
                )

            if args.store_disease_vectors:
                save_to_vectordb(
                    df=load_embeddings(paths["disease_embedding_fp"]),
                    fields=["id", "name", "embedding"],
                    db=db,
                    collection_name=paths["vectordb_disease_collection"],
                )

        except Exception as e:
            log.error(f"Error initializing the vector database: {e}")
            return
        log.info("Embeddings saved to vectordb")
