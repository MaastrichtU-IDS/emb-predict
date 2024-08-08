# ruff: noqa: PLR0913
# ruff: noqa: PLW0602
import os
import yaml
from tqdm import tqdm
from functools import partialmethod
from fastembed import TextEmbedding
import torch
from xgboost import XGBClassifier
from emb_predict.utils import log
from emb_predict.vectordb.vectordb import init_vectordb
from emb_predict.embeddings.embeddings import (
    get_embeddings_filename,
    load_embeddings,
    load_csv2pd,
)
from typing import Optional

project_paths = {}
pre_loaded_data = {}
ot_version = "24.06"


def read_config_file(file: str):
    try:
        if not os.path.exists(file):
            raise (Exception("NoFile"))
    except Exception:
        log.error(f"Config file not found: {file}")
        return None

    with open(file) as f:
        return yaml.safe_load(f)


def set_args_from_config(args):
    if args.config is not None:
        config = read_config_file(args.config)
        if config is not None:
            for key, v in config.items():
                k = key.replace("-", "_")
                setattr(args, k, v)
    return args


def set_project_paths(
    base_dir: str = "data",
    download_dir: str = "download",
    processed_dir: str = "processed",
    model_dir: str = "models",
    dataset: str = "ot",
    dataset_version: str = "latest",
    drug_embedding_model: str = "mt",
    drug_embedding_dim: int = 512,
    disease_embedding_model: str = "llama3.1",
    disease_embedding_dim: int = 4096,
    vectordb_url: str = "http://localhost:6333",
    vectordb_api_key: Optional[str] = None,
    classifier_model: str = "xgb",
    model_file: str = "model.ubj",
    make_dirs: bool = False,
    config_file: Optional[str] = None,
) -> dict:
    config = {}
    if config_file is not None:
        config = read_config_file(config_file)
        if config is not None:
            values = {}
            for key, v in config.items():
                k = key.replace("-", "_")
                values[k] = v
            config = values

    dataset = dataset if "dataset" not in config else config["dataset"]
    dataset_version = (
        dataset_version
        if "dataset_version" not in config
        else config["dataset_version"]
    )

    download_dir = (
        f"{base_dir}/{download_dir}/{dataset}/{dataset_version}"
        if "download_dir" not in config
        else config["download_dir"]
    )
    processed_dir = (
        f"{base_dir}/{processed_dir}/{dataset}/{dataset_version}"
        if "processed_dir" not in config
        else config["processed_dir"]
    )
    training_dir = (
        f"{base_dir}/training/{dataset}/{dataset_version}/{drug_embedding_model}_{drug_embedding_dim}_{disease_embedding_model}_{disease_embedding_dim}"
        if "model_dir" not in config
        else config["training_dir"]
    )
    training_results_dir = (
        f"{training_dir}/results"
        if "training_results_dir" not in config
        else config["training_results_dir"]
    )
    model_dir = (
        f"{base_dir}/{model_dir}" if "model_dir" not in config else config["model_dir"]
    )
    embpredict_model_dir = (
        f"{model_dir}/embpredict"
        if "embpredict_model_dir" not in config
        else config["embpredict_model_dir"]
    )

    if make_dirs is True:
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(training_results_dir, exist_ok=True)
        os.makedirs(embpredict_model_dir, exist_ok=True)

    dataset_drug_processed_fp = (
        f"{processed_dir}/{dataset}_drug.csv"
        if "dataset_drug_processed_fp" not in config
        else config["dataset_drug_processed_fp"]
    )
    dataset_disease_processed_fp = (
        f"{processed_dir}/{dataset}_disease.csv"
        if "dataset_disease_processed_fp" not in config
        else config["dataset_disease_processed_fp"]
    )
    dataset_drug_disease_processed_fp = (
        f"{processed_dir}/{dataset}_indications.csv"
        if "dataset_drug_disease_processed_fp" not in config
        else config["dataset_drug_disease_processed_fp"]
    )

    drug_embedding_model = (
        drug_embedding_model
        if "drug_embedding_model" not in config
        else config["drug_embedding_model"]
    )
    drug_embedding_dim = (
        drug_embedding_dim
        if "drug_embedding_dim" not in config
        else config["drug_embedding_dim"]
    )
    disease_embedding_model = (
        disease_embedding_model
        if "disease_embedding_model" not in config
        else config["disease_embedding_model"]
    )
    disease_embedding_dim = (
        disease_embedding_dim
        if "disease_embedding_dim" not in config
        else config["disease_embedding_dim"]
    )
    drug_embedding_fp = (
        get_embeddings_filename(
            processed_dir, dataset, "drug", drug_embedding_model, drug_embedding_dim
        )
        if "drug_embedding_fp" not in config
        else config["drug_embedding_fp"]
    )
    disease_embedding_fp = (
        get_embeddings_filename(
            processed_dir,
            dataset,
            "disease",
            disease_embedding_model,
            disease_embedding_dim,
        )
        if "disease_embedding_fp" not in config
        else config["disease_embedding_fp"]
    )

    vectordb_url = (
        vectordb_url if "vectordb_url" not in config else config["vectordb_url"]
    )
    vectordb_api_key = (
        vectordb_api_key
        if "vectordb_api_key" not in config
        else config["vectordb_api_key"]
    )
    vectordb_drug_collection = (
        f"{dataset}_drug_{drug_embedding_model}_{drug_embedding_dim}"
        if "vectordb_drug_collection" not in config
        else config["vectordb_drug_collection"]
    )
    vectordb_disease_collection = (
        f"{dataset}_disease_{disease_embedding_model}_{disease_embedding_dim}"
        if "vectordb_disease_collection" not in config
        else config["vectordb_disease_collection"]
    )
    vectordb_user_drug_collection = (
        f"user_drug_{drug_embedding_model}_{drug_embedding_dim}"
        if "vectordb_user_drug_collection" not in config
        else config["vectordb_user_drug_collection"]
    )
    vectordb_user_disease_collection = (
        f"user_disease_{disease_embedding_model}_{disease_embedding_dim}"
        if "vectordb_user_disease_collection" not in config
        else config["vectordb_user_disease_collection"]
    )

    classifier_model = (
        classifier_model
        if "classifier_model" not in config
        else config["classifier_model"]
    )
    model_file = (
        f"{embpredict_model_dir}/{dataset}_{classifier_model}_{drug_embedding_model}_{drug_embedding_dim}_{disease_embedding_model}_{disease_embedding_dim}.ubj"
        if model_file == "model.ubj" and "model_file" not in config
        else config["model_file"]
    )

    return {
        "dataset": dataset,
        "dataset_version": dataset_version,
        "download_dir": download_dir,
        "processed_dir": processed_dir,
        "training_dir": training_dir,
        "training_results_dir": training_results_dir,
        "model_dir": model_dir,
        "embpredict_model_dir": embpredict_model_dir,
        "mt_dir": f"{model_dir}/mt",
        "dataset_drug_processed_fp": dataset_drug_processed_fp,
        "dataset_disease_processed_fp": dataset_disease_processed_fp,
        "dataset_drug_disease_processed_fp": dataset_drug_disease_processed_fp,
        "drug_embedding_model": drug_embedding_model,
        "drug_embedding_dim": drug_embedding_dim,
        "disease_embedding_model": disease_embedding_model,
        "disease_embedding_dim": disease_embedding_dim,
        "drug_embedding_fp": drug_embedding_fp,
        "disease_embedding_fp": disease_embedding_fp,
        "vectordb_url": vectordb_url,
        "vectordb_api_key": vectordb_api_key,
        "vectordb_drug_collection": vectordb_drug_collection,
        "vectordb_disease_collection": vectordb_disease_collection,
        "vectordb_user_drug_collection": vectordb_user_drug_collection,
        "vectordb_user_disease_collection": vectordb_user_disease_collection,
        "model_file": model_file,
    }


def load_application_components(project_paths) -> dict:
    global pre_loaded_data
    pre_loaded_data["paths"] = project_paths

    # disable tqdm progress bar
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # initialize vectordb
    collections = [
        {
            "name": project_paths["vectordb_drug_collection"],
            "size": project_paths["drug_embedding_dim"],
        },
        {
            "name": project_paths["vectordb_disease_collection"],
            "size": project_paths["disease_embedding_dim"],
        },
        {
            "name": project_paths["vectordb_user_drug_collection"],
            "size": project_paths["drug_embedding_dim"],
        },
        {
            "name": project_paths["vectordb_user_disease_collection"],
            "size": project_paths["disease_embedding_dim"],
        },
    ]
    try:
        pre_loaded_data["vectordb"] = init_vectordb(
            url=project_paths["vectordb_url"],
            api_key=project_paths["vectordb_api_key"],
            collections=collections,
        )
    except Exception as e:
        log.error(f"Error initializing vectordb: {e}")

    # load embedding data files
    try:
        log.info("Loading data files")
        pre_loaded_data["drug_disease_df"] = load_csv2pd(
            project_paths["dataset_drug_disease_processed_fp"]
        )
        pre_loaded_data["disease_df"] = load_embeddings(
            project_paths["disease_embedding_fp"]
        )
        pre_loaded_data["drug_df"] = load_embeddings(project_paths["drug_embedding_fp"])

        log.info("Loaded embedding data files")

    except Exception as e:
        log.error(f"Error loading data files: {e}")

    log.info("checking for fastembed files")

    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        cache_dir=project_paths["model_dir"] + "/fastembed",
    )

    # check for molecule embedding model
    log.info("Checking for pretrained model for smiles_transformer")
    from smiles_transformer.load_data import download_pretrained

    download_pretrained(target_folder=project_paths["mt_dir"])

    # load prediction model
    log.info("Loading model")
    try:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"  # gpu
        model = XGBClassifier(device=device)
        model.load_model(project_paths["model_file"])
        if device == "cuda":
            bst = model.get_booster()
            bst.set_param({"device": device})

        log.info(f"Loaded {project_paths['model_file']}")
        pre_loaded_data["model"] = model
    except Exception as e:
        log.error(f"Error loading model: {e}")

    log.info("Application components loaded")
    return pre_loaded_data


def get_latest_version_ot_processed_dir(processed_dir: str):
    try:
        items = os.listdir(processed_dir)
        folders = [
            item
            for item in items
            if item != "latest" and os.path.isdir(os.path.join(processed_dir, item))
        ]
        folders.sort(reverse=True)
        dataset_version = None
        if len(folders) > 0:
            dataset_version = folders[0]
        else:
            log.error(f"No processed data found in {processed_dir}")
    except Exception as e:
        log.error(f"An error occurred: {e}")
    return dataset_version
