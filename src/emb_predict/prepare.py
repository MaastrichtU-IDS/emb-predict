from emb_predict.data_source.opentargets import prepare_ot_data
from emb_predict.application import set_args_from_config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="prepareTrainingData", description="Tool for preparing training data."
    )

    parser.add_argument(
        "--dataset", help="Dataset to use", choices=["ot"], default="ot"
    )
    parser.add_argument(
        "--dataset_version", help="Version of the dataset", default="latest"
    )

    parser.add_argument(
        "--all",
        help="download, prepare, and store vectors",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--download", help="Download of the dataset", action="store_true", default=False
    )
    parser.add_argument(
        "--prepare-drugs",
        help="Do not prepare the drug embeddings",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--prepare-diseases",
        help="Do not prepare the disease embeddings",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--store-drug-vectors",
        help="Store the vectors in the vectordb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--store-disease-vectors",
        help="Store the vectors in the vectordb",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--drug-embedding-model", help="Drug embedding model", default="mt"
    )
    parser.add_argument(
        "--disease-embedding-model", help="Disease embedding model", default="llama3.1"
    )
    parser.add_argument(
        "--drug-embedding-dim", help="Size of drug embedding model", default="512"
    )
    parser.add_argument(
        "--disease-embedding-dim",
        help="Size of disease embedding model",
        default="4096",
    )

    parser.add_argument(
        "--base-dir", help="The directory to store all data", default="data"
    )
    parser.add_argument(
        "--download-dir",
        help="The sub directory to store downloaded files",
        default="download",
    )
    parser.add_argument(
        "--processed-dir",
        help="The sub directory to store processed files",
        default="processed",
    )
    parser.add_argument(
        "--model-dir", help="The sub directory to store models", default="models"
    )

    parser.add_argument(
        "--vectordb-url",
        help="The URL for the vector database",
        default="http://localhost:6333",
    )
    parser.add_argument(
        "--vectordb-api-key", help="The API key for the vector database", default=None
    )

    parser.add_argument(
        "--config", help="Use the configuration file", default="config.yml"
    )

    args = parser.parse_args()
    args = set_args_from_config(args)

    if args.dataset == "ot":
        prepare_ot_data(args)
