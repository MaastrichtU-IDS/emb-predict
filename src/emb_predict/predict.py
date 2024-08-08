# ruff: noqa: PLW2901
# ruff: noqa: PLR5501
# ruff: noqa: PLR0912
# ruff: noqa: PLR0915
# ruff: noqa: S324
# ruff: noqa: PLW0602
import pandas as pd
import torch
import cupy
import hashlib
import pprint
import re
from rdkit import Chem
from trapi_predict_kit import PredictInput, PredictOutput, PredictOptions, trapi_predict

from emb_predict.utils import log
from emb_predict.embeddings.mt_embed import get_smiles_embeddings
from emb_predict.embeddings.llm_embed import (
    compute_disease_embeddings,
)
from emb_predict.train import normalize_dfs
from emb_predict.application import (
    set_project_paths,
    pre_loaded_data,
    load_application_components,
)
from emb_predict.vectordb.vectordb import QdrantDB
from emb_predict.application import get_latest_version_ot_processed_dir


def get_drug_data(
    drug_list: list[str],
    db: QdrantDB,
    main_drug_collection: str,
    user_drug_collection: str,
    mt_dir: str,
) -> pd.DataFrame:
    drug_df = pd.DataFrame()
    for subject in drug_list:
        # check if we have a curie or a label
        results = None
        drug_name = ""
        smiles = None

        if "CHEMBL" in subject and "_" not in subject:
            drug_id = re.sub(r"(CHEMBL)(\d+)", r"\1_\2", subject)
            subject = drug_id

        # search in the vector database
        log.info(f"Searching {main_drug_collection} with {subject}")
        results = db.get(
            collection_name=main_drug_collection,
            search_input=subject,
            search_fields=["id", "name", "smiles"],
            exact_match=True,
            limit=1,
        )

        if len(results) > 0:
            log.info("Found " + str(len(results)) + " in vectordb")
        else:
            # check if we have it in the user contributed database
            log.info(f"Searching {user_drug_collection} with {subject}")
            results = db.get(
                collection_name=user_drug_collection,
                search_input=subject,
                search_fields=["id", "name", "smiles"],
                exact_match=True,
                limit=1,
            )
            if len(results) > 0:
                log.info("Found " + str(len(results)) + " in vectordb")
            else:
                # let's do more complicated stuff
                if ":" in subject or "_" in subject:
                    # call the translator service to get other ids
                    log.info(f"search XXX to resolve {subject}")

                    log.error(f"Unable to get SMILES string for {subject}")

                else:
                    # see if the input is a smiles string
                    try:
                        log.info(f"Checking if {subject} is a smiles string")
                        mol = Chem.MolFromSmiles(subject)
                        if mol is not None:
                            smiles = Chem.MolToSmiles(
                                mol
                            )  # we have a valid smiles to search or compute embedding
                            log.info("Looks like we have a smiles string")
                        else:
                            log.error("Not a valid SMILES string")
                    except Exception:
                        log.error("Not a valid SMILES string")

        if results is not None and len(results) > 0:
            log.info(f"Found {subject} in vectordb")
            row = {}
            row["id"] = results[0].payload["id"]
            row["name"] = results[0].payload["name"]
            row["smiles"] = results[0].payload["smiles"]
            vector = results[0].payload["embedding"]
            row["embedding"] = [float(x) for x in vector[1:-1].split(",")]

            df = pd.DataFrame.from_dict([row])
            drug_df = pd.concat([drug_df, df], ignore_index=True)

        elif smiles is not None:
            try:
                log.info(f"Getting embedding for {smiles}")

                drug_id = "embpredict:" + hashlib.md5(smiles.encode()).hexdigest()
                drug_name = smiles
                drug_embedding = get_smiles_embeddings([smiles], target_folder=mt_dir)
                vector = drug_embedding[smiles]
                string_list = [str(f) for f in vector]
                str_vector = ",".join(string_list)
                try:
                    log.info("Adding smiles to user database")
                    item = {
                        "vector": vector,
                        "payload": {
                            "id": drug_id,
                            "name": drug_name,
                            "smiles": smiles,
                            "embedding": str_vector,
                        },
                    }
                    db.add(collection_name=user_drug_collection, item_list=[item])
                except Exception as e:
                    log.error(f"Error adding smiles to user database: {e}")

            except Exception as e:
                log.error(f"Error processing smiles '{smiles}': {e}")
                continue

            row = {}
            row["id"] = drug_id
            row["name"] = drug_name
            row["smiles"] = smiles
            row["embedding"] = vector
            df = pd.DataFrame.from_dict([row])
            drug_df = pd.concat([drug_df, df], ignore_index=True)
    return drug_df


def get_disease_data(
    disease_list: list[str],
    db: QdrantDB,
    main_disease_collection: str,
    user_disease_collection: str,
    disease_embedding_model: str,
) -> pd.DataFrame:
    disease_df = pd.DataFrame()
    for subject in disease_list:
        disease_id = disease_name = results = None

        log.info(f"Searching {main_disease_collection} with {subject}")
        results = db.get(
            collection_name=main_disease_collection,
            search_input=subject,
            search_fields=["id", "name"],
            exact_match=True,
            limit=1,
        )

        if len(results) > 0:
            log.info("Found " + str(len(results)) + " in vectordb")

        else:
            # try searching the user contributed database
            log.info(f"Searching {user_disease_collection} with {subject}")
            results = db.get(
                collection_name=user_disease_collection,
                search_input=subject,
                search_fields=["id", "name"],
                exact_match=True,
                limit=1,
            )
            if len(results) > 0:
                log.info("Found " + str(len(results)) + " in vectordb")

            else:
                # check if we have a curie or a label
                if ":" in subject:
                    # do a lookup in the database and get the label
                    log.info(f"Searching for disease id with {subject}")
                    log.info("This functionality is not yet implemented")
                else:
                    # assume disease name
                    disease_name = subject

        if disease_name is not None:
            try:
                log.info(f"generating embedding for: {disease_name}")
                embeddings_df = compute_disease_embeddings(
                    disease_embedding_model, [disease_name]
                )
                disease_embedding = embeddings_df["embedding"].values[0]

                log.info("Got embedding.")
                if disease_embedding is not None:
                    disease_id = (
                        disease_id
                        if disease_id is not None
                        else "embpredict:"
                        + hashlib.md5(disease_name.encode()).hexdigest()
                    )
                    row = {}
                    row["id"] = disease_id
                    row["name"] = disease_name
                    row["embedding"] = disease_embedding
                    df = pd.DataFrame.from_dict([row])
                    disease_df = pd.concat([disease_df, df], ignore_index=True)

                    # add to user contributed database
                    try:
                        log.info(f"Adding {disease_name} to user disease database")
                        string_list = [str(f) for f in disease_embedding]
                        str_vector = ",".join(string_list)

                        item = {
                            "vector": disease_embedding,
                            "payload": {
                                "id": disease_id,
                                "name": disease_name,
                                "embedding": str_vector,
                            },
                        }
                        db.add(
                            collection_name=user_disease_collection, item_list=[item]
                        )
                    except Exception as e:
                        log.error(
                            f"Error adding  {disease_name} to user disease database: {e}"
                        )

            except Exception as e:
                log.error(
                    f"Error getting embedding from {disease_embedding_model} with '{disease_name}': {e}"
                )

        if results:
            row = {}
            row["id"] = results[0].payload["id"]
            row["name"] = results[0].payload["name"]
            # row['embedding'] = results[0].vector   # a list of floats
            vector = results[0].payload["embedding"]
            row["embedding"] = [float(x) for x in vector[1:-1].split(",")]

            df = pd.DataFrame.from_dict([row])
            disease_df = pd.concat([disease_df, df], ignore_index=True)

    return disease_df


@trapi_predict(
    path="/predict",
    name="Predict drug-disease relationships",
    description="Return the predicted targets for a given entity: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.",
    edges=[
        {
            "subject": "biolink:Drug",
            "predicate": "biolink:treats",
            "object": "biolink:Disease",
        },
        {
            "subject": "biolink:Disease",
            "predicate": "biolink:treated_by",
            "object": "biolink:Drug",
        },
    ],
    nodes={
        "biolink:Drug": {"id_prefixes": ["CHEMBL"]},
        "biolink:Disease": {"id_prefixes": ["HPO", "EFO", "MONDO"]},
    },
)
def get_predictions(request: PredictInput) -> PredictOutput:
    # fail if no input is provided
    try:
        if len(request.subjects) == 0 and len(request.objects) == 0:
            raise Exception("No input provided")
    except Exception as e:
        log.error(f"Error processing input: {e}")
        return {"hits": [], "count": 0}

    # @todo add options to the prediction api form
    global pre_loaded_data
    if "paths" not in pre_loaded_data:
        paths = set_project_paths()
        pre_loaded_data["paths"] = paths
    else:
        paths = pre_loaded_data["paths"]

    if "vectordb" not in pre_loaded_data:
        load_application_components(paths)

    # log.info(f"Loaded application components {pre_loaded_data}" )
    db = pre_loaded_data["vectordb"]
    drug_df = pre_loaded_data["drug_df"]
    disease_df = pre_loaded_data["disease_df"]
    drug_disease_df = pre_loaded_data["drug_disease_df"]

    vectordb_drug_collection = paths["vectordb_drug_collection"]
    vectordb_user_drug_collection = paths["vectordb_user_drug_collection"]
    vectordb_disease_collection = paths["vectordb_disease_collection"]
    vectordb_user_disease_collection = paths["vectordb_user_disease_collection"]
    paths["drug_embedding_model"]
    paths["drug_embedding_dim"]
    disease_embedding_model = paths["disease_embedding_model"]

    # get the embeddings for the input, which can be a CHEMBL id, a curie, a name, or a SMILES string
    drug_df = get_drug_data(
        request.subjects,
        db,
        vectordb_drug_collection,
        vectordb_user_drug_collection,
        paths["mt_dir"],
    )
    disease_df = get_disease_data(
        request.objects,
        db,
        vectordb_disease_collection,
        vectordb_user_disease_collection,
        disease_embedding_model,
    )

    # if we didn't get the embeddings for the specified pairs, abort with error
    if len(request.subjects) > 0 and len(drug_df) == 0:
        log.error("Unable to get embeddings for the specified drugs")
        return {"hits": [], "count": 0}
    if len(request.objects) > 0 and len(disease_df) == 0:
        log.error("Unable to get embeddings for the specified diseases")
        return {"hits": [], "count": 0}

    # if no subjects or objects are provided, get the full dataframes
    if len(request.subjects) == 0:
        drug_df = pre_loaded_data["drug_df"]
        ### limit to only drugs in the known drug-disease set
        drug_df = drug_df[drug_df["id"].isin(drug_disease_df["drug_id"])]

        log.info(f"Will scan against {len(drug_df)} drugs.")
    if len(request.objects) == 0:
        disease_df = pre_loaded_data["disease_df"]
        log.info(f"Will scan against {len(disease_df)} diseases.")

    # create the prediction pairs
    log.info("Creating prediction pairs")
    df_a_t = drug_df.drop(columns=["name", "smiles", "embedding"])
    df_b_t = disease_df.drop(columns=["name", "embedding"])
    df_ab = pd.merge(df_a_t, df_b_t, how="cross")

    log.info("Normalizing dataframes")
    df_ab, df_a, df_b = normalize_dfs(
        df_ab=df_ab,
        df_a=drug_df,
        df_b=disease_df,
        df_ab_a_key="id_x",
        df_ab_b_key="id_y",
        df_a_key="id",
        df_b_key="id",
    )

    # prepare input for model
    predict_df = df_ab.merge(df_a, left_on="a", right_on="a").merge(
        df_b, left_on="b", right_on="b"
    )
    features_cols = predict_df.columns.difference(["a", "b"])
    x = predict_df[features_cols].values

    try:
        if torch.cuda.is_available():
            x = cupy.array(x)

        log.info(f"Making predictions for {len(x)} pairs")
        model = pre_loaded_data["model"]
        y_pred = model.predict_proba(x)
        log.info(f"{len(y_pred)} predictions made.")
    except Exception as e:
        log.error(f"Error predicting: {e}")
        return {"hits": [], "count": 0}

    ##### prepare output
    # add the score to the pair list
    df_ab["score"] = y_pred[:, 1]

    # add the labels
    df_ab = df_ab.merge(
        drug_df.drop(columns=["smiles", "embedding"]), left_on="a", right_on="id"
    ).merge(disease_df.drop(columns=["embedding"]), left_on="b", right_on="id")
    # add whether it is known pair
    df_ab = df_ab.merge(
        drug_disease_df,
        how="left",
        left_on=["a", "b"],
        right_on=["drug_id", "disease_id"],
        indicator=True,
    )
    df_ab["is_known"] = df_ab["_merge"] == "both"

    # apply the options filters
    log.info(request.options)
    options = request.options
    if options.min_score is None:
        options.min_score = 0.0
    if options.max_score is None:
        options.max_score = 1.0

    df_ab.loc[df_ab["score"] < float(options.min_score), "score"] = 0.0
    df_ab.loc[df_ab["score"] > float(options.max_score), "score"] = 0.0
    df_ab.sort_values(by="score", ascending=False, inplace=True)

    df_ab = df_ab.reset_index(drop=True)
    df_ab.loc[options.n_results :, "score"] = 0.0

    df_ab.rename(
        columns={
            "a": "subject",
            "b": "object",
            "name_x": "subject_label",
            "name_y": "object_label",
        },
        inplace=True,
    )
    score_df = df_ab[df_ab["score"] > 0]
    score_df = score_df[
        ["subject", "object", "subject_label", "object_label", "score", "is_known"]
    ]
    scores_list = score_df.to_dict(orient="records")
    return {"hits": scores_list, "count": len(scores_list)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="predictModel", description="Use a trained model to predict."
    )

    parser.add_argument(
        "--dataset", help="Dataset to use", choices=["ot"], default="ot"
    )
    parser.add_argument(
        "--dataset_version", help="Version of the dataset", default="latest"
    )
    parser.add_argument(
        "--drug-embedding-model", help="Drug embedding model", default="mt"
    )
    parser.add_argument(
        "--disease-embedding-model", help="Disease embedding model", default="llama3.1"
    )
    parser.add_argument(
        "--drug-embedding-dim", help="Drug embedding dimensions", default="512"
    )
    parser.add_argument(
        "--disease-embedding-dim", help="Disease embedding dimensions", default="4096"
    )

    # vectordb parameters
    parser.add_argument(
        "--vectordb-url",
        help="The URL for the vector database",
        default="http://localhost:6333",
    )
    parser.add_argument(
        "--vectordb-api-key", help="The API key for the vector database", default=None
    )

    # training parameters
    parser.add_argument(
        "--classifiers",
        help="The classifiers to use, comma-separated, with any of: xgb,lr,rf,nb",
        nargs="+",
        default="xgb",
    )
    parser.add_argument("--n_runs", help="Number of runs", default=1)
    parser.add_argument(
        "--n_proportion", help="The proportions of true to false pairs", default=1
    )
    parser.add_argument("--n_splits", help="Number of splits", default=10)
    parser.add_argument("--n_folds", help="Number of folds", default=None)

    # paths
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
        "--training-dir",
        help="The sub directory to store processed files",
        default="training",
    )
    parser.add_argument(
        "--model-dir", help="The sub directory to store models", default="models"
    )

    parser.add_argument(
        "--model-file", help="The model file for prediction", default="model.ubj"
    )  # "CHEMBL_4559134",
    # parser.add_argument('--subjects', help='The subjects to predict', nargs='*', default=["CCCCO"])
    parser.add_argument(
        "--subjects", help="The subjects to predict", nargs="*", default=[]
    )
    parser.add_argument(
        "--objects", help="The objects to predict", nargs="*", default=["obesity"]
    )  # "MONDO_0004992",
    parser.add_argument(
        "--n-results", help="The number of results to return", default=10
    )
    parser.add_argument("--min-score", help="The minimum score to return", default=0.0)
    parser.add_argument("--max-score", help="The maximum score to return", default=1.0)

    parser.add_argument(
        "--config", help="Use the configuration file", default="config.yml"
    )

    args = parser.parse_args()
    if args.dataset == "ot" and args.dataset_version == "latest":
        args.dataset_version = get_latest_version_ot_processed_dir(
            f"{args.base_dir}/{args.processed_dir}/{args.dataset}"
        )

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
        config_file=args.config,
    )
    pre_loaded_data["paths"] = paths

    pi = PredictInput()
    if "subjects" in args:
        pi.subjects = args.subjects
    if "objects" in args:
        pi.objects = args.objects

    options = PredictOptions()
    options.n_results = args.n_results
    options.min_score = args.min_score
    options.max_score = args.max_score
    pi.options = options

    predictions = get_predictions(pi)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(predictions)
