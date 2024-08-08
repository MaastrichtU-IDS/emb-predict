# ruff: noqa: S605
# ruff: noqa: PLR0913
import os
from datetime import date
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import cupy as cp
import pickle

from emb_predict.application import (
    set_project_paths,
    get_latest_version_ot_processed_dir,
)

from emb_predict.embeddings.embeddings import load_csv2pd, load_embeddings
from emb_predict.utils import log
from typing import Optional


training_models_dir = None


def set_training_models_dir(models_dir: str):
    globals()["training_models_dir"] = models_dir
    os.makedirs(dir, exist_ok=True)


def get_training_models_dir():
    return training_models_dir


def get_best_model(path: str, metric: str = "f1"):
    files = os.listdir(path)
    best_model = None
    best_score = 0
    best_row = None
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(f"{path}/{file}")
            score = df[metric].max()
            if score > best_score:
                best_score = score
                best_model = file
                best_row = df.loc[df[metric] == score]

    best_model = (
        str(best_row["method"].values[0])
        + "_"
        + str(best_row["run"].values[0])
        + "_"
        + str(best_row["fold"].values[0])
        + ".ubj"
    )
    return best_model


def copy_best_model_to_dir(results_dir: str, model_dir: str, metric: str = "f1"):
    best_model = get_best_model(results_dir, metric)
    if best_model is not None:
        os.system(
            f"cp {results_dir}/results/{best_model} {model_dir}/ot_xgb_mt_512_llama3.1_4096.ubj"
        )


def normalize_dfs(
    df_ab, df_a, df_b, df_ab_a_key: str, df_ab_b_key: str, df_a_key: str, df_b_key: str
):
    df_ab = df_ab.rename(columns={df_ab_a_key: "a", df_ab_b_key: "b"}).reset_index(
        drop=True
    )
    df_a = df_a.rename(columns={df_a_key: "a"}).reset_index(drop=True)
    df_b = df_b.rename(columns={df_b_key: "b"}).reset_index(drop=True)

    d = pd.merge(
        df_ab, df_a, left_on="a", right_on="a", how="inner", suffixes=("_a", "_b")
    )
    d = pd.merge(d, df_b, left_on="b", right_on="b", how="inner", suffixes=("_a", "_b"))

    df_ab = d[["a", "b"]]
    df_a = d[["a", "embedding_a"]].drop_duplicates(subset=["a"])
    df_b = d[["b", "embedding_b"]].drop_duplicates(subset=["b"])
    df_a = df_a.reset_index(drop=True)
    df_b = df_b.reset_index(drop=True)

    ma = []
    for e in df_a["embedding_a"]:
        ma.append(np.array(e))
    np_array = np.array(ma, dtype=np.float32)
    t = pd.DataFrame(np_array.T, columns=[f"{i}" for i in range(np_array.shape[0])])
    t = t.T.reset_index(drop=True)
    df_a = pd.concat([df_a, t], axis=1)
    df_a.drop(columns=["embedding_a"], inplace=True)

    ma = []
    for e in df_b["embedding_b"]:
        ma.append(np.array(e))
    np_array = np.array(ma, dtype=np.float32)
    t = pd.DataFrame(np_array.T, columns=[f"c{i}" for i in range(np_array.shape[0])])
    t = t.T.reset_index(drop=True)
    df_b = pd.concat([df_b, t], axis=1)
    df_b.drop(columns=["embedding_b"], inplace=True)

    df_a.columns = [
        str(col) + "_a" if col not in ["a", "b", "Class"] else col
        for col in df_a.columns
    ]
    df_b.columns = [
        str(col) + "_b" if col not in ["a", "b", "Class"] else col
        for col in df_b.columns
    ]

    return df_ab, df_a, df_b


def generate_pairs(ab_df):
    """Get pairs and their labels: All given known ab pairs are 1,
    We add pairs for missing a/b combinations as 0 (not known as interacting)"""
    ab_known = {tuple(x) for x in ab_df[["a", "b"]].values}
    pairs = []
    labels = []

    a = set(ab_df.a.unique())
    b = set(ab_df.b.unique())
    for _a in a:
        for _b in b:
            label = 1 if (_a, _b) in ab_known else 0
            pairs.append((_a, _b))
            labels.append(label)

    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels


def cross_validation(train_df, test_df, clfs, run_index, fold_index):
    features_cols = train_df.columns.difference(["a", "b", "Class"])
    ab_df = test_df[["a", "b"]].copy()

    x_train = train_df[features_cols].values
    x_true = train_df["Class"].values.ravel()

    y_test = test_df[features_cols].values
    y_true = test_df["Class"].values.ravel()

    results = pd.DataFrame()
    for name, model in clfs:
        model.fit(x_train, x_true)

        row = {}
        row["run"] = run_index
        row["fold"] = fold_index
        row["method"] = name

        # y_pred = clf.predict(y)
        if torch.cuda.is_available() and name == "xgb":
            y_test = cp.array(y_test)

        z = model.predict_proba(y_test)
        y_pred = np.argmax(
            z, axis=1
        )  # get the indice [0 or 1] with the maximum value #numpy.ndarray
        classes = model.classes_  # the classes to predict [0,1]
        y_pred = [
            classes[i] for i in y_pred
        ]  # make a list with the classes for each entry
        scores = {
            "roc_auc": float(roc_auc_score(y_true, y_pred)),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }
        row.update(scores)

        training_models_dir = get_training_models_dir()
        y_true_df = pd.DataFrame(y_true, columns=["true_class"])
        ab_df = pd.concat([ab_df, y_true_df], axis=1)
        y_pred_df = pd.DataFrame(y_pred, columns=["predicted_class"])
        ab_df = pd.concat([ab_df, y_pred_df], axis=1)
        new_df = pd.DataFrame(z, columns=["0_prob", "1_prob"])
        ab_df = pd.concat([ab_df, new_df], axis=1)

        training_results_file = (
            f"{training_models_dir}/{name}_{run_index}_{fold_index}_predictions.csv"
        )
        ab_df.to_csv(training_results_file, sep=",", index=False)

        model_file = f"{training_models_dir}/{name}_{run_index}_{fold_index}"

        if name == "xgb":
            model.save_model(f"{model_file}.ubj")
        else:
            with open(f"{model_file}.pkl", "wb") as f:
                pickle.dump(model, f)

        row["classifier"] = model_file

        df = pd.DataFrame.from_dict([row])
        results = pd.concat([results, df], ignore_index=True)

    return results


def cv_run(run_index, pairs, classes, embedding_df, train, test, fold_index, clfs):
    # print( f"Run: {run_index} Fold: {fold_index} Train size: {len(train)} Test size: {len(test)}")
    train_df = pd.DataFrame(
        list(zip(pairs[train, 0], pairs[train, 1], classes[train])),
        columns=["a", "b", "Class"],
    )
    test_df = pd.DataFrame(
        list(zip(pairs[test, 0], pairs[test, 1], classes[test])),
        columns=["a", "b", "Class"],
    )

    train_df = train_df.merge(embedding_df["a"], left_on="a", right_on="a").merge(
        embedding_df["b"], left_on="b", right_on="b"
    )
    test_df = test_df.merge(embedding_df["a"], left_on="a", right_on="a").merge(
        embedding_df["b"], left_on="b", right_on="b"
    )

    cv_results = cross_validation(train_df, test_df, clfs, run_index, fold_index)
    cv_results["empty_df"] = None
    cv_results.at[0, "empty_df"] = test_df.iloc[0:0].copy()
    return cv_results


def balance_data(pairs, classes, n_proportion):
    indices_true = np.where(classes == 1)[0]
    indices_false = np.where(classes == 0)[0]

    np.random.shuffle(indices_false)
    indices_false_proportioned = indices_false[: (n_proportion * indices_true.shape[0])]

    pairs = np.concatenate(
        (pairs[indices_true], pairs[indices_false_proportioned]), axis=0
    )
    classes = np.concatenate(
        (classes[indices_true], classes[indices_false_proportioned]), axis=0
    )

    return pairs, classes


def kfold_cv(
    pairs_all,
    classes_all,
    embedding_df,
    classifiers,
    n_runs,
    n_splits,
    n_proportion,
    random_seed,
    n_folds,
):
    all_runs_df = pd.DataFrame()
    for run in range(1, n_runs + 1):
        random_seed += run
        random.seed(random_seed)
        np.random.seed(random_seed)
        pairs, classes = balance_data(pairs_all, classes_all, n_proportion)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        cv = skf.split(pairs, classes)
        folds = [(train, test, (k + 1)) for k, (train, test) in enumerate(cv)]

        all_cv_df = pd.DataFrame()
        for train, test, k in folds:
            cv_results = cv_run(
                run, pairs, classes, embedding_df, train, test, k, classifiers
            )

            scores = cv_results.drop(columns=["classifier", "empty_df"])
            if run == 1 and k == 1:
                print(", ".join(scores.columns))
            csv_list = scores.values.tolist()
            flattened_csv_list = [
                ",".join([f"{x:.5f}" if isinstance(x, float) else str(x) for x in row])
                + ""
                for row in csv_list
            ]
            print("".join(flattened_csv_list))

            all_cv_df = pd.concat([all_cv_df, cv_results], ignore_index=True)
            if n_folds is not None and k == n_folds:
                break
        all_runs_df = pd.concat([all_runs_df, all_cv_df], ignore_index=True)
    return all_runs_df


def train(
    df_ab: pd.DataFrame,
    df_a_embeddings: pd.DataFrame,
    df_b_embeddings: pd.DataFrame,
    training_dir: str = "data/training",
    model_dir: str = "data/models",
    project_name: str = "ab",
    a_name: str = "a",
    b_name: str = "b",
    # training parameters
    classifier: list = ["xgb"],
    n_runs: int = 1,
    n_proportion: int = 1,
    n_splits: int = 10,
    n_folds: Optional[int] = None,
    random_seed: int = 100,
    max_depth: int = 6,
):
    """Training takes 3 dataframes as input
    1. a df with known pairs (2 cols: a, b)
    2. a df with a's embeddings: a col + X cols for embeddings
    3. a df with b's embeddings: b col + X cols for embeddings
    """
    embeddings = {
        "a": df_a_embeddings,
        "b": df_b_embeddings,
    }
    today = date.today()
    results_file = f"{training_dir}/{project_name}_scores_{today}.csv"
    agg_results_file = f"{training_dir}/{project_name}_agg_scores_{today}.csv"

    # Get pairs
    pairs, labels = generate_pairs(df_ab)
    na = len(embeddings["a"])
    nb = len(embeddings["b"])
    unique, counts = np.unique(labels, return_counts=True)
    nab = counts[1]
    log.info(
        f"Training based on {nab} known interactions: {na} {a_name} | {nb} {b_name}"
    )

    if torch.cuda.is_available():
        log.info("Using GPU for prediction")
    else:
        log.info("Using CPU for prediction")

    classifiers = []
    if "xgb" in classifier:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            objective="binary:logistic",  # For binary classification
            n_jobs=-1,
            random_state=42,
            tree_method="hist",  # Use GPU optimized histogram algorithm
            device="gpu",
        )
        classifiers.append(("xgb", xgb_model))
    elif "lr" in classifier:
        from sklearn import linear_model

        lr_model = linear_model.LogisticRegression()
        classifiers.append(("lr", lr_model))
    elif "rf" in classifier:
        from sklearn import ensemble

        # rf_model = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
        rf_model = ensemble.RandomForestClassifier(
            n_estimators=200,
            criterion="log_loss",
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
        )
        classifiers.append(("rf", rf_model))
    elif "nb" in classifier:
        from sklearn import naive_bayes

        nb_model = naive_bayes.GaussianNB()
        classifiers.append(("nb", nb_model))

    # Run training
    log.info("Start training")

    # n_folds = 5
    all_runs_df = kfold_cv(
        pairs,
        labels,
        embeddings,
        classifiers,
        n_runs,
        n_splits,
        n_proportion,
        random_seed,
        n_folds,
    )
    log.info("End training")

    scores_df = all_runs_df.drop(columns=["classifier", "empty_df"])
    scores_df.to_csv(results_file, sep=",", index=False)
    agg_df = scores_df.groupby(["method", "run"]).mean().groupby("method").mean()
    agg_df.to_csv(agg_results_file, sep=",", index=False)

    # save the best model
    max_index = all_runs_df["f1"].idxmax()
    max_row = all_runs_df.loc[max_index]

    dir_components = training_dir.split("/")
    dir_list = [item for item in dir_components if item not in [None, "", [], {}]]
    item = dir_list[-1]
    parse = item.split("_")

    best_model = f'{training_dir}/results/{max_row["method"]}_{max_row["run"]}_{max_row["fold"]}.ubj'
    model_file = f'{model_dir}/{project_name}_{max_row["method"]}_{parse[0]}_{parse[1]}_{parse[2]}_{parse[3]}.ubj'
    os.system(f"cp {best_model} {model_file}")
    log.info(f"Best model {best_model} saved to {model_file}")


def train_ot_models(args, paths):
    df_ab = load_csv2pd(paths["dataset_drug_disease_processed_fp"])
    df_b = load_embeddings(paths["disease_embedding_fp"])
    df_a = load_embeddings(paths["drug_embedding_fp"])

    df_ab, df_a, df_b = normalize_dfs(
        df_ab,
        df_a,
        df_b,
        df_ab_a_key="drug_id",
        df_ab_b_key="disease_id",
        df_a_key="id",
        df_b_key="id",
    )

    train(
        df_ab=df_ab,
        df_a_embeddings=df_a,
        df_b_embeddings=df_b,
        project_name=paths["dataset"],
        a_name="drug",
        b_name="disease",
        classifier=args.classifiers,
        n_runs=args.n_runs,
        n_proportion=args.n_proportion,
        n_splits=args.n_splits,
        n_folds=args.n_folds,
        training_dir=paths["training_dir"],
        model_dir=paths["embpredict_model_dir"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="trainModel", description="Train the model.")

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
    set_training_models_dir(paths["training_dir"] + "/results")
    # args.classifiers = [element.strip() for element in args.classifiers.split(',')]

    if args.dataset == "ot":
        train_ot_models(args, paths)
