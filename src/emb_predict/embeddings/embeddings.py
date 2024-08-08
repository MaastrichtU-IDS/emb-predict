# ruff: noqa: E402
import pandas as pd
import numpy as np
from typing import Optional
from emb_predict.utils import log


def get_embeddings_filename(
    path: str = "",
    dataset: Optional[str] = None,
    mol_type: Optional[str] = None,
    model: Optional[str] = None,
    dim: Optional[int] = None,
):
    return f"{path}/{dataset}_{mol_type}_{model}_{dim}.csv"


def load_embeddings(file_path: str):
    try:
        df = pd.read_csv(f"{file_path}", keep_default_na=False)
    except Exception as e:
        log.error(f"Error loading drug data: {e}")
        return pd.DataFrame()
    df["embedding"] = df["embedding"].apply(
        lambda x: x[1:-1].split(",")
    )  # replace string to list
    df["embedding"] = df["embedding"].apply(lambda x: [float(element) for element in x])
    return df


def load_csv2pd(file_path: str):
    try:
        df = pd.read_csv(f"{file_path}")
    except Exception as e:
        log.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()
    return df


from sklearn.random_projection import GaussianRandomProjection


def reduce_embedding_size(embedding, target_dimension=512):
    transformer = GaussianRandomProjection(n_components=target_dimension)
    reduced_embedding = transformer.fit_transform(np.array(embedding).reshape(1, -1))
    return reduced_embedding.flatten()
