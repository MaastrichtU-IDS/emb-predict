from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import ollama
from typing import Sequence
from fastembed import TextEmbedding
from emb_predict.utils import log


def get_ollama_embedding(text, model):
    embeddings = ollama.embeddings(model=model, prompt=text)
    return embeddings["embedding"]


def get_text_embedding(text, model):
    documents = [text]
    embeddings = list(model.embed(documents))
    sequence_of_floats: Sequence[float] = embeddings[0].tolist()
    return sequence_of_floats


def compute_disease_embeddings(model: str, items: list):
    embedding_fnc = None
    embeddings = []

    if model in ["llama3.1", "medllama"]:
        embedding_fnc = get_ollama_embedding
    else:
        log.info(f"Using default text embedding model: {model}")
        model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        providers = model.model.model.get_providers()

        if "CUDAExecutionProvider" in providers:
            log.info("Using GPU for text embedding")
            for item in tqdm(items, desc="Generating Disease Embeddings"):
                embedding = get_text_embedding(item, model)
                row = {"item": item, "embedding": embedding}
                embeddings.append(row)
            return pd.DataFrame(embeddings)

    # otherwise we process with cpu / service
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(embedding_fnc, item, model): item for item in items}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating Disease Embeddings",
        ):
            item = futures[future]
            try:
                embedding = future.result()
                if embedding is not None:
                    row = {"item": item, "embedding": embedding}
                    embeddings.append(row)
                else:
                    print(f"Failed to process text: {item}")
            except Exception as e:
                print(f"Error processing text '{item}': {e}")

    embeddings_df = pd.DataFrame(embeddings)
    return embeddings_df
