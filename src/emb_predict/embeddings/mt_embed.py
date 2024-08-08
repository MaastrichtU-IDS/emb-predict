# ruff: noqa: PLR0913
import os
from tqdm import tqdm
import torch
import numpy as np
from smiles_transformer.load_data import ALPHABET_SIZE, EXTRA_CHARS, download_pretrained
from smiles_transformer.transformer import Transformer, create_masks
from emb_predict.utils import log
from typing import Optional


def encode_char(c):
    return ord(c) - 32


def encode_smiles(string, start_char=EXTRA_CHARS["seq_start"], max_length: int = 256):
    return torch.tensor(
        [ord(start_char)] + [encode_char(c) for c in string], dtype=torch.long
    )[:max_length].unsqueeze(0)


def get_smiles_embeddings(
    smiles_strings: list[str],
    embedding_size: int = 512,
    num_layers: int = 6,
    max_length: int = 256,
    target_folder: str = "data/models/mt/",
    checkpoint_file: str = "pretrained.ckpt",
    mean: bool = True,
    out_file: Optional[str] = None,
):
    checkpoint_filepath = os.path.join(target_folder, checkpoint_file)
    if not os.path.isfile(checkpoint_filepath):
        log.info("Downloading pretrained model...")
        download_pretrained(target_folder=target_folder)

    log.debug("Initializing Transformer...")
    try:
        model = Transformer(ALPHABET_SIZE, embedding_size, num_layers).eval()
        model = torch.nn.DataParallel(model)
    except Exception as e:
        log.error("Error initializing Transformer model", e)
        return

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.debug(f"Using device: {torch_device}")

    log.debug("Loading pretrained weights from", checkpoint_filepath)
    from numpy._core import multiarray

    torch.serialization.add_safe_globals([multiarray.scalar])
    checkpoint = torch.load(
        checkpoint_filepath, map_location=torch_device, weights_only=False
    )  # will need to change this to True
    model.load_state_dict(checkpoint["state_dict"])

    log.debug("Pretrained weights loaded")
    model = model.module.cuda() if torch.cuda.is_available() else model.module.cpu()
    encoder = model.encoder.cuda() if torch.cuda.is_available() else model.encoder.cpu()

    embeddings = []
    with torch.no_grad():
        for smiles in tqdm(smiles_strings, desc="Computing SMILES embeddings"):
            encoded = encode_smiles(smiles, max_length=max_length)
            encoded = encoded.cuda() if torch.cuda.is_available() else encoded.cpu()
            mask = create_masks(encoded, device=torch_device)
            embedding = encoder(encoded, mask)[0].cpu()
            embeddings.append(embedding.numpy())
            log.debug(f"embedded {smiles} into {embedding.shape!s} matrix.")

    if mean:
        # embeddings output to 31 x 512, this means the 31 layers in 1 embeddings of size 512
        embeddings = np.stack([emb.mean(axis=0) for emb in embeddings]).tolist()

    log.info(
        f" {len(embeddings)} SMILES string embedded with dimension of {len(embeddings[0])}"
    )

    out_dict = dict(zip(smiles_strings, embeddings))
    if out_file:
        np.savez(out_file, **out_dict)
        log.info(f"Saved embeddings to {out_file}")
    return out_dict
