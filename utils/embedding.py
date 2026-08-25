# -*- coding: utf-8 -*-
"""
====================================================================
  Embeddings AraBERT / AraELECTRA
  Reproduit arabert_embeddings_csv_xlsx.py et embedding_fentrage_token
  - Calcul à la volée (sliding window, mean pooling, L2-norm)
  - OU chargement d'un CSV pré-calculé (colonnes dim_0..dim_N)
====================================================================

Installation requise (sur la machine) :
    pip install transformers torch
    pip install arabert   (pour AraBERT uniquement)
"""

import os
import re
import numpy as np
import pandas as pd


MODELES = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "araelectra": "aubmindlab/araelectra-base-discriminator",
}

CHUNK_SIZE = 510
OVERLAP = 64

_TOKENIZER = None
_MODEL = None
_MODEL_NAME = None


def natural_key(s):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', str(s))]


def transformers_disponible():
    try:
        import transformers  # noqa
        import torch  # noqa
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────
# Normalisation arabe (selon le modèle)
# ──────────────────────────────────────────────────────────────────
def normalize_arabic_araelectra(text):
    if not text or not text.strip():
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'[ىئ]', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s\d]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_arabic_arabert(text, arabert_prep=None):
    if not text or not text.strip():
        return ""
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s\d]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if arabert_prep is not None:
        text = arabert_prep.preprocess(text)
    return text


# ──────────────────────────────────────────────────────────────────
# Chargement du modèle
# ──────────────────────────────────────────────────────────────────
def charger_modele_embedding(type_embed, callback_log=None):
    global _TOKENIZER, _MODEL, _MODEL_NAME
    if callback_log is None:
        callback_log = print

    model_name = MODELES[type_embed]
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _TOKENIZER, _MODEL

    import torch
    from transformers import AutoTokenizer, AutoModel

    callback_log(f"[EMBEDDING] Chargement de {model_name} (téléchargement au 1er lancement)...")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = _MODEL.to(device)
    _MODEL.eval()
    _MODEL_NAME = model_name
    callback_log(f"[EMBEDDING] Modèle chargé sur {device}.")
    return _TOKENIZER, _MODEL


# ──────────────────────────────────────────────────────────────────
# Sliding window embedding (reproduit tes scripts)
# ──────────────────────────────────────────────────────────────────
def _mean_pooling(token_embeddings, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).float()
    return (torch.sum(token_embeddings * mask, dim=1) /
            torch.clamp(mask.sum(dim=1), min=1e-9)).squeeze(0).cpu().numpy()


def get_embedding(text, tokenizer, model, output_dim):
    import torch
    device = next(model.parameters()).device

    all_tokens = tokenizer(text, return_tensors="pt", truncation=False,
                           padding=False, add_special_tokens=False)
    input_ids = all_tokens["input_ids"][0]
    total_tokens = len(input_ids)
    if total_tokens == 0:
        return np.zeros(output_dim, dtype=np.float32)

    step = CHUNK_SIZE - OVERLAP
    chunk_vectors = []
    cls_id = torch.tensor([[tokenizer.cls_token_id]])
    sep_id = torch.tensor([[tokenizer.sep_token_id]])

    for start in range(0, total_tokens, step):
        end = min(start + CHUNK_SIZE, total_tokens)
        chunk_content = input_ids[start:end].unsqueeze(0)
        chunk_ids = torch.cat([cls_id, chunk_content, sep_id], dim=1).to(device)
        chunk_mask = torch.ones_like(chunk_ids).to(device)
        with torch.no_grad():
            outputs = model(input_ids=chunk_ids, attention_mask=chunk_mask)
        vec = _mean_pooling(outputs.last_hidden_state, chunk_mask)
        chunk_vectors.append(vec)
        if end == total_tokens:
            break

    vector = np.mean(chunk_vectors, axis=0)
    norm = np.linalg.norm(vector)
    vector = vector / norm if norm > 0 else vector
    return vector[:output_dim].astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# Caractérisation embeddings — calcul à la volée sur le corpus
# ──────────────────────────────────────────────────────────────────
def caracteriser_embedding_corpus(corpus_dir, type_embed, output_dim, callback_log=None):
    """
    Calcule les embeddings de tous les fichiers du corpus.
    Renvoie X (n_docs, output_dim), labels, doc_ids.
    """
    if callback_log is None:
        callback_log = print

    if not transformers_disponible():
        raise RuntimeError(
            "transformers/torch non installés. "
            "Installe : pip install transformers torch (et arabert pour AraBERT)."
        )

    tokenizer, model = charger_modele_embedding(type_embed, callback_log=callback_log)

    # Préprocesseur AraBERT si besoin
    arabert_prep = None
    if type_embed == "arabert":
        try:
            from arabert.preprocess import ArabertPreprocessor
            arabert_prep = ArabertPreprocessor(model_name=MODELES["arabert"])
        except ImportError:
            callback_log("[WARN] arabert non installé, préprocessing simplifié.")

    sous_dossiers = sorted(
        [d for d in os.listdir(corpus_dir) if os.path.isdir(os.path.join(corpus_dir, d))],
        key=natural_key)

    X_list, labels, doc_ids = [], [], []
    for auteur in sous_dossiers:
        adir = os.path.join(corpus_dir, auteur)
        fichiers = sorted([f for f in os.listdir(adir) if f.endswith(".txt")], key=natural_key)
        for fname in fichiers:
            with open(os.path.join(adir, fname), encoding="utf-8") as f:
                raw = f.read()
            if type_embed == "arabert":
                texte = normalize_arabic_arabert(raw, arabert_prep)
            else:
                texte = normalize_arabic_araelectra(raw)
            if not texte:
                continue
            vec = get_embedding(texte, tokenizer, model, output_dim)
            X_list.append(vec)
            labels.append(auteur)
            doc_ids.append(f"{auteur} || {fname}")
            callback_log(f"   {auteur}/{fname} -> dim {vec.shape[0]}")

    X = np.array(X_list, dtype=np.float32)
    callback_log(f"   X embeddings : {X.shape}")
    return X, np.array(labels), doc_ids


def vectoriser_texte_embedding(texte, type_embed, output_dim, callback_log=None):
    """Embedding d'un texte nouveau."""
    if callback_log is None:
        callback_log = print
    tokenizer, model = charger_modele_embedding(type_embed, callback_log=callback_log)
    arabert_prep = None
    if type_embed == "arabert":
        try:
            from arabert.preprocess import ArabertPreprocessor
            arabert_prep = ArabertPreprocessor(model_name=MODELES["arabert"])
        except ImportError:
            pass
    if type_embed == "arabert":
        texte = normalize_arabic_arabert(texte, arabert_prep)
    else:
        texte = normalize_arabic_araelectra(texte)
    return get_embedding(texte, tokenizer, model, output_dim)


# ──────────────────────────────────────────────────────────────────
# Chargement depuis CSV pré-calculé
# ──────────────────────────────────────────────────────────────────
def charger_embedding_csv(chemin_csv, output_dim=None, callback_log=None):
    """
    Charge un CSV d'embeddings pré-calculés.
    Format attendu : colonnes 'locuteur', 'fichier', 'dim_0', 'dim_1', ...
    Renvoie X (n_docs, dim), labels, doc_ids.
    """
    if callback_log is None:
        callback_log = print

    df = pd.read_csv(chemin_csv)
    df.columns = df.columns.str.strip()

    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if output_dim and output_dim < len(dim_cols):
        dim_cols = dim_cols[:output_dim]

    X = df[dim_cols].values.astype(np.float32)

    # Colonne auteur (locuteur ou Auteur)
    col_auteur = "locuteur" if "locuteur" in df.columns else df.columns[0]
    labels = df[col_auteur].astype(str).values
    col_fichier = "fichier" if "fichier" in df.columns else None
    if col_fichier:
        doc_ids = [f"{a} || {f}" for a, f in zip(labels, df[col_fichier])]
    else:
        doc_ids = [f"{a} || doc{i}" for i, a in enumerate(labels)]

    callback_log(f"   CSV chargé : {X.shape} | {len(set(labels))} auteurs")
    return X, np.array(labels), doc_ids
