# -*- coding: utf-8 -*-
"""
====================================================================
  Caractérisation des textes — méthodes multiples
  - N-gram lettre F1/F2  (reproduit 3D_F1/F2_Lettre)
  - N-gram mot F1/F2     (reproduit 3D_F1/F2_Mot)
  - TF-IDF               (reproduit TF_IDF.py)
====================================================================
"""

import os
import re
import unicodedata
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# ──────────────────────────────────────────────────────────────────
# Tri naturel + normalisation
# ──────────────────────────────────────────────────────────────────
def natural_key(s):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', str(s))]


def normaliser_invisibles(texte):
    if not texte:
        return ""
    texte = unicodedata.normalize("NFC", texte)
    if texte.startswith("\ufeff"):
        texte = texte[1:]
    for c in ["\u200b", "\u200c", "\u200d", "\u200e", "\u200f",
              "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
              "\u2066", "\u2067", "\u2068", "\u2069", "\ufeff"]:
        texte = texte.replace(c, "")
    texte = texte.replace("\xa0", " ").replace("\t", " ")
    texte = texte.replace("\r\n", "\n").replace("\r", "\n")
    texte = re.sub(r' +', ' ', texte)
    return texte.strip()


def lire_fichier(chemin):
    for encodage in ("utf-8", "utf-8-sig", "cp1256", "latin-1", "cp1252"):
        try:
            with open(chemin, "r", encoding=encodage) as f:
                contenu = f.read()
            if contenu.strip():
                return normaliser_invisibles(contenu)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    raise ValueError(f"Impossible de lire : {chemin}")


# ──────────────────────────────────────────────────────────────────
# Tokenisation
# ──────────────────────────────────────────────────────────────────
def tokeniser_caracteres(texte):
    """N-gram lettre : on enlève les espaces (comme 3D_F1/F2_Lettre)."""
    texte = re.sub(r'\s+', '', texte, flags=re.UNICODE)
    return list(texte)


def tokeniser_mots(texte):
    """N-gram mot : split sur ponctuation (comme 3D_F1/F2_Mot)."""
    texte = re.sub(r'[^\w\s]', ' ', texte, flags=re.UNICODE)
    return [t for t in texte.split() if t.strip()]


# ──────────────────────────────────────────────────────────────────
# Calcul des fréquences n-gram (F1 ou F2, lettre ou mot)
# ──────────────────────────────────────────────────────────────────
def calculer_frequences_ngram(tokens, n, methode, niveau):
    """
    methode : 'F1' (occ/total) ou 'F2' (occ/uniques)
    niveau  : 'lettre' (join sans espace) ou 'mot' (join avec espace)
    """
    ngrammes = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrammes:
        return {}

    compte = Counter(ngrammes)

    if methode.upper() == "F1":
        denom = len(ngrammes)
    else:  # F2
        denom = len(set(ngrammes))
    if denom == 0:
        return {}

    sep = "" if niveau == "lettre" else " "
    freq = {sep.join(g): c / denom for g, c in compte.items()}
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))


def extraire_vecteur_ngram(texte, n, methode, niveau):
    """Vecteur de fréquences pour UN texte (n-gram)."""
    texte = normaliser_invisibles(texte)
    if niveau == "lettre":
        tokens = tokeniser_caracteres(texte)
    else:
        tokens = tokeniser_mots(texte)
    freq = calculer_frequences_ngram(tokens, n, methode, niveau)
    return [round(f, 6) for f in freq.values()]


# ──────────────────────────────────────────────────────────────────
# Chargement corpus
# ──────────────────────────────────────────────────────────────────
def load_corpus(corpus_dir):
    entries = []
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"Dossier introuvable : {corpus_dir}")
    sous_dossiers = [
        d for d in sorted(os.listdir(corpus_dir), key=natural_key)
        if os.path.isdir(os.path.join(corpus_dir, d))
    ]
    if not sous_dossiers:
        raise ValueError(f"Aucun auteur dans : {corpus_dir}")
    for author in sous_dossiers:
        author_path = os.path.join(corpus_dir, author)
        fichiers_txt = [
            f for f in sorted(os.listdir(author_path), key=natural_key)
            if f.endswith('.txt')
        ]
        for fname in fichiers_txt:
            fpath = os.path.join(author_path, fname)
            try:
                raw = lire_fichier(fpath)
                entries.append((author, fname, raw))
            except Exception as e:
                print(f"  [WARN] {fpath} : {e}")
    return entries


# ══════════════════════════════════════════════════════════════════
# CARACTÉRISATION N-GRAM (mot/lettre, F1/F2)
# Construit X par texte : 1 échantillon = 1 texte
# ══════════════════════════════════════════════════════════════════
def caracteriser_ngram(corpus_dir, n, methode, niveau, max_features=5000, callback_log=None):
    """
    Renvoie :
      X      : (n_textes, max_features) — vecteurs de fréquences triées
      labels : (n_textes,) noms d'auteurs
      doc_ids: (n_textes,) "Auteur || fichier"
      info   : dict
    """
    if callback_log is None:
        callback_log = print

    entries = load_corpus(corpus_dir)
    callback_log(f"   {len(entries)} fichiers chargés")

    X_list, labels, doc_ids = [], [], []
    for author, fname, raw in entries:
        vec = extraire_vecteur_ngram(raw, n, methode, niveau)
        # padding/truncation à max_features
        v = np.zeros(max_features, dtype=np.float32)
        m = min(len(vec), max_features)
        v[:m] = vec[:m]
        X_list.append(v)
        labels.append(author)
        doc_ids.append(f"{author} || {fname}")

    X = np.array(X_list, dtype=np.float32)
    callback_log(f"   X shape : {X.shape} (n={n}, {methode}, {niveau})")
    return X, np.array(labels), doc_ids, {"n_features": max_features}


def vectoriser_texte_ngram(texte, n, methode, niveau, max_features=5000):
    """Vecteur pour un texte nouveau (n-gram), même dimension que l'entraînement."""
    vec = extraire_vecteur_ngram(texte, n, methode, niveau)
    v = np.zeros(max_features, dtype=np.float32)
    m = min(len(vec), max_features)
    v[:m] = vec[:m]
    return v


# ══════════════════════════════════════════════════════════════════
# CARACTÉRISATION TF-IDF
# ══════════════════════════════════════════════════════════════════
def caracteriser_tfidf(corpus_dir, callback_log=None):
    """
    Renvoie :
      X_full     : (n_docs, n_mots) matrice TF-IDF dense
      labels     : (n_docs,) auteurs
      doc_ids    : (n_docs,) "Auteur || fichier"
      vectorizer : TfidfVectorizer entraîné
      info       : dict (vocab_size)
    """
    if callback_log is None:
        callback_log = print

    entries = load_corpus(corpus_dir)
    callback_log(f"   {len(entries)} fichiers chargés")

    docs = [raw for _, _, raw in entries]
    labels = [a for a, _, _ in entries]
    fichiers = [f for _, f, _ in entries]

    vectorizer = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'[\u0600-\u06FF]+',
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()
    callback_log(f"   Vocabulaire : {len(vocab)} mots | Matrice : {tfidf_matrix.shape}")

    X_full = tfidf_matrix.toarray().astype(np.float32)
    doc_ids = [f"{a} || {f}" for a, f in zip(labels, fichiers)]

    return X_full, np.array(labels), doc_ids, vectorizer, {"vocab_size": len(vocab)}


def vectoriser_texte_tfidf(texte, vectorizer):
    """Vecteur TF-IDF pour un texte nouveau."""
    texte = normaliser_invisibles(texte)
    vec = vectorizer.transform([texte]).toarray().astype(np.float32)
    return vec[0]


def selectionner_top_mots(X_train, y_train, top_n_mots, classes):
    """
    Sélection des top_n_mots par variance inter-auteurs (sur train).
    Reproduit la logique de cnn_TEFIDF_V3.py.
    """
    if top_n_mots is None or top_n_mots >= X_train.shape[1]:
        return np.arange(X_train.shape[1])

    moyennes = np.zeros((len(classes), X_train.shape[1]), dtype=np.float32)
    for idx in range(len(classes)):
        mask = (y_train == idx)
        if mask.sum() > 0:
            moyennes[idx] = X_train[mask].mean(axis=0)

    variance_inter = moyennes.var(axis=0)
    top_indices = np.argsort(variance_inter)[::-1][:top_n_mots]
    return np.sort(top_indices)


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT DE FEATURES PRÉ-CALCULÉES DEPUIS UN CSV
# (n-gram ou TF-IDF) — analogue à charger_embedding_csv
# Format attendu : colonne 'locuteur' (ou 'Auteur'), 'fichier' (option.),
# puis des colonnes de features. Si les en-têtes sont les n-grams / mots
# eux-mêmes, l'alignement d'un nouveau texte est exact.
# ══════════════════════════════════════════════════════════════════
import pandas as _pd


def charger_csv_features(chemin_csv, callback_log=None):
    """
    Renvoie X (n_docs, n_feat), labels, doc_ids, colonnes(list[str]).
    """
    if callback_log is None:
        callback_log = print

    df = _pd.read_csv(chemin_csv)
    df.columns = [str(c).strip() for c in df.columns]

    # Colonne auteur
    col_auteur = None
    for cand in ("locuteur", "Auteur", "auteur", "label", "Label"):
        if cand in df.columns:
            col_auteur = cand
            break
    if col_auteur is None:
        col_auteur = df.columns[0]

    col_fichier = "fichier" if "fichier" in df.columns else None

    meta_cols = {col_auteur}
    if col_fichier:
        meta_cols.add(col_fichier)

    feat_cols = [c for c in df.columns if c not in meta_cols]
    # On ne garde que les colonnes numériques
    X = df[feat_cols].apply(_pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.values.astype(np.float32)

    labels = df[col_auteur].astype(str).values
    if col_fichier:
        doc_ids = [f"{a} || {f}" for a, f in zip(labels, df[col_fichier])]
    else:
        doc_ids = [f"{a} || doc{i}" for i, a in enumerate(labels)]

    callback_log(f"   CSV features chargé : {X.shape} | {len(set(labels))} auteurs")
    return X, np.array(labels), doc_ids, list(feat_cols)


def vectoriser_texte_par_colonnes_ngram(texte, n, methode, niveau, colonnes):
    """
    Vectorise un texte en alignant ses fréquences n-gram sur les en-têtes
    de colonnes fournies (les n-grams du CSV). Alignement exact si les
    en-têtes sont les n-grams eux-mêmes ; sinon vecteur de zéros.
    """
    texte = normaliser_invisibles(texte)
    if niveau == "lettre":
        tokens = tokeniser_caracteres(texte)
    else:
        tokens = tokeniser_mots(texte)
    freq = calculer_frequences_ngram(tokens, n, methode, niveau)
    return np.array([freq.get(str(c), 0.0) for c in colonnes], dtype=np.float32)


def vectoriser_texte_par_colonnes_tfidf(texte, colonnes):
    """
    Vectorise un texte en TF (normalisé) aligné sur les mots-colonnes du CSV.
    Approximation TF (sans IDF du corpus) : suffisante pour une démonstration
    lorsque le CSV pré-calculé est cohérent avec le corpus d'entraînement.
    """
    texte = normaliser_invisibles(texte)
    mots = re.findall(r'[\u0600-\u06FF]+', texte)
    if not mots:
        return np.zeros(len(colonnes), dtype=np.float32)
    compte = Counter(mots)
    total = sum(compte.values())
    return np.array([compte.get(str(c), 0) / total for c in colonnes], dtype=np.float32)
