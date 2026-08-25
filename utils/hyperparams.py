# -*- coding: utf-8 -*-
"""
====================================================================
  Hyperparamètres OPTIMAUX intégrés (prêt à l'emploi)
  --------------------------------------------------------------
  Toutes les valeurs proviennent des fichiers Excel de résultats
  du mémoire (feuilles « Résumé Global » — paramètres optimaux).
  L'utilisateur ne règle plus rien manuellement : l'application
  sélectionne automatiquement la configuration optimale selon
  (méthode, base, modèle, options).
====================================================================
"""

# ──────────────────────────────────────────────────────────────────
#  Table des hyperparamètres optimaux
#  Clé : (methode, niveau, f, embed_type, base, modele)
#  base : 'nettoye' (corpus nettoyé) / 'brut' (corpus non nettoyé)
# ──────────────────────────────────────────────────────────────────
OPTIMAL = {

    # ══════════════ N-GRAM · CNN ══════════════
    ("ngram", "lettre", "F1", None, "nettoye", "cnn"): dict(
        n_gram=7, lr=0.002, batch_size=80, epochs=80, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.9995, test_f1=0.9994),
    ("ngram", "lettre", "F1", None, "brut", "cnn"): dict(
        n_gram=13, lr=0.001, batch_size=80, epochs=60, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.9996, test_f1=0.9996),
    ("ngram", "lettre", "F2", None, "nettoye", "cnn"): dict(
        n_gram=7, lr=0.002, batch_size=80, epochs=100, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.9906, test_f1=0.9889),
    ("ngram", "lettre", "F2", None, "brut", "cnn"): dict(
        n_gram=5, lr=0.001, batch_size=16, epochs=20, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.9990, test_f1=0.9988),
    ("ngram", "mot", "F1", None, "nettoye", "cnn"): dict(
        n_gram=9, lr=0.001, batch_size=128, epochs=80, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.9976, test_f1=0.9977),
    ("ngram", "mot", "F1", None, "brut", "cnn"): dict(
        n_gram=7, lr=0.003, batch_size=64, epochs=40, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.9968, test_f1=0.9969),
    ("ngram", "mot", "F2", None, "nettoye", "cnn"): dict(
        n_gram=7, lr=0.005, batch_size=64, epochs=60, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.9975, test_f1=0.9975),
    ("ngram", "mot", "F2", None, "brut", "cnn"): dict(
        n_gram=7, lr=0.003, batch_size=16, epochs=10, activation="tanh",
        weight_decay=5e-4, dropout=0.4, test_acc=0.9954, test_f1=0.9944),

    # ══════════════ N-GRAM · SAE (AE + LogReg) ══════════════
    ("ngram", "lettre", "F1", None, "nettoye", "sae"): dict(
        n_gram=11, lr=0.03, batch_size=80, epochs=80, activation="relu",
        weight_decay=5e-4, dropout=0.2, latent_dim=64, test_acc=0.9984, test_f1=0.9982),
    ("ngram", "lettre", "F1", None, "brut", "sae"): dict(
        n_gram=11, lr=0.001, batch_size=256, epochs=80, activation="gelu",
        weight_decay=5e-4, dropout=0.4, latent_dim=64, test_acc=0.9986, test_f1=0.9984),
    ("ngram", "lettre", "F2", None, "nettoye", "sae"): dict(
        n_gram=15, lr=0.003, batch_size=144, epochs=60, activation="relu",
        weight_decay=5e-4, dropout=0.4, latent_dim=64, test_acc=0.9922, test_f1=0.9908),
    ("ngram", "lettre", "F2", None, "brut", "sae"): dict(
        n_gram=13, lr=0.01, batch_size=256, epochs=100, activation="tanh",
        weight_decay=5e-4, dropout=0.4, latent_dim=64, test_acc=0.9924, test_f1=0.9910),
    ("ngram", "mot", "F1", None, "nettoye", "sae"): dict(
        n_gram=5, lr=0.001, batch_size=16, epochs=40, activation="relu",
        weight_decay=5e-4, dropout=0.4, latent_dim=64, test_acc=0.9974, test_f1=0.9974),
    ("ngram", "mot", "F1", None, "brut", "sae"): dict(
        n_gram=11, lr=0.01, batch_size=128, epochs=100, activation="tanh",
        weight_decay=5e-4, dropout=0.4, latent_dim=64, test_acc=0.9977, test_f1=0.9979),
    ("ngram", "mot", "F2", None, "nettoye", "sae"): dict(
        n_gram=11, lr=0.01, batch_size=256, epochs=40, activation="relu",
        weight_decay=5e-4, dropout=0.4, latent_dim=64, test_acc=0.9978, test_f1=0.9979),
    ("ngram", "mot", "F2", None, "brut", "sae"): dict(
        n_gram=11, lr=0.01, batch_size=128, epochs=80, activation="relu",
        weight_decay=5e-4, dropout=0.4, latent_dim=64, test_acc=0.9977, test_f1=0.9979),

    # ══════════════ TF-IDF · CNN ══════════════
    ("tfidf", None, None, None, "nettoye", "cnn"): dict(
        top_n_mots=150, lr=3e-4, batch_size=16, epochs=125, activation="relu",
        weight_decay=1e-4, dropout=0.5, test_acc=0.8750, test_f1=0.8500),
    ("tfidf", None, None, None, "brut", "cnn"): dict(
        top_n_mots=150, lr=3e-4, batch_size=16, epochs=200, activation="relu",
        weight_decay=1e-5, dropout=0.5, test_acc=0.9062, test_f1=0.9042),

    # ══════════════ TF-IDF · SAE (joint) ══════════════
    ("tfidf", None, None, None, "nettoye", "sae"): dict(
        top_n_mots=2000, lr=5e-4, batch_size=16, epochs=500, activation="relu",
        weight_decay=5e-4, dropout=0.3, latent_dim=256, hidden=512, hidden_clf=64,
        l1_lambda=1e-4, w_recon=0.1, w_class=1.0, test_acc=0.9688, test_f1=0.9667),
    ("tfidf", None, None, None, "brut", "sae"): dict(
        top_n_mots=1000, lr=5e-4, batch_size=16, epochs=500, activation="gelu",
        weight_decay=1e-3, dropout=0.3, latent_dim=128, hidden=256, hidden_clf=64,
        l1_lambda=1e-4, w_recon=0.1, w_class=1.0, test_acc=0.9688, test_f1=0.9667),

    # ══════════════ EMBEDDING AraBERT · CNN ══════════════
    ("embedding", None, None, "arabert", "nettoye", "cnn"): dict(
        embed_dim=400, lr=0.003, batch_size=16, epochs=300, activation="relu",
        weight_decay=1e-3, dropout=0.4, test_acc=0.5000, test_f1=0.4443),
    ("embedding", None, None, "arabert", "brut", "cnn"): dict(
        embed_dim=400, lr=0.01, batch_size=16, epochs=700, activation="relu",
        weight_decay=1e-4, dropout=0.4, test_acc=0.4690, test_f1=0.4690),

    # ══════════════ EMBEDDING AraBERT · SAE (joint) ══════════════
    ("embedding", None, None, "arabert", "nettoye", "sae"): dict(
        embed_dim=600, lr=5e-4, batch_size=16, epochs=400, activation="gelu",
        weight_decay=1e-3, dropout=0.3, latent_dim=256, hidden=510, hidden_clf=64,
        l1_lambda=1e-5, w_recon=0.1, w_class=1.0, test_acc=0.8438, test_f1=0.8250),
    ("embedding", None, None, "arabert", "brut", "sae"): dict(
        embed_dim=400, lr=3e-3, batch_size=16, epochs=400, activation="sigmoid",
        weight_decay=5e-4, dropout=0.3, latent_dim=128, hidden=256, hidden_clf=128,
        l1_lambda=1e-3, w_recon=0.3, w_class=1.0, test_acc=0.8750, test_f1=0.8521),

    # ══════════════ EMBEDDING AraELECTRA · CNN ══════════════
    ("embedding", None, None, "araelectra", "nettoye", "cnn"): dict(
        embed_dim=400, lr=0.001, batch_size=64, epochs=700, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.3438, test_f1=0.3566),
    ("embedding", None, None, "araelectra", "brut", "cnn"): dict(
        embed_dim=600, lr=0.001, batch_size=16, epochs=400, activation="relu",
        weight_decay=5e-4, dropout=0.4, test_acc=0.5938, test_f1=0.5667),

    # ══════════════ EMBEDDING AraELECTRA · SAE (joint) ══════════════
    ("embedding", None, None, "araelectra", "nettoye", "sae"): dict(
        embed_dim=600, lr=5e-4, batch_size=16, epochs=400, activation="gelu",
        weight_decay=5e-4, dropout=0.3, latent_dim=128, hidden=500, hidden_clf=64,
        l1_lambda=1e-2, w_recon=0.07, w_class=1.0, test_acc=0.9062, test_f1=0.9042),
    ("embedding", None, None, "araelectra", "brut", "sae"): dict(
        embed_dim=768, lr=1e-4, batch_size=16, epochs=100, activation="gelu",
        weight_decay=5e-4, dropout=0.3, latent_dim=512, hidden=256, hidden_clf=128,
        l1_lambda=1e-2, w_recon=0.07, w_class=1.0, test_acc=0.8750, test_f1=0.8565),
}

# Valeurs par défaut communes (complètent les dicts ci-dessus)
_DEFAUTS = dict(
    lr=0.001, batch_size=16, epochs=80, weight_decay=5e-4, dropout=0.4,
    activation="relu", top_n_mots=150, latent_dim=128, hidden=256,
    hidden_clf=64, l1_lambda=1e-4, w_recon=0.1, w_class=1.0,
    n_gram=7, embed_dim=400,
)


def get_config(methode, base, modele, ngram_niveau=None, ngram_f=None, embed_type=None):
    """
    Retourne le dict complet d'hyperparamètres optimaux + métadonnées
    (test_acc, test_f1) pour la configuration demandée.
    Lève KeyError si la combinaison n'existe pas.
    """
    if methode == "ngram":
        cle = ("ngram", ngram_niveau, ngram_f, None, base, modele)
    elif methode == "tfidf":
        cle = ("tfidf", None, None, None, base, modele)
    else:  # embedding
        cle = ("embedding", None, None, embed_type, base, modele)

    if cle not in OPTIMAL:
        raise KeyError(f"Aucune configuration optimale pour : {cle}")

    cfg = dict(_DEFAUTS)
    cfg.update(OPTIMAL[cle])
    return cfg


def resume_lisible(methode, base, modele, cfg, ngram_niveau=None, ngram_f=None, embed_type=None):
    """Construit la liste (label, valeur) des hyperparamètres à afficher."""
    lignes = []
    if methode == "ngram":
        lignes.append(("N (taille du n-gram)", cfg["n_gram"]))
    elif methode == "tfidf":
        lignes.append(("TOP_N_MOTS", cfg["top_n_mots"]))
    else:
        lignes.append(("Dimension embedding", cfg["embed_dim"]))

    lignes += [
        ("Learning rate", cfg["lr"]),
        ("Batch size", cfg["batch_size"]),
        ("Epochs", cfg["epochs"]),
        ("Fonction d'activation", cfg["activation"].upper()),
    ]
    if modele == "sae" and methode in ("tfidf", "embedding"):
        lignes += [
            ("Dimension latente", cfg["latent_dim"]),
            ("Hidden AE", cfg["hidden"]),
            ("Hidden classifieur", cfg["hidden_clf"]),
            ("L1 lambda", cfg["l1_lambda"]),
            ("W_class / W_recon", f"{cfg['w_class']} / {cfg['w_recon']}"),
        ]
    elif modele == "sae" and methode == "ngram":
        lignes.append(("Dimension latente", cfg["latent_dim"]))
    return lignes


# ──────────────────────────────────────────────────────────────────
#  Métadonnées des auteurs : nationalité + drapeau
#  (best effort — vérifie / ajuste librement ce dictionnaire)
# ──────────────────────────────────────────────────────────────────
AUTEURS_META = {
    "aaidh al qarni":          ("Arabie Saoudite", "🇸🇦"),
    "abu ishaq al houwayni":   ("Égypte",          "🇪🇬"),
    "amr khaled":              ("Égypte",          "🇪🇬"),
    "ibrahim el eiky":         ("Égypte",          "🇪🇬"),
    "khaled al ghamdi":        ("Arabie Saoudite", "🇸🇦"),
    "mahmoud al masri":        ("Égypte",          "🇪🇬"),
    "masaad anouar":           ("Égypte",          "🇪🇬"),
    "massad anouar":           ("Égypte",          "🇪🇬"),
    "mohamed al ghazali":      ("Égypte",          "🇪🇬"),
    "mustafa al adawi":        ("Égypte",          "🇪🇬"),
    "nabil al awadi":          ("Koweït",          "🇰🇼"),
    "omar abdelkafi":          ("Égypte",          "🇪🇬"),
    "oussama hadad":           ("Algérie",         "🇩🇿"),
    "rachid el zahrani":       ("Arabie Saoudite", "🇸🇦"),
    "ratib al nabulsi":        ("Syrie",           "🇸🇾"),
    "saad al arafat":          ("Égypte",          "🇪🇬"),
    "saad al atik":            ("Arabie Saoudite", "🇸🇦"),
}


def _normaliser_nom(nom):
    return " ".join(str(nom).strip().lower().split())


def info_auteur(nom):
    """Retourne (nationalite, drapeau) pour un nom d'auteur (tolérant à la casse)."""
    return AUTEURS_META.get(_normaliser_nom(nom), ("—", "🏳️"))
