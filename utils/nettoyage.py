# -*- coding: utf-8 -*-
"""
====================================================================
  Nettoyage de texte arabe — reproduit netoyage.py du PFE
====================================================================
"""

import os
import re
import unicodedata


# ──────────────────────────────────────────────────────────────────
# Tri naturel
# ──────────────────────────────────────────────────────────────────
def natural_key(s):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', str(s))]


# ──────────────────────────────────────────────────────────────────
# Normalisation invisible (cohérence textarea / fichier)
# ──────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────
# Nettoyage avancé arabe (IDENTIQUE à netoyage.py)
# ──────────────────────────────────────────────────────────────────
def nettoyer_texte_arabe(texte):
    """
    Reproduit EXACTEMENT la fonction nettoyer_texte_arabe de netoyage.py :
    - Supprime les parenthèses contenant du latin
    - Supprime les mots latins
    - Supprime les durées (minutes, hours...)
    - Normalise اللّه -> الله
    - Supprime les diacritiques (tashkil)
    - Garde uniquement arabe + chiffres + espaces
    """
    texte = texte.strip()
    texte = re.sub(r'\(.*?[a-zA-Z].*?\)', '', texte)
    texte = re.sub(r'[a-zA-Z]+[\w\.\-\/]*', '', texte)
    texte = re.sub(r'\b\d+\s*(minutes?|hours?|seconds?|files?|long)\b', '', texte, flags=re.IGNORECASE)
    texte = texte.replace("اللّه", "الله")
    texte = re.sub(r'[\u064B-\u0652]', '', texte)
    texte = re.sub(r'[^\u0600-\u06FF\u0660-\u0669\u06F0-\u06F90-9\s]', '', texte)
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()


# ──────────────────────────────────────────────────────────────────
# Lecture multi-encodage
# ──────────────────────────────────────────────────────────────────
def lire_fichier(chemin):
    for encodage in ("utf-8", "utf-8-sig", "cp1256", "latin-1", "cp1252"):
        try:
            with open(chemin, "r", encoding=encodage) as f:
                contenu = f.read()
            if contenu.strip():
                return contenu
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    raise ValueError(f"Impossible de lire : {chemin}")


# ──────────────────────────────────────────────────────────────────
# Nettoyage batch d'un corpus entier
# ──────────────────────────────────────────────────────────────────
def nettoyer_corpus(dossier_source, dossier_destination, callback_log=None):
    """
    Parcourt dossier_source/AuteurX/texte.txt, nettoie chaque fichier,
    et sauvegarde dans dossier_destination en gardant la structure.
    """
    if callback_log is None:
        callback_log = print

    os.makedirs(dossier_destination, exist_ok=True)

    total_fichiers = 0
    total_mots_avant = 0
    total_mots_apres = 0

    sous_dossiers = sorted(
        [d for d in os.listdir(dossier_source)
         if os.path.isdir(os.path.join(dossier_source, d))],
        key=natural_key
    )

    if not sous_dossiers:
        raise ValueError("Aucun sous-dossier (auteur) dans le dossier source.")

    for sous_dossier in sous_dossiers:
        chemin_src = os.path.join(dossier_source, sous_dossier)
        chemin_dst = os.path.join(dossier_destination, sous_dossier)
        os.makedirs(chemin_dst, exist_ok=True)

        fichiers_txt = sorted(
            [f for f in os.listdir(chemin_src) if f.lower().endswith(".txt")],
            key=natural_key
        )

        callback_log(f"  {sous_dossier} : {len(fichiers_txt)} fichier(s)")

        for fichier in fichiers_txt:
            src = os.path.join(chemin_src, fichier)
            dst = os.path.join(chemin_dst, fichier)
            try:
                texte = lire_fichier(src)
                nb_avant = len(texte.split())
                texte_nettoye = nettoyer_texte_arabe(texte)
                nb_apres = len(texte_nettoye.split())

                with open(dst, "w", encoding="utf-8") as f:
                    f.write(texte_nettoye)

                total_fichiers += 1
                total_mots_avant += nb_avant
                total_mots_apres += nb_apres
            except Exception as e:
                callback_log(f"    ERREUR {fichier} : {e}")

    taux = 0.0
    if total_mots_avant > 0:
        taux = ((total_mots_avant - total_mots_apres) / total_mots_avant) * 100

    resume = {
        "fichiers": total_fichiers,
        "mots_avant": total_mots_avant,
        "mots_apres": total_mots_apres,
        "mots_supprimes": total_mots_avant - total_mots_apres,
        "taux_reduction": round(taux, 1),
    }
    callback_log(f"  Terminé : {total_fichiers} fichiers, {taux:.1f}% de réduction")
    return resume
