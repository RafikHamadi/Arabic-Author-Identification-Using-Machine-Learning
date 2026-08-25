# -*- coding: utf-8 -*-
"""
====================================================================
  Transcription audio/vidéo via Whisper (local, open-source)
  Le texte transcrit est ajouté automatiquement au corpus brut.
====================================================================

Installation requise (sur la machine de l'utilisateur) :
    pip install openai-whisper
    + ffmpeg installé (https://ffmpeg.org/download.html)

Modèles disponibles (du + rapide au + précis) :
    tiny, base, small, medium, large
Pour l'arabe, 'small' ou 'medium' donnent un bon compromis.
"""

import os

# Le modèle Whisper est chargé paresseusement (lazy) une seule fois
_WHISPER_MODEL = None
_WHISPER_TAILLE = None


def whisper_disponible():
    """Vérifie si la librairie whisper est installée."""
    try:
        import whisper  # noqa
        return True
    except ImportError:
        return False


def charger_modele(taille="small", callback_log=None):
    """
    Charge le modèle Whisper (une seule fois, mis en cache).
    taille : tiny / base / small / medium / large
    """
    global _WHISPER_MODEL, _WHISPER_TAILLE
    if callback_log is None:
        callback_log = print

    if _WHISPER_MODEL is not None and _WHISPER_TAILLE == taille:
        return _WHISPER_MODEL

    import whisper
    callback_log(f"[WHISPER] Chargement du modèle '{taille}' (peut prendre du temps au 1er lancement)...")
    _WHISPER_MODEL = whisper.load_model(taille)
    _WHISPER_TAILLE = taille
    callback_log(f"[WHISPER] Modèle '{taille}' chargé.")
    return _WHISPER_MODEL


def transcrire(chemin_fichier, taille="small", langue="ar", callback_log=None):
    """
    Transcrit un fichier audio/vidéo en texte.

    chemin_fichier : chemin vers le .mp3/.wav/.mp4/.m4a...
    taille         : modèle Whisper
    langue         : 'ar' pour l'arabe (force la langue)

    Retourne le texte transcrit (str).
    """
    if callback_log is None:
        callback_log = print

    if not whisper_disponible():
        raise RuntimeError(
            "La librairie Whisper n'est pas installée. "
            "Installe-la avec : pip install openai-whisper "
            "(et assure-toi que ffmpeg est installé)."
        )

    model = charger_modele(taille, callback_log=callback_log)

    callback_log(f"[WHISPER] Transcription de {os.path.basename(chemin_fichier)} (langue={langue})...")
    resultat = model.transcribe(chemin_fichier, language=langue, fp16=False)
    texte = resultat.get("text", "").strip()
    callback_log(f"[WHISPER] Terminé : {len(texte)} caractères transcrits.")
    return texte
