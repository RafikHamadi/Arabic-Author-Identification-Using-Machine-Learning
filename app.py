# -*- coding: utf-8 -*-
"""
====================================================================
  Application PFE V2 — Reconnaissance automatique d'auteurs arabes
  --------------------------------------------------------------
  · Hyperparamètres OPTIMAUX intégrés (prêt à l'emploi)
  · Parcours guidé en étapes : Base → Caractérisation → Modèle → Résultat
  · Options utilisateur limitées : type n-gram (lettre/mot), F1/F2,
    modèle de langue (AraBERT/AraELECTRA), + CSV pré-calculé
  · Modèles : CNN 1D · SAE (Auto-encodeur sparse)
====================================================================
"""

import os
import sys
import re
import threading

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
os.chdir(_BASE)

from flask import (Flask, render_template, request, redirect,
                   url_for, jsonify, flash, session)

from utils.nettoyage import nettoyer_corpus, normaliser_invisibles, natural_key
from utils.caracterisation import (
    caracteriser_ngram, caracteriser_tfidf,
    vectoriser_texte_ngram, vectoriser_texte_tfidf,
    charger_csv_features,
    vectoriser_texte_par_colonnes_ngram, vectoriser_texte_par_colonnes_tfidf,
)
from utils.model_cnn import (
    entrainer_cnn_ngram, entrainer_cnn_tfidf, entrainer_cnn_embedding,
    predire_ngram, predire_tfidf, predire_embedding_cnn,
)
from utils.model_sae import (
    entrainer_sae_tfidf, entrainer_sae_ngram, entrainer_sae_embedding,
    predire_sae_tfidf, predire_sae_ngram, predire_sae_embedding,
)
from utils.embedding import (
    caracteriser_embedding_corpus, vectoriser_texte_embedding,
    charger_embedding_csv,
)
from utils.transcription import transcrire, whisper_disponible
from utils.hyperparams import get_config, resume_lisible, info_auteur


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
BASE_DIR = _BASE
CORPUS_BRUT = os.path.join(BASE_DIR, "corpus_brut")
CORPUS_NETTOYE = os.path.join(BASE_DIR, "corpus_nettoye")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
for d in (CORPUS_BRUT, CORPUS_NETTOYE, UPLOADS_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.secret_key = "pfe_arabic_authorship_v2_2026"

# ──────────────────────────────────────────────────────────────────
# Verrou global : une seule analyse à la fois
# ──────────────────────────────────────────────────────────────────
# Le serveur Flask est multithread : si l'utilisateur lance une 2ᵉ analyse
# avant la fin de la 1ʳᵉ, les deux trainings tournent en parallèle et leurs
# logs se mélangent. Le verrou ci-dessous sérialise les analyses : si une
# est déjà en cours, la nouvelle est rejetée proprement.
_ANALYSE_LOCK = threading.Lock()

# Libellés lisibles pour le récapitulatif
LBL_BASE = {"brut": "Base brute (non nettoyée)", "nettoye": "Base nettoyée"}
LBL_METHODE = {"ngram": "N-grammes", "tfidf": "TF-IDF", "embedding": "Embedding"}
LBL_MODELE = {"cnn": "CNN 1D (réseau de neurones convolutif)",
              "sae": "SAE (auto-encodeur sparse)"}


# ──────────────────────────────────────────────────────────────────
# Helpers corpus
# ──────────────────────────────────────────────────────────────────
def corpus_path(base):
    return CORPUS_NETTOYE if base == "nettoye" else CORPUS_BRUT


def lister_auteurs(base):
    chemin = corpus_path(base)
    auteurs = []
    if not os.path.exists(chemin):
        return auteurs
    sous = sorted([d for d in os.listdir(chemin)
                   if os.path.isdir(os.path.join(chemin, d))], key=natural_key)
    for auteur in sous:
        fichiers = [f for f in os.listdir(os.path.join(chemin, auteur))
                    if f.lower().endswith(".txt")]
        nationalite, drapeau = info_auteur(auteur)
        auteurs.append({"nom": auteur, "nb_textes": len(fichiers),
                        "nationalite": nationalite, "drapeau": drapeau})
    return auteurs


def prochain_numero(base):
    nums = [int(re.search(r'\d+', a["nom"]).group())
            for a in lister_auteurs(base) if re.search(r'\d+', a["nom"])]
    return max(nums) + 1 if nums else 1


def lire_fichier_upload(fichier):
    raw = fichier.read()
    for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            t = raw.decode(enc)
            if t.strip():
                return t
        except UnicodeDecodeError:
            continue
    return ""


# ──────────────────────────────────────────────────────────────────
# Accueil
# ──────────────────────────────────────────────────────────────────
@app.route("/")
def accueil():
    return render_template(
        "index.html",
        auteurs_brut=lister_auteurs("brut"),
        auteurs_nettoye=lister_auteurs("nettoye"),
    )


# ──────────────────────────────────────────────────────────────────
# Enregistrer un texte
# ──────────────────────────────────────────────────────────────────
@app.route("/enregistrer", methods=["GET", "POST"])
def enregistrer():
    if request.method == "POST":
        base = request.form.get("base", "brut")
        mode = request.form.get("mode", "nouveau")
        nom_auteur = request.form.get("nom_auteur", "").strip()
        texte_saisi = request.form.get("texte", "").strip()
        fichier = request.files.get("fichier_txt")
        chemin_base = corpus_path(base)

        if mode == "nouveau":
            dossier_auteur = f"Auteur {prochain_numero(base)}"
        else:
            if not nom_auteur:
                flash("Veuillez choisir un auteur existant.", "erreur")
                return redirect(url_for("enregistrer"))
            dossier_auteur = nom_auteur

        chemin_dossier = os.path.join(chemin_base, dossier_auteur)
        os.makedirs(chemin_dossier, exist_ok=True)
        existants = [f for f in os.listdir(chemin_dossier) if f.lower().endswith(".txt")]
        nom_fichier = f"texte{len(existants) + 1}.txt"

        contenu = ""
        if fichier and fichier.filename:
            contenu = lire_fichier_upload(fichier)
        elif texte_saisi:
            contenu = texte_saisi
        else:
            flash("Veuillez fournir un texte ou un fichier.", "erreur")
            return redirect(url_for("enregistrer"))

        contenu = normaliser_invisibles(contenu)
        with open(os.path.join(chemin_dossier, nom_fichier), "w", encoding="utf-8") as f:
            f.write(contenu)
        flash(f"Texte enregistré : [{base}] {dossier_auteur}/{nom_fichier}", "succes")
        return redirect(url_for("enregistrer"))

    return render_template(
        "enregistrer.html",
        auteurs_brut=lister_auteurs("brut"),
        auteurs_nettoye=lister_auteurs("nettoye"),
        prochain_brut=prochain_numero("brut"),
        prochain_nettoye=prochain_numero("nettoye"),
    )


# ──────────────────────────────────────────────────────────────────
# Nettoyer le corpus
# ──────────────────────────────────────────────────────────────────
@app.route("/nettoyer", methods=["POST"])
def nettoyer():
    logs = []
    def log(m):
        logs.append(m); print(m)
    try:
        resume = nettoyer_corpus(CORPUS_BRUT, CORPUS_NETTOYE, callback_log=log)
        flash(f"Nettoyage terminé : {resume['fichiers']} fichiers, "
              f"{resume['taux_reduction']}% de réduction "
              f"({resume['mots_avant']} → {resume['mots_apres']} mots).", "succes")
    except Exception as e:
        flash(f"Erreur nettoyage : {e}", "erreur")
    return redirect(url_for("accueil"))


# ══════════════════════════════════════════════════════════════════
#  PARCOURS GUIDÉ DE CLASSIFICATION  (Base → Caractér. → Modèle → Résultat)
# ══════════════════════════════════════════════════════════════════
CLE_WIZ = ("cl_base", "cl_methode", "cl_ngram_niveau", "cl_ngram_f",
           "cl_embed_type", "cl_source", "cl_csv_path", "cl_modele")


def _reset_wizard():
    for k in CLE_WIZ:
        session.pop(k, None)


@app.route("/classifier")
def classifier():
    """Point d'entrée : réinitialise le parcours et démarre à l'étape 1."""
    _reset_wizard()
    return redirect(url_for("classifier_base"))


# ─────────────── ÉTAPE 1 · BASE ───────────────
@app.route("/classifier/base", methods=["GET", "POST"])
def classifier_base():
    if request.method == "POST":
        session["cl_base"] = request.form.get("base", "brut")
        return redirect(url_for("classifier_caracterisation"))
    return render_template(
        "classifier_base.html", etape=1,
        choix=session.get("cl_base"),
        auteurs_brut=lister_auteurs("brut"),
        auteurs_nettoye=lister_auteurs("nettoye"),
    )


# ─────────────── ÉTAPE 2 · CARACTÉRISATION ───────────────
@app.route("/classifier/caracterisation", methods=["GET", "POST"])
def classifier_caracterisation():
    if not session.get("cl_base"):
        return redirect(url_for("classifier_base"))

    if request.method == "POST":
        methode = request.form.get("methode", "ngram")
        session["cl_methode"] = methode
        session["cl_ngram_niveau"] = request.form.get("ngram_niveau", "lettre")
        session["cl_ngram_f"] = request.form.get("ngram_f", "F1")
        session["cl_embed_type"] = request.form.get("embed_type", "arabert")

        # Source : corpus (calcul) ou CSV pré-calculé
        source = request.form.get(f"source_{methode}", "corpus")
        session["cl_source"] = source
        session["cl_csv_path"] = None
        if source == "csv":
            fichier_csv = request.files.get(f"csv_{methode}")
            if fichier_csv and fichier_csv.filename:
                csv_path = os.path.join(UPLOADS_DIR, f"features_{methode}.csv")
                fichier_csv.save(csv_path)
                session["cl_csv_path"] = csv_path
            else:
                flash("Veuillez fournir un fichier CSV pré-calculé, ou choisir « Corpus ».", "erreur")
                return redirect(url_for("classifier_caracterisation"))

        return redirect(url_for("classifier_modele"))

    return render_template(
        "classifier_caracterisation.html", etape=2,
        base=session.get("cl_base"),
    )


# ─────────────── ÉTAPE 3 · MODÈLE ───────────────
@app.route("/classifier/modele", methods=["GET", "POST"])
def classifier_modele():
    if not session.get("cl_methode"):
        return redirect(url_for("classifier_caracterisation"))

    if request.method == "POST":
        session["cl_modele"] = request.form.get("modele", "cnn")
        return redirect(url_for("classifier_resultat"))

    return render_template("classifier_modele.html", etape=3)


# ─────────────── ÉTAPE 4 · TEXTE → RÉSULTAT ───────────────
@app.route("/classifier/resultat", methods=["GET", "POST"])
def classifier_resultat():
    if not session.get("cl_modele"):
        return redirect(url_for("classifier_modele"))

    base = session["cl_base"]
    methode = session["cl_methode"]
    modele = session["cl_modele"]
    ngram_niveau = session.get("cl_ngram_niveau", "lettre")
    ngram_f = session.get("cl_ngram_f", "F1")
    embed_type = session.get("cl_embed_type", "arabert")
    source = session.get("cl_source", "corpus")
    csv_path = session.get("cl_csv_path")

    # Récapitulatif affiché à l'utilisateur
    recap = _construire_recap(base, methode, modele, ngram_niveau, ngram_f, embed_type, source)

    if request.method == "GET":
        return render_template("classifier_resultat.html", etape=4, recap=recap,
                               base=base, methode=methode)

    # ── POST : récupérer le texte et lancer la classification ──
    texte = request.form.get("texte", "").strip()
    fichier = request.files.get("fichier_txt")
    if fichier and fichier.filename:
        texte = lire_fichier_upload(fichier)
    texte = normaliser_invisibles(texte)

    if not texte.strip():
        flash("Veuillez fournir un texte à classifier.", "erreur")
        return render_template("classifier_resultat.html", etape=4, recap=recap,
                               base=base, methode=methode)

    auteurs = lister_auteurs(base)
    if source != "csv" and len([a for a in auteurs if a["nb_textes"] >= 2]) < 2:
        flash(f"La base [{base}] doit avoir ≥2 auteurs avec ≥2 textes.", "erreur")
        return render_template("classifier_resultat.html", etape=4, recap=recap,
                               base=base, methode=methode)

    logs = []
    def log(m):
        logs.append(m); print(m)

    # ── Verrou : refuser si une analyse est déjà en cours ──
    if not _ANALYSE_LOCK.acquire(blocking=False):
        flash("Une analyse est déjà en cours. Patientez quelques instants "
              "puis recommencez, ou redémarrez l'application si elle semble bloquée.",
              "erreur")
        return render_template("classifier_resultat.html", etape=4, recap=recap,
                               base=base, methode=methode)

    try:
        cfg = get_config(methode, base, modele, ngram_niveau, ngram_f, embed_type)
        hp = dict(cfg)
        contexte = _executer_pipeline(
            base, methode, modele, ngram_niveau, ngram_f, embed_type,
            source, csv_path, texte, cfg, hp, log)

        hp_affiche = resume_lisible(methode, base, modele, cfg,
                                    ngram_niveau, ngram_f, embed_type)
        return render_template(
            "resultat.html",
            recap=recap, hp_affiche=hp_affiche,
            cfg=cfg, logs=logs, methode=methode, base=base,
            texte_apercu=texte[:500] + ("..." if len(texte) > 500 else ""),
            **contexte,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc(); print(tb)
        flash(f"Erreur : {e}", "erreur")
        return render_template("classifier_resultat.html", etape=4, recap=recap,
                               base=base, methode=methode, logs=logs + [tb])
    finally:
        _ANALYSE_LOCK.release()


def _construire_recap(base, methode, modele, ngram_niveau, ngram_f, embed_type, source):
    if methode == "ngram":
        detail = f"N-grammes {ngram_niveau} · {ngram_f}"
    elif methode == "tfidf":
        detail = "TF-IDF"
    else:
        detail = f"Embedding {'AraBERT' if embed_type == 'arabert' else 'AraELECTRA'}"
    src = "CSV pré-calculé" if source == "csv" else "Corpus (calcul à la volée)"
    return {
        "base": LBL_BASE.get(base, base),
        "methode": LBL_METHODE.get(methode, methode),
        "detail": detail,
        "modele": LBL_MODELE.get(modele, modele),
        "source": src,
    }


def _executer_pipeline(base, methode, modele, ngram_niveau, ngram_f, embed_type,
                        source, csv_path, texte, cfg, hp, log):
    chemin = corpus_path(base)

    # ───────────── N-GRAM ─────────────
    if methode == "ngram":
        n = cfg["n_gram"]
        log(f"[CARACTÉRISATION] N-gram {ngram_niveau} {ngram_f} n={n} (source={source})")
        if source == "csv" and csv_path:
            X, labels, doc_ids, colonnes = charger_csv_features(csv_path, callback_log=log)
            vec = vectoriser_texte_par_colonnes_ngram(texte, n, ngram_f, ngram_niveau, colonnes)
        else:
            X, labels, doc_ids, info = caracteriser_ngram(chemin, n, ngram_f, ngram_niveau, callback_log=log)
            vec = vectoriser_texte_ngram(texte, n, ngram_f, ngram_niveau)

        if modele == "sae":
            log("[MODÈLE] SAE + LogReg (n-gram)")
            model, metrics, le, scaler, clf = entrainer_sae_ngram(X, labels, hp, callback_log=log)
            auteur, top = predire_sae_ngram(model, clf, vec, scaler, le, top_k=5)
        else:
            log("[MODÈLE] CNN (n-gram)")
            model, metrics, le, scaler, _ = entrainer_cnn_ngram(X, labels, hp, callback_log=log)
            auteur, top = predire_ngram(model, vec, scaler, le, top_k=5)

    # ───────────── TF-IDF ─────────────
    elif methode == "tfidf":
        log(f"[CARACTÉRISATION] TF-IDF (source={source})")
        if source == "csv" and csv_path:
            X_full, labels, doc_ids, colonnes = charger_csv_features(csv_path, callback_log=log)
            vec_full = vectoriser_texte_par_colonnes_tfidf(texte, colonnes)
        else:
            X_full, labels, doc_ids, vectorizer, info = caracteriser_tfidf(chemin, callback_log=log)
            vec_full = vectoriser_texte_tfidf(texte, vectorizer)

        holder = []
        if modele == "sae":
            log("[MODÈLE] SAE joint (TF-IDF)")
            model, metrics, le, top_idx = entrainer_sae_tfidf(X_full, labels, holder, hp, callback_log=log)
            auteur, top = predire_sae_tfidf(model, vec_full, top_idx, le, top_k=5)
        else:
            log("[MODÈLE] CNN (TF-IDF)")
            model, metrics, le, _, top_idx = entrainer_cnn_tfidf(X_full, labels, holder, hp, callback_log=log)
            auteur, top = predire_tfidf(model, vec_full, top_idx, le, top_k=5)

    # ───────────── EMBEDDING ─────────────
    else:
        dim = cfg["embed_dim"]
        log(f"[CARACTÉRISATION] Embedding {embed_type} dim={dim} (source={source})")
        if source == "csv" and csv_path:
            X, labels, doc_ids = charger_embedding_csv(csv_path, dim, callback_log=log)
        else:
            X, labels, doc_ids = caracteriser_embedding_corpus(chemin, embed_type, dim, callback_log=log)
        vec = vectoriser_texte_embedding(texte, embed_type, dim, callback_log=log)

        if modele == "sae":
            log("[MODÈLE] SAE joint (embedding)")
            model, metrics, le, scaler = entrainer_sae_embedding(X, labels, hp, callback_log=log)
            auteur, top = predire_sae_embedding(model, vec, scaler, le, top_k=5)
        else:
            log("[MODÈLE] CNN (embedding)")
            model, metrics, le, scaler, _ = entrainer_cnn_embedding(X, labels, hp, callback_log=log)
            auteur, top = predire_embedding_cnn(model, vec, scaler, le, top_k=5)

    nat, drap = info_auteur(auteur)
    log(f"   ► Auteur identifié : {auteur}")
    return {
        "auteur_predit": auteur,
        "auteur_drapeau": drap,
        "auteur_nationalite": nat,
        "top_results": top,
        "metrics": metrics,
        "nb_auteurs": len(le.classes_),
    }


@app.route("/api/corpus/<base>")
def api_corpus(base):
    return jsonify(lister_auteurs(base))


# ──────────────────────────────────────────────────────────────────
# Transcription audio/vidéo (Whisper)
# ──────────────────────────────────────────────────────────────────
@app.route("/transcrire", methods=["GET", "POST"])
def transcrire_page():
    if request.method == "POST":
        mode = request.form.get("mode", "nouveau")
        nom_auteur = request.form.get("nom_auteur", "").strip()
        taille = request.form.get("taille", "small")
        fichier = request.files.get("fichier_media")

        if not fichier or not fichier.filename:
            flash("Veuillez téléverser un fichier audio ou vidéo.", "erreur")
            return redirect(url_for("transcrire_page"))
        if not whisper_disponible():
            flash("Whisper n'est pas installé (pip install openai-whisper + ffmpeg).", "erreur")
            return redirect(url_for("transcrire_page"))

        media_path = os.path.join(UPLOADS_DIR, fichier.filename)
        fichier.save(media_path)
        logs = []
        def log(m):
            logs.append(m); print(m)
        try:
            texte = transcrire(media_path, taille=taille, langue="ar", callback_log=log)
            if not texte.strip():
                flash("La transcription est vide.", "erreur")
                return redirect(url_for("transcrire_page"))
            dossier_auteur = (f"Auteur {prochain_numero('brut')}"
                              if mode == "nouveau" else nom_auteur)
            if not dossier_auteur:
                flash("Veuillez indiquer l'auteur.", "erreur")
                return redirect(url_for("transcrire_page"))
            chemin_dossier = os.path.join(CORPUS_BRUT, dossier_auteur)
            os.makedirs(chemin_dossier, exist_ok=True)
            existants = [f for f in os.listdir(chemin_dossier) if f.lower().endswith(".txt")]
            nom_fichier = f"texte{len(existants) + 1}.txt"
            texte = normaliser_invisibles(texte)
            with open(os.path.join(chemin_dossier, nom_fichier), "w", encoding="utf-8") as f:
                f.write(texte)
            try:
                os.remove(media_path)
            except OSError:
                pass
            flash(f"Transcription ajoutée : {dossier_auteur}/{nom_fichier}.", "succes")
            return render_template("transcrire.html",
                                   auteurs_brut=lister_auteurs("brut"),
                                   prochain_brut=prochain_numero("brut"),
                                   texte_transcrit=texte, logs=logs)
        except Exception as e:
            flash(f"Erreur transcription : {e}", "erreur")
            return redirect(url_for("transcrire_page"))

    return render_template("transcrire.html",
                           auteurs_brut=lister_auteurs("brut"),
                           prochain_brut=prochain_numero("brut"),
                           whisper_ok=whisper_disponible())


if __name__ == "__main__":
    print("=" * 70)
    print("  APPLICATION PFE V2 — Reconnaissance automatique d'auteurs arabes")
    print("  → http://localhost:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
