# Reconnaissance d'auteurs arabes

Plateforme web (Flask) de classification de textes arabes par auteur.

## ✅ Fonctionnalités de cette version

- **Deux bases** : `corpus_brut/` et `corpus_nettoye/` (choix à chaque opération)
- **Nettoyage** intégré (bouton sur l'accueil) — reproduit `netoyage.py`
- **Caractérisation** :
  - N-gram **lettre** F1 / F2 (paramètre N)
  - N-gram **mot** F1 / F2 (paramètre N)
  - **TF-IDF** (paramètre TOP_N_MOTS)
- **Modèles** :
  - **CNN 1D** (architecture adaptée selon la méthode)
  - **SAE** (Sparse Autoencoder) :
    - joint (AE+classifieur) pour TF-IDF — reproduit `encodeur_tfidf2.py`
    - AE + LogisticRegression pour n-gram — reproduit `auto_encodeur_ngram.py`
- **Hyperparamètres réglables** : lr, batch_size, epochs, weight_decay, dropout, activation
  (ReLU/Sigmoid/Tanh/GELU), + spécifiques SAE (latent_dim, hidden, L1_lambda, W_recon, W_class)
- **Classification** d'un texte (fichier ou collé) avec top-5 + métriques

## 🔜 Modules à venir
- Embeddings AraBERT / AraELECTRA (calcul + CSV pré-calculés)
- Transcription audio/vidéo (Whisper local)

## 📌 Corpus inclus
Ton corpus nettoyé (16 auteurs × 10 textes) est déjà dans `corpus_nettoye/`.

## 📁 Structure

```
app_pfe/
├── app.py
├── requirements.txt
├── corpus_brut/          ← tes 16 dossiers Auteur ICI
├── corpus_nettoye/       ← généré par le bouton Nettoyer
├── uploads/
├── static/style.css
├── templates/
│   ├── base.html / index.html / enregistrer.html
│   ├── classifier.html   ← page principale (menus dynamiques)
│   └── resultat.html
└── utils/
    ├── nettoyage.py      ← reproduit netoyage.py
    ├── caracterisation.py← n-gram (mot/lettre F1/F2) + TF-IDF
    └── model_cnn.py      ← CNN_v5 (n-gram) + CNN_TFIDF
```

## 🚀 Lancement

```bash
pip install flask
# copier tes dossiers Auteur dans corpus_brut/
cd app_pfe_v2
python app.py
# http://localhost:5000
```

Dans Spyder : Set working directory → `app_pfe_v2/`, Restart kernel, puis F5.

## 📊 Hyperparamètres de référence 

| Méthode | lr | batch | epochs | autres |
|---------|-----|-------|--------|--------|
| N-gram CNN | 0.002 | 80 | 80 | dropout 0.4 |
| TF-IDF CNN | 0.0003 | 16 | 200 | top_n_mots 150, dropout 0.5 |

## 👤 Auteur
HAMADI Rafik
