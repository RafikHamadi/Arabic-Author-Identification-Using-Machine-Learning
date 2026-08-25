# -*- coding: utf-8 -*-
"""
====================================================================
  Modèle SAE (Sparse Autoencoder) — 2 variantes
  - SAE joint     : pour TF-IDF (reproduit encodeur_tfidf2.py)
                    AE + classifieur entraînés ensemble (MSE+CE+L1)
  - SAE + LogReg  : pour n-gram (reproduit auto_encodeur_ngram.py)
                    AE seul (MSE) puis LogisticRegression sur le latent
====================================================================
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reset_seeds():
    """Réinitialise tous les générateurs aléatoires pour garantir
    la reproductibilité d'un entraînement à l'autre dans le même processus."""
    import random as _random
    _random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _make_loader_generator():
    """Generator dédié au DataLoader pour reproductibilité du shuffle."""
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


def num_key(s):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', str(s))]


def get_activation(nom):
    nom = (nom or "relu").lower()
    return {
        "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(), "gelu": nn.GELU(),
    }.get(nom, nn.ReLU())


def encoder_labels(labels):
    unique_authors = sorted(np.unique(labels), key=num_key)
    le = LabelEncoder()
    le.classes_ = np.array(unique_authors)
    y = np.array([list(le.classes_).index(l) for l in labels], dtype=np.int64)
    return y, le


# ══════════════════════════════════════════════════════════════════
# VARIANTE 1 : SAE JOINT (TF-IDF) — reproduit encodeur_tfidf2.py
# ══════════════════════════════════════════════════════════════════
class SparseAEClassifier(nn.Module):
    """
    AE + classifieur entraînés conjointement.
    enc: input -> hidden (BN, activation, Dropout) -> latent (ReLU)
    dec: latent -> hidden (activation) -> input
    clf: latent -> hidden_clf (BN, activation, Dropout) -> num_classes
    """
    def __init__(self, input_dim, hidden=256, latent=128, hidden_clf=64,
                 num_classes=16, dropout=0.3, activation="gelu"):
        super().__init__()
        act = get_activation(activation)
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            get_activation(activation),
            nn.Dropout(0.2),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden),
            get_activation(activation),
            nn.Linear(hidden, input_dim),
        )
        self.clf = nn.Sequential(
            nn.Linear(latent, hidden_clf),
            nn.BatchNorm1d(hidden_clf),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_clf, num_classes),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z, self.clf(z)

    def encode(self, x):
        return self.enc(x)


def _eval_joint(model, loader, mse, ce, hp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            x_rec, z, logits = model(xb)
            loss = (hp["w_class"] * ce(logits, yb)
                    + hp["w_recon"] * mse(x_rec, xb)
                    + hp["l1_lambda"] * z.abs().mean())
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total, total_loss / total


def entrainer_sae_tfidf(X_full, labels, top_holder, hp, callback_log=None):
    """
    SAE joint pour TF-IDF.
    Sélection top-N + L2-norm (comme encodeur_tfidf2.py).
    """
    _reset_seeds()
    if callback_log is None:
        callback_log = print

    from utils.caracterisation import selectionner_top_mots

    y, le = encoder_labels(labels)
    num_classes = len(le.classes_)
    top_n_mots = hp.get("top_n_mots", 1000)

    indices_all = np.arange(len(X_full))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        indices_all, y, test_size=0.40, random_state=SEED, stratify=y)
    idx_va, idx_te, y_va, y_te = train_test_split(
        idx_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    if top_n_mots and top_n_mots < X_full.shape[1]:
        top_indices = selectionner_top_mots(X_full[idx_tr], y_tr, top_n_mots, le.classes_)
    else:
        top_indices = np.arange(X_full.shape[1])
    top_holder.append(top_indices)

    X_tr = normalize(X_full[idx_tr][:, top_indices], norm='l2', axis=1).astype(np.float32)
    X_va = normalize(X_full[idx_va][:, top_indices], norm='l2', axis=1).astype(np.float32)
    X_te = normalize(X_full[idx_te][:, top_indices], norm='l2', axis=1).astype(np.float32)
    input_dim = X_tr.shape[1]
    callback_log(f"   Top-{input_dim} mots | L2-norm")

    bs = hp.get("batch_size", 16)
    loader_gen = _make_loader_generator()
    def make_loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya).long())
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          generator=loader_gen if shuffle else None)

    train_ld = make_loader(X_tr, y_tr, True)
    val_ld = make_loader(X_va, y_va, False)
    test_ld = make_loader(X_te, y_te, False)

    model = SparseAEClassifier(
        input_dim=input_dim, hidden=hp.get("hidden", 256),
        latent=hp.get("latent_dim", 128), hidden_clf=hp.get("hidden_clf", 64),
        num_classes=num_classes, dropout=hp.get("dropout", 0.3),
        activation=hp.get("activation", "gelu"),
    ).to(DEVICE)

    epochs = hp.get("epochs", 500)
    lr = hp.get("lr", 0.0005)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=hp.get("weight_decay", 5e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc, best_state, best_epoch = 0.0, None, 0
    callback_log(f"   Dispositif : {DEVICE}")
    callback_log(f"   Train: {len(idx_tr)} | Val: {len(idx_va)} | Test: {len(idx_te)}")
    callback_log(f"   SAE joint : lr={lr}, batch={bs}, epochs={epochs}, "
                 f"latent={hp.get('latent_dim',128)}, L1={hp.get('l1_lambda')}, "
                 f"W_recon={hp.get('w_recon')}")
    callback_log("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            x_rec, z, logits = model(xb)
            loss = (hp["w_class"] * ce(logits, yb)
                    + hp["w_recon"] * mse(x_rec, xb)
                    + hp["l1_lambda"] * z.abs().mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        tr_acc = correct / total
        va_acc, va_loss = _eval_joint(model, val_ld, mse, ce, hp)

        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            callback_log(f"   Epoch {epoch:3d}/{epochs} | Train {tr_acc*100:5.2f}% | "
                         f"Val {va_acc*100:5.2f}% | Loss {va_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    tr_acc_f, _ = _eval_joint(model, train_ld, mse, ce, hp)
    va_acc_f, _ = _eval_joint(model, val_ld, mse, ce, hp)
    te_acc_f, _ = _eval_joint(model, test_ld, mse, ce, hp)

    callback_log("=" * 70)
    callback_log(f"   ► Train: {tr_acc_f*100:.2f}% | Val: {va_acc_f*100:.2f}% | Test: {te_acc_f*100:.2f}%")

    metrics = {"train_acc": tr_acc_f, "val_acc": va_acc_f, "test_acc": te_acc_f, "best_epoch": best_epoch}
    return model, metrics, le, top_indices


def predire_sae_tfidf(model, vec_full, top_indices, le, top_k=5):
    vec = vec_full[top_indices].astype(np.float32)
    vec = normalize(vec.reshape(1, -1), norm='l2', axis=1)[0]
    model.eval()
    X_t = torch.from_numpy(vec).unsqueeze(0).to(DEVICE)
    # BatchNorm exige batch>1 en train, mais en eval c'est OK avec batch=1
    with torch.no_grad():
        _, _, logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return _top_results(probs, le, top_k)


# ══════════════════════════════════════════════════════════════════
# SAE JOINT pour EMBEDDINGS (StandardScaler) — reproduit autoencodeur33.py
# ══════════════════════════════════════════════════════════════════
def entrainer_sae_embedding(X, labels, hp, callback_log=None):
    """
    SAE joint pour embeddings. StandardScaler (comme autoencodeur33.py).
    Pas de sélection top-N (les dims sont déjà denses).
    """
    _reset_seeds()
    if callback_log is None:
        callback_log = print

    from sklearn.preprocessing import StandardScaler

    y, le = encoder_labels(labels)
    num_classes = len(le.classes_)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=SEED, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    input_dim = X_tr.shape[1]

    bs = hp.get("batch_size", 16)
    loader_gen = _make_loader_generator()
    def make_loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya).long())
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          generator=loader_gen if shuffle else None)

    train_ld = make_loader(X_tr, y_tr, True)
    val_ld = make_loader(X_va, y_va, False)
    test_ld = make_loader(X_te, y_te, False)

    model = SparseAEClassifier(
        input_dim=input_dim, hidden=hp.get("hidden", 500),
        latent=hp.get("latent_dim", 128), hidden_clf=hp.get("hidden_clf", 128),
        num_classes=num_classes, dropout=hp.get("dropout", 0.3),
        activation=hp.get("activation", "gelu")).to(DEVICE)

    epochs = hp.get("epochs", 400)
    lr = hp.get("lr", 0.001)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=hp.get("weight_decay", 5e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc, best_state, best_epoch = 0.0, None, 0
    callback_log(f"   Dispositif : {DEVICE}")
    callback_log(f"   Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")
    callback_log(f"   SAE embed : lr={lr}, batch={bs}, epochs={epochs}, latent={hp.get('latent_dim',128)}")
    callback_log("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            x_rec, z, logits = model(xb)
            loss = (hp["w_class"] * ce(logits, yb)
                    + hp["w_recon"] * mse(x_rec, xb)
                    + hp["l1_lambda"] * z.abs().mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        tr_acc = correct / total
        va_acc, va_loss = _eval_joint(model, val_ld, mse, ce, hp)
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            callback_log(f"   Epoch {epoch:3d}/{epochs} | Train {tr_acc*100:5.2f}% | "
                         f"Val {va_acc*100:5.2f}% | Loss {va_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    tr_acc_f, _ = _eval_joint(model, train_ld, mse, ce, hp)
    va_acc_f, _ = _eval_joint(model, val_ld, mse, ce, hp)
    te_acc_f, _ = _eval_joint(model, test_ld, mse, ce, hp)
    callback_log("=" * 70)
    callback_log(f"   ► Train: {tr_acc_f*100:.2f}% | Val: {va_acc_f*100:.2f}% | Test: {te_acc_f*100:.2f}%")

    metrics = {"train_acc": tr_acc_f, "val_acc": va_acc_f, "test_acc": te_acc_f, "best_epoch": best_epoch}
    return model, metrics, le, scaler


def predire_sae_embedding(model, vecteur, scaler, le, top_k=5):
    vec = scaler.transform(vecteur.reshape(1, -1)).astype(np.float32)
    model.eval()
    X_t = torch.from_numpy(vec).to(DEVICE)
    with torch.no_grad():
        _, _, logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return _top_results(probs, le, top_k)


# ══════════════════════════════════════════════════════════════════
# VARIANTE 2 : SAE + LogReg (n-gram) — reproduit auto_encodeur_ngram.py
# ══════════════════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    """
    AE pur (sans classifieur). enc 256->128->latent, dec latent->128->256->input+Sigmoid.
    """
    def __init__(self, input_dim, latent_dim=64, dropout=0.4, activation="relu"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            get_activation(activation),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def entrainer_sae_ngram(X, labels, hp, callback_log=None):
    """
    SAE + LogReg pour n-gram.
    1) Entraîne l'AE (MSE) avec MinMax. 2) Extrait le latent. 3) LogReg.
    """
    _reset_seeds()
    if callback_log is None:
        callback_log = print

    y, le = encoder_labels(labels)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    input_dim = X.shape[1]
    latent_dim = hp.get("latent_dim", 64)
    bs = hp.get("batch_size", 48)
    epochs = hp.get("epochs", 10)
    lr = hp.get("lr", 0.001)

    Xtr_t = torch.from_numpy(X_train)
    ds = TensorDataset(Xtr_t)
    loader_train = DataLoader(ds, batch_size=bs, shuffle=True,
                              generator=_make_loader_generator())

    model = Autoencoder(input_dim, latent_dim, hp.get("dropout", 0.4),
                        hp.get("activation", "relu")).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=hp.get("weight_decay", 5e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    callback_log(f"   Dispositif : {DEVICE}")
    callback_log(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    callback_log(f"   SAE+LogReg : lr={lr}, batch={bs}, epochs={epochs}, latent={latent_dim}")
    callback_log("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, total = 0.0, 0
        for (X_b,) in loader_train:
            X_b = X_b.to(DEVICE)
            optimizer.zero_grad()
            x_hat, _ = model(X_b)
            loss = criterion(x_hat, X_b)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * X_b.size(0)
            total += X_b.size(0)
        train_loss = run_loss / total
        model.eval()
        with torch.no_grad():
            x_hat_val, _ = model(X_val_t)
            val_loss = criterion(x_hat_val, X_val_t).item()
        scheduler.step()
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            callback_log(f"   Epoch {epoch:3d}/{epochs} | Train MSE {train_loss:.6f} | Val MSE {val_loss:.6f}")

    # Extraction latent
    model.eval()
    with torch.no_grad():
        Z_train = model.encode(torch.from_numpy(X_train).to(DEVICE)).cpu().numpy()
        Z_val = model.encode(torch.from_numpy(X_val).to(DEVICE)).cpu().numpy()
        Z_test = model.encode(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()

    callback_log("   Classification LogisticRegression sur le latent...")
    clf = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs", random_state=SEED)
    clf.fit(Z_train, y_train)

    tr_acc = accuracy_score(y_train, clf.predict(Z_train))
    va_acc = accuracy_score(y_val, clf.predict(Z_val))
    te_acc = accuracy_score(y_test, clf.predict(Z_test))

    callback_log("=" * 70)
    callback_log(f"   ► Train: {tr_acc*100:.2f}% | Val: {va_acc*100:.2f}% | Test: {te_acc*100:.2f}%")

    metrics = {"train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc, "best_epoch": epochs}
    return model, metrics, le, scaler, clf


def predire_sae_ngram(model, clf, vecteur, scaler, le, top_k=5):
    vec = scaler.transform(vecteur.reshape(1, -1)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        z = model.encode(torch.from_numpy(vec).to(DEVICE)).cpu().numpy()
    probs = clf.predict_proba(z)[0]
    # clf.classes_ sont des indices ; on les mappe aux noms
    full_probs = np.zeros(len(le.classes_))
    for i, cls_idx in enumerate(clf.classes_):
        full_probs[cls_idx] = probs[i]
    return _top_results(full_probs, le, top_k)


# ══════════════════════════════════════════════════════════════════
def _top_results(probs, le, top_k):
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_results = [(le.classes_[i], float(probs[i])) for i in top_idx]
    return top_results[0][0], top_results
