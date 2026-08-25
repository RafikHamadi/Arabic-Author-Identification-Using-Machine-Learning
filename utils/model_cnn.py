# -*- coding: utf-8 -*-
"""
====================================================================
  Modèle CNN — supporte n-gram (CNN_v5) et TF-IDF (CNN_TFIDF)
  Hyperparamètres dynamiques + choix de fonction d'activation
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


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reset_seeds():
    """Réinitialise tous les générateurs aléatoires (reproductibilité inter-runs)."""
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
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


def num_key(s):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', str(s))]


def get_activation(nom):
    """Retourne le module d'activation selon le nom."""
    nom = (nom or "relu").lower()
    return {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
    }.get(nom, nn.ReLU())


# ══════════════════════════════════════════════════════════════════
# Architecture CNN_v5 (pour n-gram) — reproduit CNN_validé.py
# ══════════════════════════════════════════════════════════════════
class CNN_v5(nn.Module):
    def __init__(self, num_classes, activation="relu", dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ══════════════════════════════════════════════════════════════════
# Architecture CNN_TFIDF (pour TF-IDF) — reproduit cnntfidf_op.py
# ══════════════════════════════════════════════════════════════════
class CNN_TFIDF(nn.Module):
    def __init__(self, num_features, num_classes, activation="relu", dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=10, padding=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool2(self.act(self.bn2(self.conv2(x))))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)


# ══════════════════════════════════════════════════════════════════
# Encodage des labels (tri naturel)
# ══════════════════════════════════════════════════════════════════
def encoder_labels(labels):
    unique_authors = sorted(np.unique(labels), key=num_key)
    le = LabelEncoder()
    le.classes_ = np.array(unique_authors)
    y = np.array([list(le.classes_).index(l) for l in labels], dtype=np.int64)
    return y, le


def _evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            out = model(X_b)
            loss = criterion(out, y_b)
            total_loss += loss.item() * X_b.size(0)
            correct += (out.argmax(1) == y_b).sum().item()
            total += X_b.size(0)
    return correct / total, total_loss / total


# ══════════════════════════════════════════════════════════════════
# Entraînement CNN n-gram
# ══════════════════════════════════════════════════════════════════
def entrainer_cnn_ngram(X, labels, hp, callback_log=None):
    """
    X : (n_textes, n_features). hp : dict d'hyperparamètres.
    Normalisation MinMax (comme CNN_validé.py).
    """
    _reset_seeds()
    if callback_log is None:
        callback_log = print

    y, le = encoder_labels(labels)
    num_classes = len(le.classes_)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    model, metrics = _boucle_entrainement(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_factory=lambda: CNN_v5(
            num_classes=num_classes,
            activation=hp.get("activation", "relu"),
            dropout=hp.get("dropout", 0.4),
        ),
        hp=hp, callback_log=callback_log,
    )
    return model, metrics, le, scaler, None


# ══════════════════════════════════════════════════════════════════
# Entraînement CNN TF-IDF
# ══════════════════════════════════════════════════════════════════
def entrainer_cnn_tfidf(X_full, labels, top_indices_holder, hp, callback_log=None):
    """
    X_full : (n_docs, n_mots). Sélection top-N + L2-norm (comme cnntfidf_op).
    top_indices_holder : liste qui recevra les indices sélectionnés (pour prédiction).
    """
    _reset_seeds()
    if callback_log is None:
        callback_log = print

    from utils.caracterisation import selectionner_top_mots

    y, le = encoder_labels(labels)
    num_classes = len(le.classes_)
    top_n_mots = hp.get("top_n_mots", 150)

    indices_all = np.arange(len(X_full))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices_all, y, test_size=0.40, random_state=SEED, stratify=y)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    # Sélection top-N sur train
    if top_n_mots and top_n_mots < X_full.shape[1]:
        top_indices = selectionner_top_mots(X_full[idx_train], y_train, top_n_mots, le.classes_)
    else:
        top_indices = np.arange(X_full.shape[1])
    top_indices_holder.append(top_indices)
    callback_log(f"   Top-{len(top_indices)} mots sélectionnés")

    X_train = normalize(X_full[idx_train][:, top_indices], norm='l2', axis=1).astype(np.float32)
    X_val = normalize(X_full[idx_val][:, top_indices], norm='l2', axis=1).astype(np.float32)
    X_test = normalize(X_full[idx_test][:, top_indices], norm='l2', axis=1).astype(np.float32)
    n_features = X_train.shape[1]

    model, metrics = _boucle_entrainement(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_factory=lambda: CNN_TFIDF(
            num_features=n_features, num_classes=num_classes,
            activation=hp.get("activation", "relu"),
            dropout=hp.get("dropout", 0.5),
        ),
        hp=hp, callback_log=callback_log, grad_clip=True,
    )
    return model, metrics, le, None, top_indices


# ══════════════════════════════════════════════════════════════════
# Boucle d'entraînement générique
# ══════════════════════════════════════════════════════════════════
def _boucle_entrainement(X_train, y_train, X_val, y_val, X_test, y_test,
                          model_factory, hp, callback_log, grad_clip=False):
    lr = hp.get("lr", 0.001)
    batch_size = hp.get("batch_size", 16)
    epochs = hp.get("epochs", 80)
    weight_decay = hp.get("weight_decay", 5e-4)

    def make_loader(Xn, yn, bs, shuffle=True):
        Xt = torch.from_numpy(Xn).unsqueeze(1)
        yt = torch.from_numpy(yn)
        gen = _make_loader_generator() if shuffle else None
        return DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=shuffle, generator=gen)

    loader_train = make_loader(X_train, y_train, batch_size, True)
    loader_val = make_loader(X_val, y_val, batch_size, False)
    loader_test = make_loader(X_test, y_test, batch_size, False)

    model = model_factory().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc, best_state, best_epoch = 0.0, None, 0

    callback_log(f"   Dispositif : {DEVICE}")
    callback_log(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    callback_log(f"   Hyperparams : lr={lr}, batch={batch_size}, epochs={epochs}, "
                 f"act={hp.get('activation','relu')}, dropout={hp.get('dropout')}")
    callback_log("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in loader_train:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item() * X_b.size(0)
            correct += (out.argmax(1) == y_b).sum().item()
            total += X_b.size(0)

        train_acc = correct / total
        train_loss = run_loss / total
        val_acc, val_loss = _evaluate(model, loader_val, criterion)
        scheduler.step()

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            callback_log(f"   Epoch {epoch:3d}/{epochs} | Train {train_acc*100:5.2f}% | "
                         f"Val {val_acc*100:5.2f}% | Loss {train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, _ = _evaluate(model, loader_test, criterion)
    train_acc_f, _ = _evaluate(model, loader_train, criterion)
    val_acc_f, _ = _evaluate(model, loader_val, criterion)

    callback_log("=" * 70)
    callback_log(f"   ► Train: {train_acc_f*100:.2f}% | Val: {val_acc_f*100:.2f}% | Test: {test_acc*100:.2f}%")

    return model, {
        "train_acc": train_acc_f,
        "val_acc": val_acc_f,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
    }


# ══════════════════════════════════════════════════════════════════
# Prédiction
# ══════════════════════════════════════════════════════════════════
def predire_ngram(model, vecteur, scaler, le, top_k=5):
    vec = scaler.transform(vecteur.reshape(1, -1)).astype(np.float32)
    model.eval()
    X_t = torch.from_numpy(vec).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(X_t), dim=1).cpu().numpy()[0]
    return _top_results(probs, le, top_k)


def predire_tfidf(model, vec_full, top_indices, le, top_k=5):
    vec_reduit = vec_full[top_indices].astype(np.float32)
    vec_reduit = normalize(vec_reduit.reshape(1, -1), norm='l2', axis=1)[0]
    model.eval()
    X_t = torch.from_numpy(vec_reduit).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(X_t), dim=1).cpu().numpy()[0]
    return _top_results(probs, le, top_k)


# ══════════════════════════════════════════════════════════════════
# Entraînement CNN embeddings (StandardScaler) — reproduit cnn_embedding_3.py
# ══════════════════════════════════════════════════════════════════
def entrainer_cnn_embedding(X, labels, hp, callback_log=None):
    """
    X : (n_docs, dim) embeddings. StandardScaler (comme cnn_embedding_3.py).
    Utilise CNN_v5.
    """
    _reset_seeds()
    if callback_log is None:
        callback_log = print

    from sklearn.preprocessing import StandardScaler

    y, le = encoder_labels(labels)
    num_classes = len(le.classes_)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    model, metrics = _boucle_entrainement(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_factory=lambda: CNN_v5(
            num_classes=num_classes,
            activation=hp.get("activation", "relu"),
            dropout=hp.get("dropout", 0.2)),
        hp=hp, callback_log=callback_log)
    return model, metrics, le, scaler, None


def predire_embedding_cnn(model, vecteur, scaler, le, top_k=5):
    vec = scaler.transform(vecteur.reshape(1, -1)).astype(np.float32)
    model.eval()
    X_t = torch.from_numpy(vec).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(X_t), dim=1).cpu().numpy()[0]
    return _top_results(probs, le, top_k)


def _top_results(probs, le, top_k):
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_results = [(le.classes_[i], float(probs[i])) for i in top_idx]
    return top_results[0][0], top_results
