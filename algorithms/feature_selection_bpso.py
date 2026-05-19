"""Feature Selection – Binary PSO trên Breast Cancer dataset."""
import numpy as np
import random
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(test_size=0.3, seed=42):
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, feature_names


def evaluate_subset(mask, X_tr, X_te, y_tr, y_te, alpha=0.9):
    if mask.sum() == 0:
        return 1.0  # tệ nhất
    idx = np.where(mask == 1)[0]
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_tr[:, idx], y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te[:, idx]))
    err = 1 - acc
    feat_ratio = mask.sum() / len(mask)
    return alpha * err + (1 - alpha) * feat_ratio


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def binary_pso_feature_selection(n_particles=20, n_iter=25, w=0.7, c1=1.5,
                                  c2=1.5, alpha=0.9, seed=42, progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    X_tr, X_te, y_tr, y_te, feat_names = load_data(seed=seed)
    D = X_tr.shape[1]

    X = np.random.randint(0, 2, size=(n_particles, D))
    V = np.random.uniform(-1, 1, size=(n_particles, D))
    pbest = X.copy()
    pbest_fit = np.array([evaluate_subset(x, X_tr, X_te, y_tr, y_te, alpha) for x in X])
    gbest = pbest[np.argmin(pbest_fit)].copy()
    gbest_fit = pbest_fit.min()
    history = []
    start_t = time.time()

    for it in range(n_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(D), np.random.rand(D)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            probs = sigmoid(V[i])
            X[i] = (np.random.rand(D) < probs).astype(int)
            f = evaluate_subset(X[i], X_tr, X_te, y_tr, y_te, alpha)
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = X[i].copy()
                if f < gbest_fit:
                    gbest_fit = f
                    gbest = X[i].copy()
        history.append(gbest_fit)
        if progress_cb:
            progress_cb(it + 1, n_iter, gbest_fit)

    runtime = time.time() - start_t

    # tính baseline & selected acc
    clf0 = LogisticRegression(max_iter=2000, random_state=42).fit(X_tr, y_tr)
    baseline_acc = accuracy_score(y_te, clf0.predict(X_te))
    idx = np.where(gbest == 1)[0]
    clf1 = LogisticRegression(max_iter=2000, random_state=42).fit(X_tr[:, idx], y_tr)
    selected_acc = accuracy_score(y_te, clf1.predict(X_te[:, idx]))

    return {
        "gbest": gbest, "gbest_fit": gbest_fit, "history": history,
        "runtime": runtime, "feature_names": feat_names,
        "baseline_acc": baseline_acc, "selected_acc": selected_acc,
        "n_features_total": D, "n_features_selected": int(gbest.sum()),
        "selected_indices": idx.tolist(),
    }
