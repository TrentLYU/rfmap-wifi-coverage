"""Shared model utilities for single-router RSSI experiments."""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROUTER = np.array([1.0, 6.0])  # demo setting; overridden by optimize_wifi.py
SEEDS = [7, 19, 41]

def fit_ensemble(X, target, config):
    models, convergence_count = [], 0
    for seed in SEEDS:
        network = MLPRegressor(hidden_layer_sizes=tuple(config["layers"]), activation="tanh",
                               solver="lbfgs", alpha=config["alpha"], max_iter=6000,
                               max_fun=200000, tol=1e-5, random_state=seed)
        model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), network),
                                           transformer=StandardScaler())
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, target)
        convergence_count += sum(issubclass(w.category, ConvergenceWarning) for w in log)
        models.append(model)
    return models, convergence_count


def neural_predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def distance_feature(X):
    # Horizontal distance only: Tx/Rx heights have not been provided.
    # A 1 m floor keeps the empirical log-distance baseline away from its singularity.
    return np.log10(np.maximum(np.linalg.norm(X - ROUTER, axis=1), 1.0))


def fit_formula(X, target):
    ld = distance_feature(X)
    slope, intercept = np.linalg.lstsq(np.column_stack([ld, np.ones(len(ld))]), target, rcond=None)[0]
    if slope > 0:
        slope, intercept = 0.0, float(target.mean())
    return {"p0_dbm": float(intercept), "n": float(-slope / 10), "d_floor_m": 1.0}


def formula_predict(model, X):
    return model["p0_dbm"] - 10 * model["n"] * distance_feature(X)


def idw_predict(X_train, target, X, power):
    distance = np.linalg.norm(X[:, None, :] - X_train[None, :, :], axis=-1)
    weights = np.maximum(distance, 1e-9) ** (-power)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ target


def portable_model(models):
    result = []
    for model in models:
        scale, network = model.regressor_.steps[0][1], model.regressor_.steps[1][1]
        result.append({"input_mean": scale.mean_.tolist(), "input_scale": scale.scale_.tolist(),
                       "target_mean": model.transformer_.mean_.tolist(),
                       "target_scale": model.transformer_.scale_.tolist(),
                       "coefs": [a.tolist() for a in network.coefs_],
                       "intercepts": [a.tolist() for a in network.intercepts_],
                       "n_iter": int(network.n_iter_)})
    return {"input_order": ["x_m_up", "y_m_left"], "router_xy": ROUTER.tolist(),
            "activation": "tanh", "output_activation": "identity", "ensemble_aggregation": "mean",
            "models": result}
