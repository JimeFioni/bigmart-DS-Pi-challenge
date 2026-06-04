"""
train.py
────────
Script principal de entrenamiento.

Integra:
  • Optuna  → búsqueda bayesiana de hiperparámetros (con pruning)
  • MLflow  → tracking de experimentos + Model Registry

Uso:
    python train.py                          # con defaults de config.yaml
    python train.py --n-trials 50            # más trials de Optuna
    python train.py --model xgboost          # solo XGBoost
    python train.py --model lgbm             # solo LightGBM
    python train.py --no-optuna              # entrena con params del config

Luego abrí la UI de MLflow con:
    mlflow ui
    # → http://localhost:5000
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb
import lightgbm as lgb

# Módulos propios
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing import load_data, validate_schema, DataCleaner
from src.features import FeatureEngineer, CategoricalEncoder, get_feature_matrix

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """
    Aplica limpieza + feature engineering y devuelve matrices listas para el modelo.
    El pipeline se fitea SOLO sobre train para evitar data leakage.
    """
    full = pd.concat([train_raw, test_raw], ignore_index=True)

    # Cleaner: se fitea sobre todo el dataset combinado solo para la imputación
    # de peso (necesitamos saber qué ítems existen en test).
    # El target (ventas) nunca se usa en fit.
    cleaner = DataCleaner()
    cleaner.fit(full)
    full_clean = cleaner.transform(full)

    train_clean = full_clean[full_clean["split"] == "train"].copy()
    test_clean  = full_clean[full_clean["split"] == "test"].copy()

    # Feature engineering: se fitea SOLO en train
    fe = FeatureEngineer()
    fe.fit(train_clean)
    train_fe = fe.transform(train_clean)
    test_fe  = fe.transform(test_clean)

    # Encoding: se fitea SOLO en train
    enc = CategoricalEncoder()
    enc.fit(train_fe)
    train_enc = enc.transform(train_fe)
    test_enc  = enc.transform(test_fe)

    X_train = get_feature_matrix(train_enc)
    y_train = train_clean["Item_Outlet_Sales"].values
    X_test  = get_feature_matrix(test_enc)

    log.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, y_train, X_test, test_raw


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def cv_rmse(model, X, y, n_splits: int = 5, seed: int = 42) -> tuple[float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(
        model, X, y, cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return -scores.mean(), scores.std()


# ── Optuna objectives ─────────────────────────────────────────────────────────

def xgboost_objective(trial: optuna.Trial, X, y, cfg: dict, parent_run_id: str):
    """Cada trial es un nested run de MLflow."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", *cfg["n_estimators"]),
        "max_depth":         trial.suggest_int("max_depth", *cfg["max_depth"]),
        "learning_rate":     trial.suggest_float("learning_rate", *cfg["learning_rate"], log=True),
        "subsample":         trial.suggest_float("subsample", *cfg["subsample"]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", *cfg["colsample_bytree"]),
        "reg_alpha":         trial.suggest_float("reg_alpha", *cfg["reg_alpha"], log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", *cfg["reg_lambda"], log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", *cfg["min_child_weight"]),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    model = xgb.XGBRegressor(**params)
    mean_rmse, std_rmse = cv_rmse(model, X, y)

    # Loggear cada trial como nested run
    with mlflow.start_run(nested=True, run_name=f"xgb_trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metrics({"cv_rmse": mean_rmse, "cv_rmse_std": std_rmse})
        mlflow.set_tag("trial_number", trial.number)

    return mean_rmse


def lgbm_objective(trial: optuna.Trial, X, y, cfg: dict, parent_run_id: str):
    params = {
        "n_estimators":   trial.suggest_int("n_estimators", *cfg["n_estimators"]),
        "max_depth":      trial.suggest_int("max_depth", *cfg["max_depth"]),
        "learning_rate":  trial.suggest_float("learning_rate", *cfg["learning_rate"], log=True),
        "num_leaves":     trial.suggest_int("num_leaves", *cfg["num_leaves"]),
        "subsample":      trial.suggest_float("subsample", *cfg["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg["colsample_bytree"]),
        "reg_alpha":      trial.suggest_float("reg_alpha", *cfg["reg_alpha"], log=True),
        "reg_lambda":     trial.suggest_float("reg_lambda", *cfg["reg_lambda"], log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", *cfg["min_child_samples"]),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    mean_rmse, std_rmse = cv_rmse(model, X, y)

    with mlflow.start_run(nested=True, run_name=f"lgbm_trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metrics({"cv_rmse": mean_rmse, "cv_rmse_std": std_rmse})
        mlflow.set_tag("trial_number", trial.number)

    return mean_rmse


# ── Training ──────────────────────────────────────────────────────────────────

def run_optuna_study(
    objective_fn,
    X, y,
    cfg: dict,
    n_trials: int,
    study_name: str,
    parent_run_id: str,
) -> tuple[dict, optuna.Study]:
    """Crea un estudio Optuna con pruning (MedianPruner) y lo corre."""
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    study.optimize(
        lambda trial: objective_fn(trial, X, y, cfg, parent_run_id),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    log.info(f"  Mejor trial #{study.best_trial.number}: CV RMSE = {study.best_value:.2f}")
    log.info(f"  Mejores params: {study.best_params}")
    return study.best_params, study


def train_final_model(model_name: str, params: dict, X, y):
    """Entrena el modelo final con los mejores hiperparámetros."""
    if model_name == "xgboost":
        model = xgb.XGBRegressor(**params)
    elif model_name == "lgbm":
        model = lgb.LGBMRegressor(**params)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    model.fit(X, y)
    return model


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_model_to_registry(
    model,
    model_name: str,
    params: dict,
    train_metrics: dict,
    cv_rmse_val: float,
    X_train,
    feature_names: list,
    study: optuna.Study,
):
    """Loggea el modelo final y lo registra en el MLflow Model Registry."""
    # Tags
    mlflow.set_tag("model_type", model_name)
    mlflow.set_tag("optuna_n_trials", len(study.trials))
    mlflow.set_tag("author", "Jimena Fioni")
    mlflow.set_tag("challenge", "PI Consulting DS Challenge")

    # Params
    mlflow.log_params(params)
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_train_rows", X_train.shape[0])

    # Métricas
    mlflow.log_metric("cv_rmse", cv_rmse_val)
    for k, v in train_metrics.items():
        mlflow.log_metric(f"train_{k}", v)

    # Artefactos: feature importance como CSV
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")

    # Optuna: loggear el mejor trial solo si hubo búsqueda
    if len(study.trials) > 0:
        mlflow.set_tag("best_trial", study.best_trial.number)
        best_trial_df = pd.DataFrame([{
            "trial_number": study.best_trial.number,
            "cv_rmse": study.best_value,
            **study.best_params,
        }])
        best_trial_df.to_csv("/tmp/best_trial.csv", index=False)
        mlflow.log_artifact("/tmp/best_trial.csv")

    # Log model
    registry_name = f"bigmart-{model_name}"
    if model_name == "xgboost":
        mlflow.xgboost.log_model(
            model, artifact_path="model",
            registered_model_name=registry_name,
        )
    else:
        mlflow.lightgbm.log_model(
            model, artifact_path="model",
            registered_model_name=registry_name,
        )

    log.info(f"  Modelo registrado en MLflow Model Registry como '{registry_name}'")


# ── XGBoost + RandomizedSearchCV (mejor modelo) ──────────────────────────────

def train_xgb_randsearch(X, y, feature_names: list):
    """Entrena el modelo ganador: XGBoost + RandomizedSearchCV sin log-transform."""
    param_dist = {
        "n_estimators":     [300, 500, 700, 900],
        "learning_rate":    [0.05, 0.1, 0.15],
        "max_depth":        [4, 6, 8],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha":        [0, 0.1, 1.0],
        "reg_lambda":       [1.0, 2.0, 5.0],
    }

    xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    rs = RandomizedSearchCV(
        xgb_base, param_dist,
        n_iter=30, cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=1, verbose=0,
    )
    rs.fit(X, y)

    best_model  = rs.best_estimator_
    best_params = rs.best_params_
    cv_rmse_val = -rs.best_score_

    train_preds  = best_model.predict(X)
    train_metrics = compute_metrics(y, train_preds)

    log.info(f"  CV RMSE: {cv_rmse_val:.2f}")
    log.info(f"  Train RMSE: {train_metrics['rmse']:.2f} | R²: {train_metrics['r2']:.4f}")
    log.info(f"  Mejores hiperparámetros: {best_params}")

    with mlflow.start_run(run_name="xgb_randsearch_WINNER"):
        mlflow.set_tag("model_type",  "xgboost_randomizedsearch")
        mlflow.set_tag("best_model",  "true")
        mlflow.set_tag("author",      "Jimena Fioni")
        mlflow.set_tag("challenge",   "PI Consulting DS Challenge")
        mlflow.set_tag("note",        "Mejor modelo — sin log-transform, max_depth conservador")

        mlflow.log_params(best_params)
        mlflow.log_param("n_iter",       30)
        mlflow.log_param("n_features",   X.shape[1])
        mlflow.log_param("n_train_rows", X.shape[0])
        mlflow.log_param("log_transform", False)

        mlflow.log_metric("cv_rmse", cv_rmse_val)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        feat_imp = pd.DataFrame({
            "feature":    feature_names,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        feat_imp.to_csv("/tmp/feature_importance_winner.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance_winner.csv")

        mlflow.xgboost.log_model(
            best_model, artifact_path="model",
            registered_model_name="bigmart-xgb-randsearch-winner",
        )
        log.info("  Modelo registrado en MLflow Model Registry como 'bigmart-xgb-randsearch-winner'")

    return best_model, best_params, cv_rmse_val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BigMart Sales — Training Pipeline")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--n-trials",  type=int, default=None,
                        help="Número de trials de Optuna (override del config)")
    parser.add_argument("--model",     choices=["xgboost", "lgbm", "both"], default="both")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Saltea Optuna y usa los params del config.yaml")
    parser.add_argument("--experiment", default=None,
                        help="Nombre del experimento MLflow (override del config)")
    args = parser.parse_args()

    # ── Cargar config
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    mlflow_cfg = cfg["mlflow"]
    n_trials = args.n_trials or cfg["optuna"]["n_trials"]
    experiment_name = args.experiment or mlflow_cfg["experiment_name"]

    # ── MLflow setup (SQLite backend — compatible con MLflow 3.x)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    log.info(f"MLflow experiment: '{experiment_name}'")

    # ── Datos
    log.info("Cargando datos...")
    train_raw, test_raw = load_data(data_cfg["train_path"], data_cfg["test_path"])
    validate_schema(train_raw)

    log.info("Aplicando preprocessing + feature engineering...")
    X_train, y_train, X_test, test_raw_orig = build_features(train_raw, test_raw)
    feature_names = list(X_train.columns)

    models_to_run = (
        ["xgboost", "lgbm"] if args.model == "both"
        else [args.model]
    )

    results = {}

    for model_name in models_to_run:
        log.info(f"\n{'='*55}")
        log.info(f"  Modelo: {model_name.upper()}")
        log.info(f"{'='*55}")

        model_cfg = cfg["models"][model_name]
        optuna_cfg = cfg["optuna"]["search_space"][model_name]

        with mlflow.start_run(run_name=f"{model_name}_final") as run:
            parent_run_id = run.info.run_id

            # ── Optuna
            if not args.no_optuna:
                log.info(f"  Iniciando búsqueda Optuna ({n_trials} trials)...")
                objective_fn = xgboost_objective if model_name == "xgboost" else lgbm_objective
                best_params, study = run_optuna_study(
                    objective_fn, X_train, y_train,
                    optuna_cfg, n_trials,
                    study_name=f"bigmart_{model_name}",
                    parent_run_id=parent_run_id,
                )
                best_params["random_state"] = 42
                if model_name == "xgboost":
                    best_params.update({"n_jobs": -1, "verbosity": 0})
                else:
                    best_params.update({"n_jobs": -1, "verbose": -1})
            else:
                log.info("  Usando hiperparámetros del config (--no-optuna)")
                best_params = model_cfg["default_params"]
                study = optuna.create_study()  # dummy para el logger

            # ── Entrenamiento final
            log.info("  Entrenando modelo final...")
            final_model = train_final_model(model_name, best_params, X_train, y_train)

            # ── Métricas
            cv_mean, cv_std = cv_rmse(final_model, X_train, y_train)
            train_preds = final_model.predict(X_train)
            train_metrics = compute_metrics(y_train, train_preds)

            log.info(f"  CV RMSE: {cv_mean:.2f} ± {cv_std:.2f}")
            log.info(f"  Train RMSE: {train_metrics['rmse']:.2f} | R²: {train_metrics['r2']:.4f}")

            # ── MLflow logging + Model Registry
            log_model_to_registry(
                final_model, model_name, best_params,
                train_metrics, cv_mean,
                X_train, feature_names, study,
            )

            # ── Predicciones en test
            test_preds = final_model.predict(X_test)
            results[model_name] = {
                "model": final_model,
                "preds": test_preds,
                "cv_rmse": cv_mean,
                "params": best_params,
            }

    # ── XGBoost + RandomizedSearchCV (modelo ganador)
    log.info(f"\n{'='*55}")
    log.info("  Modelo: XGB + RANDOMIZEDSEARCHCV (GANADOR)")
    log.info(f"{'='*55}")
    log.info("  Entrenando XGBoost + RandomizedSearchCV (30 iteraciones × 5 folds)...")
    winner_model, winner_params, winner_cv_rmse = train_xgb_randsearch(
        X_train, y_train, feature_names
    )
    winner_preds = winner_model.predict(X_test)
    results["xgb_randsearch"] = {
        "model":   winner_model,
        "preds":   winner_preds,
        "cv_rmse": winner_cv_rmse,
        "params":  winner_params,
    }

    # ── Ensemble ponderado (si corrimos ambos modelos)
    if len(results) == 2:
        log.info("\n📦 Generando ensemble ponderado...")
        rmse_xgb  = results["xgboost"]["cv_rmse"]
        rmse_lgbm = results["lgbm"]["cv_rmse"]
        total = 1/rmse_xgb + 1/rmse_lgbm
        w_xgb  = (1/rmse_xgb)  / total
        w_lgbm = (1/rmse_lgbm) / total

        ensemble_preds = (
            w_xgb  * results["xgboost"]["preds"] +
            w_lgbm * results["lgbm"]["preds"]
        )

        with mlflow.start_run(run_name="ensemble"):
            mlflow.log_params({"w_xgboost": round(w_xgb, 4), "w_lgbm": round(w_lgbm, 4)})
            mlflow.log_metric("xgb_cv_rmse",  rmse_xgb)
            mlflow.log_metric("lgbm_cv_rmse", rmse_lgbm)
    else:
        model_name_only = list(results.keys())[0]
        ensemble_preds = results[model_name_only]["preds"]

    # ── Guardar predicciones
    output_df = pd.DataFrame({
        "Item_Identifier":            test_raw_orig["Item_Identifier"].values,
        "Outlet_Identifier":          test_raw_orig["Outlet_Identifier"].values,
        "Item_Outlet_Sales_XGB":      results.get("xgboost",       {}).get("preds", np.nan),
        "Item_Outlet_Sales_LGB":      results.get("lgbm",          {}).get("preds", np.nan),
        "Item_Outlet_Sales_Ensemble": ensemble_preds,
        "Item_Outlet_Sales":          results.get("xgb_randsearch", {}).get("preds", ensemble_preds),
    })
    output_path = data_cfg.get("output_path", "predictions_test.csv")
    output_df.to_csv(output_path, index=False)
    log.info(f"\n✅ Predicciones guardadas en '{output_path}'")
    log.info("🎯 Para ver los experimentos: mlflow ui")


if __name__ == "__main__":
    main()
