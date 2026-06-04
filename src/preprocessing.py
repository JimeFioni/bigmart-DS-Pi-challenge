"""
src/preprocessing.py
────────────────────
Pipeline de limpieza y preprocesamiento para BigMart Sales.

Diseño con sklearn Pipeline + ColumnTransformer para que sea
completamente reproducible y evitar data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# ── Custom Transformers ────────────────────────────────────────────────────────

class FatContentNormalizer(BaseEstimator, TransformerMixin):
    """
    Normaliza las variantes de Item_Fat_Content y asigna 'Non-Edible'
    a productos del hogar / salud.
    """
    FAT_MAP = {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    NON_FOOD = {"Health and Hygiene", "Household", "Others"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(self.FAT_MAP)
        mask = df["Item_Type"].isin(self.NON_FOOD)
        df.loc[mask, "Item_Fat_Content"] = "Non-Edible"
        return df


class ItemWeightImputer(BaseEstimator, TransformerMixin):
    """
    Imputa Item_Weight con la media por Item_Identifier
    (el mismo producto tiene el mismo peso en todas las tiendas).
    Fallback: media global.
    """
    def fit(self, X, y=None):
        self.weight_map_ = (
            X.groupby("Item_Identifier")["Item_Weight"].mean().to_dict()
        )
        self.global_mean_ = X["Item_Weight"].mean()
        return self

    def transform(self, X):
        df = X.copy()
        mask = df["Item_Weight"].isna()
        df.loc[mask, "Item_Weight"] = df.loc[mask, "Item_Identifier"].map(
            self.weight_map_
        )
        df["Item_Weight"] = df["Item_Weight"].fillna(self.global_mean_)
        return df


class OutletSizeImputer(BaseEstimator, TransformerMixin):
    """
    Imputa Outlet_Size con la moda por Outlet_Type.
    Grocery Store → Small, Supermarket Type1 → Small, etc.
    """
    def fit(self, X, y=None):
        self.mode_map_ = (
            X.groupby("Outlet_Type")["Outlet_Size"]
            .agg(lambda s: s.mode()[0] if not s.mode().empty else "Small")
            .to_dict()
        )
        return self

    def transform(self, X):
        df = X.copy()
        mask = df["Outlet_Size"].isna()
        df.loc[mask, "Outlet_Size"] = df.loc[mask, "Outlet_Type"].map(
            self.mode_map_
        )
        return df


class VisibilityZeroFixer(BaseEstimator, TransformerMixin):
    """
    Reemplaza Item_Visibility == 0 con la media positiva por Item_Identifier.
    Visibilidad 0 indica error de registro, no un producto invisible.
    """
    def fit(self, X, y=None):
        self.vis_map_ = (
            X[X["Item_Visibility"] > 0]
            .groupby("Item_Identifier")["Item_Visibility"]
            .mean()
            .to_dict()
        )
        self.global_mean_ = X[X["Item_Visibility"] > 0]["Item_Visibility"].mean()
        return self

    def transform(self, X):
        df = X.copy()
        mask = df["Item_Visibility"] == 0
        df.loc[mask, "Item_Visibility"] = df.loc[mask, "Item_Identifier"].map(
            self.vis_map_
        ).fillna(self.global_mean_)
        return df


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Encadena todos los pasos de limpieza en orden.
    Equivale a un mini-pipeline interno.
    """
    def __init__(self):
        self.steps_ = [
            FatContentNormalizer(),
            ItemWeightImputer(),
            OutletSizeImputer(),
            VisibilityZeroFixer(),
        ]

    def fit(self, X, y=None):
        df = X.copy()
        for step in self.steps_:
            step.fit(df)
            df = step.transform(df)
        return self

    def transform(self, X):
        df = X.copy()
        for step in self.steps_:
            df = step.transform(df)
        return df


# ── Función de carga ───────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga train y test, les agrega columna 'split' y combina para
    que el cleaner pueda aprender de toda la distribución.

    Returns
    -------
    train_raw, test_raw : DataFrames sin mezclar, listos para DataCleaner.fit()
    """
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    train["split"] = "train"
    test["split"]  = "test"
    test["Item_Outlet_Sales"] = np.nan
    return train, test


def validate_schema(df: pd.DataFrame) -> None:
    """
    Valida que el DataFrame tenga las columnas esperadas.
    Lanza ValueError si falta alguna.
    """
    required = {
        "Item_Identifier", "Item_Weight", "Item_Fat_Content",
        "Item_Visibility", "Item_Type", "Item_MRP",
        "Outlet_Identifier", "Outlet_Establishment_Year",
        "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")
