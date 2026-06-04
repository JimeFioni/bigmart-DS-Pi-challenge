"""
src/features.py
───────────────
Feature engineering para BigMart Sales.

Todos los transformers son compatibles con sklearn Pipeline,
lo que garantiza que no haya data leakage entre train y test.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


# ── Feature creators ───────────────────────────────────────────────────────────

class OutletAgeFeature(BaseEstimator, TransformerMixin):
    """Outlet_Age = año de referencia del dataset (2013) - Outlet_Establishment_Year."""
    REFERENCE_YEAR = 2013

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["Outlet_Age"] = self.REFERENCE_YEAR - df["Outlet_Establishment_Year"]
        return df


class VisibilityRatioFeature(BaseEstimator, TransformerMixin):
    """
    Item_Visibility_MeanRatio = visibilidad del ítem / media de visibilidad del ítem.
    Captura si un producto está sobre- o sub-expuesto en comparación con su norma.
    Se aprende la media en fit() para evitar leakage.
    """
    def fit(self, X, y=None):
        self.mean_vis_ = (
            X.groupby("Item_Identifier")["Item_Visibility"].mean().to_dict()
        )
        return self

    def transform(self, X):
        df = X.copy()
        item_mean = df["Item_Identifier"].map(self.mean_vis_)
        df["Item_Visibility_MeanRatio"] = df["Item_Visibility"] / item_mean.replace(0, np.nan)
        df["Item_Visibility_MeanRatio"] = df["Item_Visibility_MeanRatio"].fillna(1.0)
        return df


class ItemCategoryFeature(BaseEstimator, TransformerMixin):
    """Agrupa Item_Type en 3 categorías amplias: Food, Drinks, Non-Consumable."""
    FOOD_MAP = {
        "Dairy": "Food", "Soft Drinks": "Drinks", "Meat": "Food",
        "Fruits and Vegetables": "Food", "Household": "Non-Consumable",
        "Baking Goods": "Food", "Snack Foods": "Food", "Frozen Foods": "Food",
        "Breakfast": "Food", "Health and Hygiene": "Non-Consumable",
        "Hard Drinks": "Drinks", "Canned": "Food", "Breads": "Food",
        "Starchy Foods": "Food", "Others": "Non-Consumable", "Seafood": "Food",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["Item_Category"] = df["Item_Type"].map(self.FOOD_MAP).fillna("Food")
        return df


class MRPTierFeature(BaseEstimator, TransformerMixin):
    """
    Discretiza Item_MRP en cuartiles (Budget / Mid / Premium / Luxury).
    Los bins se aprenden en fit() sobre el conjunto de entrenamiento.
    """
    LABELS = ["Budget", "Mid", "Premium", "Luxury"]

    def fit(self, X, y=None):
        _, self.bins_ = pd.qcut(X["Item_MRP"], q=4, labels=self.LABELS, retbins=True)
        return self

    def transform(self, X):
        df = X.copy()
        df["MRP_Tier"] = pd.cut(
            df["Item_MRP"], bins=self.bins_, labels=self.LABELS, include_lowest=True
        ).astype(str)
        return df


class ItemPrefixFeature(BaseEstimator, TransformerMixin):
    """
    Extrae el prefijo del código de ítem (FD = Food, DR = Drinks, NC = Non-Consumable).
    Sirve como proxy de categoría directamente desde el código de producto.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["Item_Prefix"] = df["Item_Identifier"].str[:2]
        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Encadena todos los feature creators.
    """
    def __init__(self):
        self.steps_ = [
            OutletAgeFeature(),
            VisibilityRatioFeature(),
            ItemCategoryFeature(),
            MRPTierFeature(),
            ItemPrefixFeature(),
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


# ── Encoding ───────────────────────────────────────────────────────────────────

CAT_COLS = [
    "Item_Fat_Content", "Item_Type", "Outlet_Identifier",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
    "Item_Category", "MRP_Tier", "Item_Prefix",
]

FEATURE_COLS = [
    "Item_Weight",
    "Item_Visibility",
    "Item_Visibility_MeanRatio",
    "Item_MRP",
    "Outlet_Age",
    # Encoded categoricals (added by CategoricalEncoder below)
    "Item_Fat_Content_enc",
    "Item_Type_enc",
    "Outlet_Identifier_enc",
    "Outlet_Size_enc",
    "Outlet_Location_Type_enc",
    "Outlet_Type_enc",
    "Item_Category_enc",
    "MRP_Tier_enc",
    "Item_Prefix_enc",
]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Ordinal encoding de variables categóricas.
    Aprende el mapeado en fit() para evitar leakage.
    """
    def fit(self, X, y=None):
        self.encoders_ = {}
        for col in CAT_COLS:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(X[[col]].astype(str))
            self.encoders_[col] = enc
        return self

    def transform(self, X):
        df = X.copy()
        for col, enc in self.encoders_.items():
            df[col + "_enc"] = enc.transform(df[[col]].astype(str)).astype(int)
        return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el subconjunto de columnas que van al modelo."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].fillna(0)
