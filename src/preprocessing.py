import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.feature_engineering import feature_engineering


def load_data(parquet_path):
    """Charge les données Parquet."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File {parquet_path} not found")

    parquet_data = pd.read_parquet(parquet_path)
    return parquet_data


def get_train_test_data(df, target_col='TARGET', test_size=0.2, random_state=42):
    """Sépare les features (X) et la target (y), puis fait le split train/test."""

    df_featured = feature_engineering(df)

    X = df_featured.drop(target_col, axis=1)
    y = df_featured[target_col]

    # Separation des donnees d'entrainement et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def create_preprocessor(X_train):
    """
    Crée le pipeline de preprocessing scikit-learn.
    Cette fonction doit détecter automatiquement les colonnes numériques et catégorielles.
    """
    # 1. Identifier les colonnes numériques et catégorielles
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # 2. Pipeline pour les données numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 3. Pipeline pour les données catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 4. Combiner les deux avec ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


if __name__ == "__main__":
    # Test rapide du script
    df = load_data("../data/processed/application_train.parquet")
    X_train, X_test, y_train, y_test = get_train_test_data(df)

    pipeline = create_preprocessor(X_train)

    # On fit juste pour voir si ça ne plante pas
    print("Test du pipeline de preprocessing...")
    X_train_processed = pipeline.fit_transform(X_train)
    print(f"Forme des données traitées : {X_train_processed.shape}")