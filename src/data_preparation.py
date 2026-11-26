"""Cleans the raw article data and builds the preprocessing steps."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def log_squash(values):
    """Compress large numbers with log(1 + x). Negatives are treated as 0 first."""
    return np.log1p(np.clip(values, 0, None))


class DataPreparation:
    """Turns the raw CSV into clean features ready for a model.

    Attributes:
        config (dict): Settings loaded from config.yaml.
    """

    def __init__(self, config: dict):
        self.config = config

    def clean_data(self, df: pd.DataFrame, use_post_publication: bool) -> pd.DataFrame:
        """Apply the cleaning decisions made during EDA.

        Args:
            df: The raw data.
            use_post_publication: If False, drop columns only known after publishing.

        Returns:
            A cleaned copy of the data.
        """
        df = df.copy()

        # 1. Drop labels and the placeholder column
        to_drop = self.config["id_columns"] + self.config["columns_to_drop"]
        if not use_post_publication:
            to_drop = to_drop + self.config["post_publication_columns"]
        df = df.drop(columns=to_drop)

        # 2. Blank video count means "not recorded": flag it, then fill with 0
        for column in self.config["flag_missing_columns"]:
            df[f"{column}_missing"] = df[column].isna().astype(int)
            df[column] = df[column].fillna(0)

        # 3. Articles with no topic become their own category
        for column in self.config["categorical_features"]:
            df[column] = df[column].fillna(self.config["missing_category_label"])

        # 4. Share counts cannot be negative
        for column in self.config["clip_at_zero_columns"]:
            df[column] = df[column].clip(lower=0)

        return df

    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build the steps that run inside every model pipeline.

        Numbers: fill remaining blanks with the median, squash with log, then scale.
        Categories: turn each category into its own 0/1 column.

        The steps are fitted on the training data only, so nothing leaks from
        the validation or test data.
        """
        categorical = self.config["categorical_features"]
        numerical = [column for column in X.columns if column not in categorical]

        numerical_steps = Pipeline([
            ("fill_blanks", SimpleImputer(strategy="median")),
            ("log_squash", FunctionTransformer(log_squash)),
            ("scale", StandardScaler()),
        ])
        categorical_steps = OneHotEncoder(handle_unknown="ignore")

        return ColumnTransformer([
            ("numbers", numerical_steps, numerical),
            ("categories", categorical_steps, categorical),
        ])
