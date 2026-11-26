"""Splits the data, trains the models and scores them."""

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


class ModelTraining:
    """Trains several regression models and compares them.

    Attributes:
        config (dict): Settings loaded from config.yaml.
        preprocessor: The preprocessing steps from DataPreparation.
    """

    def __init__(self, config: dict, preprocessor):
        self.config = config
        self.preprocessor = preprocessor

    def split_data(self, df: pd.DataFrame):
        """Split into train (80%), validation (10%) and test (10%)."""
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]
        seed = self.config["random_state"]
        holdout = self.config["val_size"] + self.config["test_size"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=holdout, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config["test_size"] / holdout, random_state=seed
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _on_log_scale(self, model):
        """Train the model on log(1 + shares), then convert predictions back to shares.

        Shares are heavily skewed by a few viral articles. The log scale stops
        those few from dominating the training.
        """
        return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)

    def build_models(self) -> dict:
        """Return the models to compare. The median baseline is the bar to beat."""
        seed = self.config["random_state"]
        forest = self.config["random_forest"]
        return {
            "baseline_median": DummyRegressor(strategy="median"),
            "linear": self._on_log_scale(LinearRegression()),
            "ridge": self._on_log_scale(Ridge()),
            "lasso": self._on_log_scale(Lasso(max_iter=5000)),
            "decision_tree": self._on_log_scale(DecisionTreeRegressor(random_state=seed)),
            "random_forest": self._on_log_scale(
                RandomForestRegressor(
                    n_estimators=forest["n_estimators"],
                    max_features=forest["max_features"],
                    random_state=seed,
                    n_jobs=-1,
                )
            ),
        }

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train every model. Models with a settings grid are tuned with cross-validation."""
        trained = {}
        for name, model in self.build_models().items():
            grid = self.config["param_grids"].get(name)
            if grid:
                model = GridSearchCV(
                    model, grid, cv=self.config["cv_folds"], scoring="neg_mean_absolute_error"
                )
            pipeline = Pipeline([("preprocess", self.preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            trained[name] = pipeline
        return trained

    def evaluate_model(self, pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
        """Score one model on data it has not been trained on.

        MAE:    average miss, in shares.
        RMSE:   like MAE, but big misses count much more.
        R2_log: share of the variation explained, measured on the log scale.
        """
        predictions = np.clip(pipeline.predict(X), 0, None)
        return {
            "MAE": mean_absolute_error(y, predictions),
            "RMSE": root_mean_squared_error(y, predictions),
            "R2_log": r2_score(np.log1p(y), np.log1p(predictions)),
        }

    def compare_models(self, trained: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Score every model and return a table sorted by MAE, best first."""
        rows = {name: self.evaluate_model(pipeline, X, y) for name, pipeline in trained.items()}
        return pd.DataFrame(rows).T.sort_values("MAE")
