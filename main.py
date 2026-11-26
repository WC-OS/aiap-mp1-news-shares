"""Runs the full pipeline: load, clean, split, train, compare, final test."""

import sys

import pandas as pd
import yaml

sys.path.append("src")
from data_preparation import DataPreparation
from model_training import ModelTraining

SCENARIOS = {
    "A. Before publishing (no reader comments)": False,
    "B. After publishing (reader comments available)": True,
}


def run_scenario(name: str, use_post_publication: bool, df: pd.DataFrame, config: dict) -> None:
    print(f"\n=== Scenario {name} ===")

    preparation = DataPreparation(config)
    clean_df = preparation.clean_data(df, use_post_publication)

    features = clean_df.drop(columns=config["target_column"])
    training = ModelTraining(config, preparation.create_preprocessor(features))
    X_train, X_val, X_test, y_train, y_val, y_test = training.split_data(clean_df)

    trained = training.train_models(X_train, y_train)

    print("\nValidation results (lower MAE is better):")
    results = training.compare_models(trained, X_val, y_val)
    print(results.round(3).to_string())

    best_name = results.drop(index="baseline_median").index[0]
    print(f"\nBest model on validation: {best_name}")

    print("Final check on the test set:")
    final = training.evaluate_model(trained[best_name], X_test, y_test)
    baseline = training.evaluate_model(trained["baseline_median"], X_test, y_test)
    print(pd.DataFrame({best_name: final, "baseline_median": baseline}).T.round(3).to_string())


def main() -> None:
    with open("src/config.yaml") as file:
        config = yaml.safe_load(file)

    df = pd.read_csv(config["file_path"])
    print(f"Loaded {len(df):,} articles with {df.shape[1]} columns.")

    for name, use_post_publication in SCENARIOS.items():
        run_scenario(name, use_post_publication, df, config)


if __name__ == "__main__":
    main()
