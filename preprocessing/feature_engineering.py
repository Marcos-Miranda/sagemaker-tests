import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple, Any
import json
import os
import sys
import argparse
import subprocess
import boto3

# Install additional packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn==0.7.*"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "boruta==0.3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])


from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from boruta import BorutaPy
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from datetime import datetime

CAT_FEATURES = ["g", "j", "n", "o", "p", "dia_semana"]
NUM_FEATURES = ["a", "b", "c", "d", "e", "f", "h", "k", "l", "m", "monto", "dia_mes", "hora", "text_feat"]


def fix_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Fix column values to correspond to their types."""

    df["n"] = np.select([df["n"] == 1, df["n"] == 0, df["n"].isna()], ["Y", "N", np.nan])

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-related features from the transaction's timestamp."""

    time_column = "fecha"
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df["dia_mes"] = df[time_column].dt.day
    df["dia_semana"] = df[time_column].dt.day_name()
    df["hora"] = df[time_column].dt.round("H").dt.hour
    df.drop(columns=time_column, inplace=True)

    return df


def dataset_split(df: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, ...]:
    """Split the dataframe into train and test sets."""

    train_set, dev_set = train_test_split(df, test_size=1 - train_size, stratify=df["fraude"], random_state=15)
    dev_set, test_set = train_test_split(dev_set, test_size=0.5, stratify=dev_set["fraude"], random_state=15)

    return (train_set, dev_set, test_set)


def create_text_feature(dfs: Tuple[pd.DataFrame, ...]) -> Tuple[Any]:
    """Create a feature from the text column by training a Logistic Regression with Bag of Words."""

    pipe = Pipeline(
        [
            ("bow", CountVectorizer(ngram_range=(1, 2))),
            ("over", RandomOverSampler(random_state=15)),
            ("lr", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )

    text_column = "i"
    train_set, dev_set, test_set = dfs

    train_set["text_feat"] = cross_val_predict(
        pipe, train_set[text_column], train_set["fraude"], cv=10, method="predict_proba"
    )[:, 1]
    pipe.fit(train_set[text_column], train_set["fraude"])
    dev_set["text_feat"] = pipe.predict_proba(dev_set[text_column])[:, 1]
    test_set["text_feat"] = pipe.predict_proba(test_set[text_column])[:, 1]

    text_model = pipe.fit(
        pd.concat([train_set[text_column], dev_set[text_column], test_set[text_column]]),
        pd.concat([train_set["fraude"], dev_set["fraude"], test_set["fraude"]]),
    )

    train_set.drop(columns=text_column, inplace=True)
    dev_set.drop(columns=text_column, inplace=True)
    test_set.drop(columns=text_column, inplace=True)

    return (text_model, train_set, dev_set, test_set)


def feature_selection(dfs: Tuple[pd.DataFrame, ...]) -> Tuple[pd.DataFrame, ...]:
    """Select the most import features with Boruta."""

    train_set, dev_set, test_set = dfs
    X_boruta = train_set.drop(columns=["fraude", "id"])

    for col in CAT_FEATURES:
        X_boruta[col] = LabelEncoder().fit_transform(X_boruta[col].fillna("missing"))
    for col in NUM_FEATURES:
        X_boruta[col] = X_boruta[col].fillna(X_boruta[col].mean())

    rf_model = RandomForestClassifier(max_depth=5, class_weight="balanced", n_jobs=-1)
    boruta = BorutaPy(rf_model, n_estimators="auto", verbose=2)
    boruta.fit(X_boruta.values, train_set["fraude"].values)
    selected_features = X_boruta.columns[boruta.support_].tolist()

    train_set = train_set[selected_features + ["id", "fraude"]]
    dev_set = dev_set[selected_features + ["id", "fraude"]]
    test_set = test_set[selected_features + ["id", "fraude"]]

    return (train_set, dev_set, test_set)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--feature_store_prefix", type=str, default="feature-store")
    args = parser.parse_args()
    print(f"Received arguments: {args}")

    output_path = "/opt/ml/processing/output"

    print("Reading the input file...")
    input_data_path = os.path.join("/opt/ml/processing/input", "dados_fraude.tsv")
    df = pd.read_csv(input_data_path, sep="\t")

    print("Fixing column types...")
    df = fix_column_types(df)

    print("Creating time-related features...")
    df = create_time_features(df)

    print("Splitting the dataset into train, dev and test sets...")
    train_set, dev_set, test_set = dataset_split(df, args.train_size)

    print("Creating a feature from the text column...")
    text_model, train_set, dev_set, test_set = create_text_feature((train_set, dev_set, test_set))
    print("Writing the text model to the output directory...")
    joblib.dump(text_model, os.path.join(output_path, "text_model/text_model.joblib"))

    print("Selecting the most important features with Boruta...")
    train_set, dev_set, test_set = feature_selection((train_set, dev_set, test_set))

    print("Writing the output datasets...")
    train_set.to_csv(os.path.join(output_path, "train/train.csv"), index=False)
    dev_set.to_csv(os.path.join(output_path, "dev/dev.csv"), index=False)
    test_set.to_csv(os.path.join(output_path, "test/test.csv"), index=False)

    print("Setting up the feature store...")
    sagemaker_session = sagemaker.local.LocalSession()
    region = sagemaker_session.boto_region_name
    print(region)
    boto_session = boto3.Session(region_name=region)
    role = sagemaker.get_execution_role()
    print(role)
    default_bucket = sagemaker_session.default_bucket()
    prefix = args.feature_store_prefix
    offline_feature_store_bucket = f"s3://{default_bucket}/{prefix}"
    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    featurestore_runtime = boto_session.client(service_name="sagemaker-featurestore-runtime", region_name=region)
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )
    feature_group_name = "fraude-feature-group"
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    print("Preparing the data for the feature store...")
    train_set.loc[:, "split"] = "train"
    dev_set.loc[:, "split"] = "dev"
    test_set.loc[:, "split"] = "test"
    df_fg = pd.concat([train_set, dev_set, test_set])
    df_fg["EventTime"] = [datetime.now().timestamp()] * len(df_fg)
    for col in df_fg.columns:
        if df_fg[col].dtype == "object":
            df_fg[col] = df_fg[col].astype("string")
    feature_group.load_feature_definitions(df_fg)
    feature_group.create(
        s3_uri=offline_feature_store_bucket,
        record_identifier_name="id",
        event_time_feature_name="EventTime",
        role_arn=role,
        enable_online_store=True,
    )
    print("Sending the data to the feature store...")
    feature_group.ingest(df_fg, max_workers=3)

    selected_features = train_set.drop(columns=["fraude", "id"]).columns.tolist()
    features_info = {
        "selected_features": selected_features,
        "categorical_features": [feat for feat in CAT_FEATURES if feat in selected_features],
        "numerical_features": [feat for feat in NUM_FEATURES if feat in selected_features],
    }
    print(f"Selected features: {selected_features}")
    print("Writing the features information file...")
    json.dump(features_info, open(os.path.join(output_path, "features/features_info.json"), "w"))
