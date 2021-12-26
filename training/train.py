from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import json
import os
import argparse
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.base import clone
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


def compute_metrics(y_true, y_pred, prefix=""):

    return {
        f"recall_0_{prefix}": recall_score(y_true, y_pred, pos_label=0),
        f"recall_1_{prefix}": recall_score(y_true, y_pred, pos_label=1),
        f"precision_0_{prefix}": precision_score(y_true, y_pred, pos_label=0),
        f"precision_1_{prefix}": precision_score(y_true, y_pred, pos_label=1),
        f"f1_0_{prefix}": f1_score(y_true, y_pred, pos_label=0),
        f"f1_1_{prefix}": f1_score(y_true, y_pred, pos_label=1),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--dev_dir", type=str, default=os.environ["SM_CHANNEL_DEV"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--subsample_freq", type=int, default=0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--oversampling_rate", type=float, default=1.0)

    args = parser.parse_args()
    print(f"Received arguments: {args}")

    feat_infos = json.load(open(os.path.join(os.environ["SM_CHANNEL_FEATS"], "features_info.json"), "r"))
    text_model = joblib.load(os.path.join(os.environ["SM_CHANNEL_TXTML"], "text_model.joblib"))

    hypers = {
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "min_child_samples": args.min_child_samples,
        "colsample_bytree": args.colsample_bytree,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
    }

    pipe = Pipeline(
        steps=[
            ("oversampler", RandomOverSampler(sampling_strategy=args.oversampling_rate)),
            ("classifier", LGBMClassifier(**hypers)),
        ]
    )

    print("Loading data...")
    train_df = pd.read_csv(os.path.join(args.train_dir, "train.csv"))
    dev_df = pd.read_csv(os.path.join(args.dev_dir, "dev.csv"))
    test_df = pd.read_csv(os.path.join(args.test_dir, "test.csv"))

    for col in feat_infos["categorical_features"]:
        train_df[col] = train_df[col].astype("category")
        dev_df[col] = dev_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    print("Training and predicting...")
    model = clone(pipe).fit(train_df[feat_infos["selected_features"]], train_df["fraude"])
    train_preds = model.predict(train_df[feat_infos["selected_features"]])
    dev_preds = model.predict(dev_df[feat_infos["selected_features"]])
    test_preds = model.predict(test_df[feat_infos["selected_features"]])

    print("Computing metrics...")
    train_metrics = compute_metrics(train_df["fraude"], train_preds, "train")
    print(f"Train metrics: {train_metrics}")
    dev_metrics = compute_metrics(dev_df["fraude"], dev_preds, "dev")
    print(f"Dev metrics: {dev_metrics}")
    test_metrics = compute_metrics(test_df["fraude"], test_preds, "test")
    print(f"Test metrics: {test_metrics}")
    print("Saving metrics...")
    json.dump(
        {"train": train_metrics, "dev": dev_metrics, "test": test_metrics},
        open(os.path.join(args.output_dir, "metrics.json"), "w"),
    )

    print("Training the model on the whole dataset...")
    df = pd.concat([train_df, dev_df, test_df])
    for col in feat_infos["categorical_features"]:
        df[col] = df[col].astype("category")
    final_model = clone(pipe).fit(df[feat_infos["selected_features"]], df["fraude"])

    print("Saving the model...")
    joblib.dump([feat_infos, text_model, final_model], os.path.join(args.model_dir, "model.joblib"), compress=1)


def model_fn(model_dir):
    """Load the fitted model."""

    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, request_content_type):

    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data
    raise ValueError(f"Content type {request_content_type} not suported")


def predict_fn(input_object, model):

    features = model[0]["selected_features"]

    data = {}

    try:
        dt = datetime.strptime(input_object["fecha"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        dt = None

    for col in features:
        if col == "dia_mes":
            data["dia_mes"] = dt.day if dt else np.nan
        if col == "dia_semana":
            data["dia_semana"] = dt.strftime("%A") if dt else np.nan
        elif col == "hora":
            data["hora"] = (dt.hour if dt.minute < 30 else dt.hour + 1) if dt else np.nan
        elif col == "text_feat":
            data["i"] = str(input_object["i"]) if "i" in input_object else ""
        elif col == "n":
            data["n"] = ("Y" if input_object["n"] == 1 else "N") if "n" in input_object else np.nan
        else:
            data[col] = input_object[col] if col in input_object else np.nan

    input_data = pd.DataFrame(data=data, index=[0])

    if "text_feat" in features:
        input_data["text_feat"] = model[1].predict_proba(input_data["i"])[0, 1]
    for col in model[0]["categorical_features"]:
        input_data[col] = input_data[col].astype("category")
    input_data = input_data[model[0]["selected_features"]]

    prob_pred = model[2].predict_proba(input_data)[0, 1]

    return {"fraude": "sim" if prob_pred > 0.5 else "nao", "probabilidade": prob_pred}
