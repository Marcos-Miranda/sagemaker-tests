from sklearn.base import BaseEstimator, TransformerMixin


class GBMFeatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_model, features_info):
        self.text_model = text_model
        self.features_info = features_info

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if "text_feat" not in X.columns:
            if "text_feat" in self.features_info["selected_features"]:
                X["text_feat"] = self.text_model.predict_proba(X["i"])[:, 1]
            X = X[self.features_info["selected_features"]]
            for col in self.features_info["categorical_features"]:
                X[col] = X[col].astype("category")
        return X
