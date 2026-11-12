"""
encoding.py  --  Step 2 of the pipeline (paper Section 3.2, point 3)

One-hot encodes Sector, transport mode, reduction strategy and industry.
13 numeric + 17 dummy columns = 30 input features.
The encoder is fitted on the training data only and stored in the .pickle file,
so a user's query is encoded with exactly the same columns as during training.
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class FeatureEncoder:
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame):
        self.ohe.fit(X[CATEGORICAL_FEATURES])
        self.feature_names_ = NUMERIC_FEATURES + list(
            self.ohe.get_feature_names_out(CATEGORICAL_FEATURES)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dummies = self.ohe.transform(X[CATEGORICAL_FEATURES])
        out = pd.concat(
            [
                X[NUMERIC_FEATURES].reset_index(drop=True),
                pd.DataFrame(dummies, columns=self.ohe.get_feature_names_out(CATEGORICAL_FEATURES)),
            ],
            axis=1,
        )
        return out[self.feature_names_]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def categories(self) -> dict:
        """{column: [allowed values]} - used by the web form drop-downs."""
        return {c: list(v) for c, v in zip(CATEGORICAL_FEATURES, self.ohe.categories_)}
