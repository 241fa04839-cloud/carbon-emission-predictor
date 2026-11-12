"""
scaling.py  --  Step 3 of the pipeline (paper Section 3.2, point 5)

Standardisation  z = (x - mean) / std.  Mean and std come from the TRAINING set
only (no test-set leakage) and are stored in the .pickle file.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns_ = None

    def fit(self, X: pd.DataFrame):
        self.columns_ = list(X.columns)
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(X[self.columns_]), columns=self.columns_, index=X.index)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
