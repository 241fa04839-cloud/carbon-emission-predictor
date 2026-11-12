"""
preprocessing.py  --  Step 1 of the pipeline (paper Section 3.2, points 1, 2 and 4)

* load the dataset
* remove identifier columns        (Company_ID, Date)
* remove leakage columns          (Carbon_Tax_USD, Energy_Cost_USD)
* split into 80 % train / 20 % test with random seed 42
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = "carbon_emission_dataset_with_Industry.csv"
TARGET = "Carbon_Emission_tCO2e_TARGET"
ID_COLUMNS = ["Company_ID", "Date"]
LEAKAGE_COLUMNS = ["Carbon_Tax_USD", "Energy_Cost_USD"]
SEED = 42

NUMERIC_FEATURES = [
    "Total_Energy_Consumption_kWh",
    "Renewable_Energy_Consumption_kWh",
    "NonRenewable_Energy_Consumption_kWh",
    "Production_Output_Units",
    "Supply_Chain_Transport_km",
    "Raw_Material_Usage_kg",
    "Process_Efficiency_Percent",
    "Employment_Count",
    "Public_Acceptance_Index",
    "Strategy_Implementation_Cost_USD",
    "Expected_Carbon_Reduction_Percent",
    "Expected_Renewable_Share_Percent",
    "Social_Impact_Score",
]
CATEGORICAL_FEATURES = [
    "Sector",
    "Supply_Chain_Transport_Mode",
    "Carbon_Reduction_Strategy",
    "Industry_Sectors",
]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame):
    """Drop identifier + leakage columns; return (X_raw, y)."""
    df = df.drop(columns=ID_COLUMNS + LEAKAGE_COLUMNS, errors="ignore")
    y = df[TARGET]
    X = df[RAW_FEATURES].copy()
    return X, y


def split_data(X, y, test_size: float = 0.20):
    """80/20 split, fixed seed 42 (paper Section 3.2 step 4)."""
    return train_test_split(X, y, test_size=test_size, random_state=SEED)


def prepare_query(query) -> pd.DataFrame:
    """Turn one dict (or a DataFrame) of user inputs into a clean raw-feature
    DataFrame in the column order the pipeline expects."""
    q = pd.DataFrame([query]) if isinstance(query, dict) else query.copy()
    missing = [c for c in RAW_FEATURES if c not in q.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    q = q[RAW_FEATURES].copy()
    for c in NUMERIC_FEATURES:
        q[c] = pd.to_numeric(q[c], errors="raise")
    return q
