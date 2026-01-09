import pandas as pd

def preprocess(path):
    df = pd.read_csv(path)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Handle TotalCharges if present as string
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(0, inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    return df
