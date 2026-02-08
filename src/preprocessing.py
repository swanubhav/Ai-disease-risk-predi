import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(filename):
    df = pd.read_csv("data/" + filename)

    X = df.drop("disease", axis=1)
    y = df["disease"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

