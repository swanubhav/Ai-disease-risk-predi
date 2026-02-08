import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess

diseases = ["heart", "diabetes", "stroke"]

for disease in diseases:
    X, y, scaler = preprocess(f"{disease}.csv")

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    joblib.dump(model, f"models/{disease}_ml.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

print("✅ ML models trained")

