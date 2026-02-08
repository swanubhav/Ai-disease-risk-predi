import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

files = ["heart.csv", "diabetes.csv", "stroke.csv"]

X_all, y_all = [], []

for file in files:
    df = pd.read_csv("data/" + file)
    X_all.append(df.drop("disease", axis=1))
    y_all.append(df["disease"])

X = pd.concat(X_all)
y = pd.concat(y_all)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=30, batch_size=16)

model.save("models/dl_multidisease.h5")

print("✅ DL model trained")

