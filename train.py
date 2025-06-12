import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
df = pd.read_csv("student_data.csv")

# Create new columns
df["final score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["pass"] = df["final score"].apply(lambda x: 1 if x >= 40 else 0)

# Split dataset
X = df[["math score", "reading score", "writing score"]]
y_class = df["pass"]
y_reg = df["final score"]

X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2)

# Train models
clf = RandomForestClassifier()
clf.fit(X_train, y_class_train)

reg = LinearRegression()
reg.fit(X_train, y_reg_train)

# Save with pickle (NOT joblib)
with open("classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("regressor.pkl", "wb") as f:
    pickle.dump(reg, f)

print("✅ Models retrained and saved with pickle.")
