import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("data.csv")

# Remove name column if exists
if 'name' in data.columns:
    data = data.drop(columns=['name'])

# Create NDI (optional)
data['NDI'] = (data['MDVP:Jitter(%)'] + data['MDVP:Shimmer'] + data['PPE']) / 3

# Features & Target
X = data[['MDVP:Jitter(%)', 'MDVP:Shimmer', 'PPE']]
y = data['status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# ✅ Improved Model (correct place)
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1
)

# Train
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")