
import pandas as pd

data = pd.read_csv("data.csv")


if 'name' in data.columns:
    data = data.drop(columns=['name'])

print(data.columns)


data['NDI'] = (data['MDVP:Jitter(%)'] + data['MDVP:Shimmer'] + data['PPE']) / 3

print(data[['NDI']].head())

def classify_ndi(value):
    if value < 0.12:
        return "Normal"
    elif value < 0.16:
        return "Mild"
    else:
        return "Severe"

data['Condition'] = data['NDI'].apply(classify_ndi)

print("\nNDI + Condition:\n")
print(data[['NDI', 'Condition']].head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

X = data.drop(columns=['status', 'Condition'])
y = data['status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
import matplotlib.pyplot as plt

plt.hist(data['NDI'], bins=30)
plt.title("NDI Distribution")
plt.xlabel("NDI")
plt.ylabel("Frequency")
plt.show()

print("\n--- Enter Patient Values ---")

jitter = float(input("Enter Jitter (%): "))
shimmer = float(input("Enter Shimmer: "))
ppe = float(input("Enter PPE: "))

ndi_value = (jitter + shimmer + ppe) / 3

if ndi_value < 0.12:
    condition = "Normal"
elif ndi_value < 0.16:
    condition = "Mild"
else:
    condition = "Severe"

print("\nPredicted NDI:", ndi_value)
print("Predicted Condition:", condition)
