import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# SAMPLE DATA (simple demo dataset)
data = {
    "jitter": [0.005,0.01,0.02,0.03,0.04],
    "shimmer": [0.02,0.03,0.04,0.05,0.06],
    "ppe": [0.1,0.15,0.2,0.25,0.3],
    "result": [0,0,1,1,1]
}

df = pd.DataFrame(data)

X = df[["jitter","shimmer","ppe"]]
y = df["result"]

model = RandomForestClassifier()
model.fit(X,y)

# SAVE MODEL
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model trained and saved!")