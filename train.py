import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('hand_data.csv', header=None)
X = df.iloc[:, 1:] 
y = df.iloc[:, 0]  

# Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the brain
with open('model.p', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.p")