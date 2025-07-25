
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('https://raw.githubusercontent.com/ThanviShetty/datasets/main/Crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pickle.dump(model, open("crop_model.pkl", "wb"))
sample = np.array([[90, 42, 43, 20.87, 82.00, 6.5, 202.93]])
prediction = model.predict(sample)
print(f"Recommended Crop: {prediction[0]}")
