import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dataset and model paths
csv_file_path = 'datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
model_file_path = 'models/onlyddos.pkl'

# Read the dataset
print("Reading the dataset...")
df = pd.read_csv(csv_file_path)

# Drop missing values
print("Dropping missing values...")
df.dropna(inplace=True)

# Remove whitespaces from column names
print("Removing whitespaces from column names...")
df.columns = df.columns.str.strip()

# Separate features and labels
df = df.drop('Label', axis=1)

# Check for infinity values and replace with np.nan
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Extract features
X = df.values

# Impute missing values
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Load the pre-trained model
print("Loading the trained model...")
model = joblib.load(model_file_path)

# Predict labels
print("Predicting labels...")
y_pred = model.predict(X)
probabilities = model.predict_proba(X)

# Count the number of predicted attacks
num_attacks = np.sum(y_pred != "BENIGN")
print("Number of attacks predicted:", num_attacks)

unique_labels = np.unique(y_pred)
print("Unique predicted labels:", unique_labels)

print("Probability estimates:")
for i, label in enumerate(unique_labels):
    label_probabilities = probabilities[:, i]
    print(f"{label}: {label_probabilities}")

print("Analysis completed! Bye!")