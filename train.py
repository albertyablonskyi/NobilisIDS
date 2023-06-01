import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib
import datetime

# Dataset and model paths
csv_file_path = 'datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
model_file_path = 'models/ddos.pkl'

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
X = df.drop('Label', axis=1)
Y = df['Label']

# Identify and replace problematic values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute missing values
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train the classifier
print("Training the classifier...")
classifier = RandomForestClassifier(n_estimators=100)
progress_bar = tqdm(total=100, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
for _ in range(classifier.n_estimators):
    classifier.fit(X, Y)
    progress_bar.update(1)
    print("Training progress: {}%".format(int(progress_bar.n / progress_bar.total * 100)), end="\r")
progress_bar.close()
print("Training completed! Finished at: ", datetime.datetime.now().time())

# Make predictions
print("Making predictions...")
y_pred = classifier.predict(X)

# Get unique labels from the dataset
unique_labels = np.unique(df['Label'])

# Generate classification report
report = classification_report(df['Label'], y_pred, labels=unique_labels)

print("Classification Report:")
print(report)

print("Saving the model to:", model_file_path)
joblib.dump(classifier, model_file_path)

print("Model saved! Bye!")
