# Import Libraries
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load Dataset
df = pd.read_csv("dataset_heart.csv")

# Show First 5 Rows
print(df.head())

# Features and Target
X = df.drop("heart disease", axis=1)
y = df["heart disease"]

# Convert target values (1,2) to (0,1)
y = y.replace({1: 0, 2: 1})

# Encode categorical columns
X = pd.get_dummies(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print Results
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nFirst 10 Actual Values:")
print(y_test.head(10).values)

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])

from sklearn.metrics import confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nTN =", cm[0][0])
print("FP =", cm[0][1])
print("FN =", cm[1][0])
print("TP =", cm[1][1])

# Model Evaluation

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save Model

joblib.dump(model, "heart_model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nModel Saved Successfully!")

# Load Saved Model

loaded_model = joblib.load("heart_model.pkl")
loaded_columns = joblib.load("columns.pkl")

# Sample Patient Data
sample = X.iloc[[0]]

# Prediction
prediction = loaded_model.predict(sample)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Heart Disease: Yes")
else:
    print("Heart Disease: No")