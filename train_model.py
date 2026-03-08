import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------
# Load dataset
# -------------------------

X = np.load("X_data.npy")
y = np.load("y_labels.npy")

print("Dataset shape:", X.shape)

# -------------------------
# Scale features
# -------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------
# Train test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Train model
# -------------------------

model = SVC(
    kernel="rbf",
    probability=True
)

model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------

y_pred = model.predict(X_test)

# -------------------------
# Accuracy
# -------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -------------------------
# Precision Recall F1
# -------------------------

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# -------------------------
# Save model
# -------------------------

pickle.dump(model, open("gesture_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel and scaler saved")