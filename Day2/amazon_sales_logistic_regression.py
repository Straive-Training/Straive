# Logistic Regression on Amazon Sales Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
df = pd.read_csv("C:/Users/Administrator/Downloads/Amazon Sale Report.csv")

# Step 2: Clean column names (remove spaces, etc.)
df.columns = df.columns.str.strip()

# Step 3: Drop rows with missing Amount
df = df.dropna(subset=["Amount"]).copy()

# Step 4: Create binary target variable: High Sale = 1, Low Sale = 0
threshold = df["Amount"].median()
df["HighSale"] = (df["Amount"] >= threshold).astype(int)

# Step 5: Select features
features = ["Qty", "Category", "Fulfilment", "Sales Channel", "B2B"]
X = df[features]
y = df["HighSale"]

# Step 6: Encode categorical features
for col in X.select_dtypes(include=["object", "bool"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12, stratify=y
)

# Step 8: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Step 10: Predictions
y_pred = model.predict(X_test_scaled)

# Step 11: Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 12: Example Prediction
sample = pd.DataFrame({
    "Qty": [2],
    "Category": ["Set"],
    "Fulfilment": ["Amazon"],
    "Sales Channel": ["Amazon.in"],
    "B2B": [False]
})

# Encode sample in same way
for col in sample.select_dtypes(include=["object", "bool"]).columns:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    sample[col] = le.transform(sample[col].astype(str))

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("\nPrediction for sample:", "High Sale" if prediction[0]==1 else "Low Sale")