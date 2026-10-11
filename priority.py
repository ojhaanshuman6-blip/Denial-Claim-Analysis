import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Step 1: Data
data = {
    'CLAIM_ID': np.arange(1,101),
    'CLAIM_AMOUNT': np.random.randint(1000,50000,100),
    'Days_outstanding': np.random.randint(1,90,100),
    'Customer_risk': np.random.choice(['low','medium','high'],100),
    'documents_missing': np.random.randint(0,2,100)
}

df = pd.DataFrame(data)
print("Step 1 Output:")
print(df.head())

# ✅ Convert ALL columns to lowercase
df.columns = df.columns.str.lower()

# Step 2: Assign priority
def assign_priority(row):
 amount = row['claim_amount']
 days = row['days_outstanding']

 if amount > 30000 and days > 40:
        return 'High'
 elif amount > 10000:
        return 'Medium'
 else:
        return 'Low'

df['priority'] = df.apply(assign_priority, axis=1)

print("\nStep 2 Output (Priority Created):")
print(df[['claim_amount','days_outstanding','priority']].head())

# Step 3: Encoding
le = LabelEncoder()
df['customer_risk'] = le.fit_transform(df['customer_risk'])
df['priority'] = le.fit_transform(df['priority'])

# Features
X = df[['claim_amount','days_outstanding','customer_risk','documents_missing']]
y = df['priority']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
# Step 4: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", acc)
# Separate Encoders for risk and priority
risk_encoder = LabelEncoder()
risk_encoder.fit(['low', 'medium', 'high'])

priority_encoder = LabelEncoder()
priority_encoder.fit(['Low', 'Medium', 'High'])

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\n=== Try Your Own Claim ===")

amount = int(input("Enter claim amount: "))
days = int(input("Enter days outstanding: "))
risk = input("Enter customer risk (low / medium / high): ")
docs = int(input("Documents missing? (0 = No, 1 = Yes): "))

# Encode risk using same label encoder
risk_encoded = risk_encoder.transform([risk])[0]
# Prepare data for model
new_data = [[amount, days, risk_encoded, docs]]
# Predict
pred = model.predict(new_data)[0]

# Decode priority back to original label
priority_label = le.inverse_transform([pred])[0]

print("\nPredicted Priority:", priority_label)
import pickle
with open("priority_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as priority_model.pkl")



     