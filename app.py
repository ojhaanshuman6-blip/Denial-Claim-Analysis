import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.title("📌 Claim Priority Prediction System")
st.write("Enter claim details and get automated priority prediction.")

# Load model
with open("priority_model.pkl", "rb") as f:
    model = pickle.load(f)

# Encoders (same as training)
risk_encoder = LabelEncoder()
risk_encoder.fit(['low', 'medium', 'high'])

priority_encoder = LabelEncoder()
priority_encoder.fit(['Low', 'Medium', 'High'])

# User Inputs
amount = st.number_input("Claim Amount", min_value=0, value=10000)
days = st.number_input("Days Outstanding", min_value=0, value=10)
risk = st.selectbox("Customer Risk", ['low', 'medium', 'high'])
docs = st.selectbox("Documents Missing?", ["No (0)", "Yes (1)"])

docs_val = 0 if docs == "No (0)" else 1
risk_val = risk_encoder.transform([risk])[0]

# Predict Button
if st.button("Predict Priority"):
    new_df = pd.DataFrame([[amount, days, risk_val, docs_val]], 
                          columns=['claim_amount', 'days_outstanding', 'customer_risk', 'documents_missing'])

    pred = model.predict(new_df)[0]
    result = priority_encoder.inverse_transform([pred])[0]

    st.success(f"✅ Predicted Priority: **{result}**") 
    
