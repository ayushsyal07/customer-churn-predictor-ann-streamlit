import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App title and instructions
st.title('🔮 Customer Churn Prediction App')
st.markdown("""
Welcome to the Customer Churn Predictor. Fill out the customer details below and we'll predict the likelihood of churn.

---  
""")

# Layout: Two columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92)
    balance = st.number_input('💰 Balance', min_value=0.0, step=100.0)
    credit_score = st.number_input('📊 Credit Score', min_value=0, max_value=900, value=0)

with col2:
    estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0, step=100.0)
    tenure = st.slider('📅 Tenure (Years)', 0, 10)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', ['Yes', 'No'])
    is_active_member = st.selectbox('📶 Is Active Member?', ['Yes', 'No'])

# Preprocessing inputs
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Prepare input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
if st.button('🧠 Predict Churn'):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.subheader('📈 Prediction Result:')
    st.write(f'**Churn Probability:** `{prediction_proba:.2f}`')

    if prediction_proba > 0.5:
        st.error('⚠️ The customer is **likely to churn.**')
    else:
        st.success('✅ The customer is **not likely to churn.**')

# Footer
st.markdown("""
---
*This tool is intended for demonstration purposes only. Predictions may not be fully accurate and should not be used for real-world financial decisions.*
""")
