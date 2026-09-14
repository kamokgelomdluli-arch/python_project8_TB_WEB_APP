import streamlit as st
import joblib
import pandas as pd

#trained model and scaler
model=joblib.load('TB_model_200k.pkl')
scaler=joblib.load('TB_model_scaler_200k.pkl')

features=scaler.feature_names_in_

st.title('TB SYMPTOM Diagnosis BOT')
st.write('_created by kamokgelo Ashley mdluli_')
st.write('**for education purposes only**')

#columns
col1, col2 = st.columns(2)

symptoms_features = {}

for i, symptomp in enumerate(features):
    # FIXED: Changed len(symptomp) to len(features)
    if i < len(features) // 2:
        with col1:
            symptoms_features[symptomp] = st.selectbox(symptomp, ['NO','YES'])
    else:
        with col2:
            symptoms_features[symptomp] = st.selectbox(symptomp, ['NO','YES'])
    

#encoding
if st.button("CHECK TB"):
    symptom_option = {}
    for option in features:
        symptom_option[option] = 1 if symptoms_features[option] == 'YES' else 0

    df = pd.DataFrame([symptom_option])
    
    scaled_inputs = scaler.transform(df)
    prediction = model.predict(scaled_inputs)[0]
    probability = model.predict_proba(scaled_inputs)[0][1]

    if prediction == 1:
         st.error('TB POSITIVE')
         st.write(probability)
    else:
         st.success('TB NEGATIVE')
         st.balloons()
         st.write(probability)
