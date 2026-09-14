# python_project8_TB_WEB_APP
TB Symptom Diagnosis Bot - Streamlit App

## Project Description
A Streamlit web app that diagnoses Tuberculosis (TB) risk based on user-selected symptoms using a pre-trained machine learning model and scaler.

## Files
- `TB_model_200k.pkl` - Trained TB classification model
- `TB_model_scaler_200k.pkl` - Feature scaler

## Steps Performed

**1. Imports**
- Imported streamlit, joblib, pandas

**2. Load Files**
- Loaded TB model with `joblib.load()`
- Loaded scaler with `joblib.load()`

**3. Get Symptoms**
- Extracted feature names directly from `scaler.feature_names_in_`

**4. Page Header**
- Added title "TB SYMPTOM Diagnosis BOT"
- Added author credit "created by kamokgelo Ashley mdluli"
- Added disclaimer "for education purposes only"

**5. Two-Column Layout**
- Split screen with `st.columns(2)`
- Initialized empty dictionary `symptoms_features`

**6. Dropdown Loop**
- Looped through symptom features
- First half (`i < len(features) // 2`) → column 1
- Second half → column 2
- Each symptom gets 'NO'/'YES' dropdown using `st.selectbox()`
- Saved responses in dictionary

**7. Button Trigger**
- Execution runs only when "CHECK TB" button is clicked

**8. Encode Answers**
- Looped through selected features
- YES → 1
- NO → 0

**9. Prepare Data**
- Encapsulated encoded dictionary into a single-row DataFrame
- Scaled features using `scaler.transform()`

**10. Predict**
- Obtained binary diagnosis via `model.predict()`
- Calculated probability score via `model.predict_proba()`

**11. Show Result**
- Positive (1) → red error box + probability value
- Negative (0) → green success box + balloons animation + probability value

## Libraries Used
- streamlit, joblib, pandas
