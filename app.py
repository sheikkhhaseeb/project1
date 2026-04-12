import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# 1. Page configuration
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='📊',
    layout='wide'
)

# --- VISUALIZATION FUNCTIONS ---
# Inhein hamesha top par ya main block se bahar rakhein
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

    fig = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h',
                 title='Top 10 Factors Influencing Churn',
                 color='Importance',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# 2. Title
st.title('Customer Churn Prediction System')

# 3. Load model
@st.cache_resource
def load_model():
    # Ensure this filename matches your GitHub file
    with open('best_churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    st.sidebar.success('Model loaded successfully!')
except Exception as e:
    st.error(f"Error loading model: {e}")

# 4. Layout columns for user input
col1, col2 = st.columns(2)

with col1:
    st.subheader('Customer Demographics')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])

with col2:
    st.subheader('Account Information')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.number_input(
        'Monthly Charges ($)',
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

# 5. Prediction Logic
if st.button('Predict Churn', type='primary'):
    # Input data dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges
    }

    input_df = pd.DataFrame([input_data])
    
    # Simple encoding to match typical churn models
    # Note: Adjust these columns based on your specific model's training
    input_encoded = pd.get_dummies(input_df)

    # Align columns with model
    try:
        model_columns = model.feature_names_in_
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Make Prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0]
        churn_prob = probability[1] * 100

        # Display Results
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.error('### HIGH RISK: Customer likely to churn')
                st.metric('Churn Probability', f'{churn_prob:.1f}%')
            else:
                st.success('### LOW RISK: Customer likely to stay')
                st.metric('Retention Probability', f'{100 - churn_prob:.1f}%')
        
        with res_col2:
            # Visualization: Gauge Chart for Probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob,
                title = {'text': "Churn Risk Score"},
                gauge = {'axis': {'range': [None, 100]},
                         'bar': {'color': "red" if prediction == 1 else "green"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 6. Feature Importance (Graphics requested by Sir)
        st.divider()
        st.subheader("Why this prediction?")
        plot_feature_importance(model, model_columns)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if your model features match the input columns.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed for Churn Dataset Project")
