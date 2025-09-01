
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError: 
        st.error(
            "Model file 'model.pkl' not found. Please ensure the model is trained and saved.")
        return None


def preprocess_input(transaction_type, amount, old_balance_orig, new_balance_orig, old_balance_dest):
    type_mapping = {'CASH_OUT': 0, 'PAYMENT': 1,
                    'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4}
    type_encoded = type_mapping.get(transaction_type, 0)

    transaction_type_weights = {0: 2.0, 1: 1.0, 2: 1.0, 3: 2.0, 4: 1.0}

    if amount <= 100:
        bin_weight = 1.0
    elif amount <= 500:
        bin_weight = 1.0
    elif amount <= 1000:
        bin_weight = 1.0
    elif amount <= 5000:
        bin_weight = 1.0
    elif amount <= 10000:
        bin_weight = 1.0
    elif amount <= 50000:
        bin_weight = 1.0
    elif amount <= 100000:
        bin_weight = 1.0
    elif amount <= 1000000:
        bin_weight = 1.5
    else:
        bin_weight = 2.0

    combined_weight = transaction_type_weights.get(
        type_encoded, 1.0) * bin_weight

    scaler = RobustScaler()
    amount_scaled = scaler.fit_transform([[amount]])[0][0]

    new_balance_orig_log = np.log1p(new_balance_orig)
    old_balance_dest_log = np.log1p(old_balance_dest)
    balance_diff_orig = old_balance_orig - new_balance_orig
    features = np.array([[
        type_encoded,
        amount_scaled,
        new_balance_orig_log,
        old_balance_dest_log,
        balance_diff_orig
    ]])

    return features


def main():
    st.set_page_config(
        page_title="Fraud Detection App",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Online Payment Fraud Detection")
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    st.sidebar.header("Transaction Details")

    transaction_type = st.sidebar.selectbox(
        "Transaction Type",
        options=["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
    )

    amount = st.sidebar.number_input(
        "Transaction Amount",
        min_value=0.01,
        value=1000.0,
        step=0.01
    )

    old_balance_orig = st.sidebar.number_input(
        "Original Account Old Balance",
        min_value=0.0,
        value=0.0,
        step=0.01
    )

    new_balance_orig = st.sidebar.number_input(
        "Original Account New Balance",
        min_value=0.0,
        value=0.0,
        step=0.01
    )

    old_balance_dest = st.sidebar.number_input(
        "Destination Account Old Balance",
        min_value=0.0,
        value=0.0,
        step=0.01
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Transaction Summary")

        transaction_data = {
            "Transaction Type": transaction_type,
            "Amount": f"₹{amount:,.2f}",
            "Original Account Old Balance": f"₹{old_balance_orig:,.2f}",
            "Original Account New Balance": f"₹{new_balance_orig:,.2f}",
            "Destination Account Old Balance": f"₹{old_balance_dest:,.2f}",
            "Balance Difference": f"₹{old_balance_orig - new_balance_orig:,.2f}"
        }

        for key, value in transaction_data.items():
            st.write(f"**{key}:** {value}")

    with col2:
        st.subheader("Prediction")

        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            try:
                features = preprocess_input(
                    transaction_type, amount, old_balance_orig,
                    new_balance_orig, old_balance_dest
                )

                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]

                if prediction == 1:
                    st.error("🚨 **FRAUDULENT TRANSACTION DETECTED**")
                    st.write(
                        f"**Fraud Probability:** {prediction_proba[1]:.2%}")
                else:
                    st.success("✅ **LEGITIMATE TRANSACTION**")
                    st.write(
                        f"**Fraud Probability:** {prediction_proba[1]:.2%}")

                st.subheader("Risk Assessment")
                fraud_prob = prediction_proba[1]

                if fraud_prob < 0.3:
                    color = "green"
                    risk_level = "Low Risk"
                elif fraud_prob < 0.7:
                    color = "orange"
                    risk_level = "Medium Risk"
                else:
                    color = "red"
                    risk_level = "High Risk"

                st.markdown(f"""
                <div style='background-color: {color}; padding: 10px; border-radius: 5px; color: white; text-align: center;'>
                    <h4>{risk_level}</h4>
                    <p>Fraud Probability: {fraud_prob:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    st.markdown("---")
    st.subheader("ℹ️ About This Model")
    st.info("""
    This fraud detection model was trained on online payment transaction data using machine learning techniques.
    The model analyzes various features of a transaction to determine the likelihood of fraud.
    
    **Key Features Analyzed:**
    - Transaction type and amount
    - Account balance changes
    - Transaction patterns
    - Historical data patterns
    
    **Note:** This is a demonstration model. For production use, additional security measures and model validation should be implemented.
    """)


if __name__ == "__main__":
    main()
