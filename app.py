import streamlit as st
import pandas as pd
import joblib

# Load the trained AI model
try:
    model = joblib.load('misuse_detector_model.pkl')
except:
    st.error("Model file not found!")

@st.cache_data
def load_database():
    return pd.read_csv("processed_features.csv")

df_db = load_database()

st.set_page_config(page_title="Beyond Binary - Advanced Misuse Detector", layout="wide")
st.title("🛡️ Advanced Misuse Detector (Enterprise Grade AI)")
st.markdown("**Team ID:** AX-061 | **Track:** AI | **Team Name:** Beyond Binary")

# --- DATABASE INTEGRATION PANEL ---
st.sidebar.header("📁 Database Record Lookup")
record_idx = st.sidebar.number_input(f"Enter Customer Row Index (0 to {len(df_db)-1}):", min_value=0, max_value=len(df_db)-1, value=0)

customer_row = df_db.iloc[record_idx]
st.sidebar.success(f"Loaded Customer Record #{record_idx}")

# Map features to base data indicators
true_label = "Abusive (1)" if customer_row['abuse_label'] == 1 else "Genuine (0)"
st.sidebar.info(f"📋 Database Ground Truth Label: {true_label}")

# --- TWO TAB LAYOUT FOR INPUT FEATURES ---
col1, col2 = st.columns([1.2, 1])

with col1:
    st.header("👤 Customer Behavioral Profile")
    tab1, tab2 = st.tabs(["📊 Core Purchasing Metrics", "⚠️ Advanced Fraud Signals"])
    
    with tab1:
        age = st.slider("Customer Age", 18, 100, int(customer_row['age']))
        account_age_days = st.number_input("Account Age (Days)", min_value=1, value=int(customer_row['account_age_days']))
        avg_order_value_usd = st.number_input("Average Order Value ($)", min_value=0.0, value=float(customer_row['avg_order_value_usd']))
        refund_amount_requested_usd = st.number_input("Refund Amount Requested ($)", min_value=0.0, value=float(customer_row['refund_amount_requested_usd']))
        total_orders_lifetime = st.number_input("Total Orders Lifetime", min_value=1, value=int(customer_row['total_orders_lifetime']))
        total_returns_lifetime = st.number_input("Total Returns Lifetime", min_value=0, value=int(customer_row['total_returns_lifetime']))
        days_to_return = st.slider("Average Days to Return Item", 1, 30, int(customer_row['days_to_return']))
        
        # Calculate dynamic metrics
        return_rate_pct = (total_returns_lifetime / total_orders_lifetime) * 100 if total_orders_lifetime > 0 else 0.0
        st.info(f"Calculated Return Rate: {return_rate_pct:.1f}%")

    with tab2:
        st.markdown("### Anomaly & Fraud Intent Checklists")
        item_returned_opened = st.checkbox("Item Was Returned Opened/Used", value=bool(customer_row['item_returned_opened']))
        return_packaging_intact = st.checkbox("Original Return Packaging Intact", value=bool(customer_row['return_packaging_intact']))
        photo_evidence_provided = st.checkbox("Photo Evidence Provided by Customer", value=bool(customer_row['photo_evidence_provided']))
        address_change_before_delivery = st.checkbox("Address Change Triggered Before Delivery", value=bool(customer_row['address_change_before_delivery']))
        multiple_accounts_flag = st.checkbox("Linked to Multiple Guest Accounts", value=bool(customer_row['multiple_accounts_flag']))
        
        customer_support_contacts = st.slider("Customer Support Escalations", 0, 20, int(customer_row['customer_support_contacts']))
        previous_dispute_count = st.slider("Previous Claims/Disputes Count", 0, 10, int(customer_row['previous_dispute_count']))

with col2:
    st.header("🔍 Engine Risk Analysis")
    
    # Pack exactly according to advanced training list structure
    input_df = pd.DataFrame([{
        'age': age, 'account_age_days': account_age_days, 'avg_order_value_usd': avg_order_value_usd, 
        'refund_amount_requested_usd': refund_amount_requested_usd, 'total_orders_lifetime': total_orders_lifetime, 
        'total_returns_lifetime': total_returns_lifetime, 'return_rate_pct': return_rate_pct,
        'customer_support_contacts': customer_support_contacts, 'days_to_return': days_to_return, 
        'previous_dispute_count': previous_dispute_count, 'item_returned_opened': int(item_returned_opened), 
        'return_packaging_intact': int(return_packaging_intact), 'photo_evidence_provided': int(photo_evidence_provided), 
        'address_change_before_delivery': int(address_change_before_delivery), 'multiple_accounts_flag': int(multiple_accounts_flag)
    }])
    
    if st.button("Run Advanced Risk Assessment", type="primary"):
        probabilities = model.predict_proba(input_df)[0]
        risk_score = int(probabilities[1] * 100)
        st.session_state['last_score'] = risk_score
        
        st.subheader(f"Calculated System Risk Score: {risk_score}%")
        if risk_score < 30:
            st.success("🟢 LOW RISK: Legitimate Buyer Verified")
        elif risk_score < 65:
            st.warning("🟡 MEDIUM RISK: Suspicious Context Found (Audit Profile)")
        else:
            st.error("🔴 HIGH RISK: Confirmed Return Abuse Pattern")

# --- ADVANCED AI EXPLAINABILITY CHATBOT ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 Advanced Risk Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am connected to the advanced features dataset layer. Run an assessment and ask me to diagnose complex trends like 'wardrobing' or 'account structures'."}]

for msg in st.session_state.messages:
    st.sidebar.chat_message(msg["role"]).write(msg["content"])

if user_query := st.sidebar.chat_input("Ask about fraud flags..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.sidebar.chat_message("user").write(user_query)
    
    current_score = st.session_state.get('last_score', None)
    query = user_query.lower()
    
    if "explain" in query or "why" in query:
        if current_score is None:
            reply = "Please execute the assessment button first so I can trace individual feature weights."
        else:
            reply = f"System Diagnosis (Risk: {current_score}%): "
            flags = []
            if item_returned_opened and not return_packaging_intact:
                flags.append("Physical item open/damaged state points heavily to **Wardrobing** anomalies")
            if address_change_before_delivery:
                flags.append("Mid-transit routing adjustments highlight potential **fraudulent delivery claims**")
            if multiple_accounts_flag:
                flags.append("Device linkage across duplicate fake account nodes detected")
            if return_rate_pct > 50:
                flags.append(f"Highly abnormal volumetric return metrics ({return_rate_pct:.1f}%)")
            
            if flags:
                reply += "Critical flags triggered: " + " | ".join(flags) + "."
            else:
                reply += "No hazardous behavioral markers detected across advanced fields."
    elif "wardrobing" in query:
        reply = "Wardrobing is returning used merchandise. Our advanced engine flags this by checking if 'item_returned_opened' is true while 'return_packaging_intact' is false."
    else:
        reply = "Advanced features analysis ready. Ask me to 'explain the score' for a structured audit trail breakdown."
        
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.sidebar.chat_message("assistant").write(reply)
