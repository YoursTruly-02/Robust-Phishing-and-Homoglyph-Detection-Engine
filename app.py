import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import math
import re
from collections import Counter

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Robust Phishing and Homoglyph Detection Engine", page_icon="🛡️", layout="centered")

# --- 2. LOAD THE MODEL (Cached for speed) ---
@st.cache_resource
def load_model():
    model= XGBClassifier()
    model.load_model('phishing_model.json')  # Load the model from a JSON file
    return model

try:
    model = load_model()
    # For now, we'll mock the loading to ensure the UI builds perfectly
    st.session_state['model_loaded'] = True 
except FileNotFoundError:
    st.error("Model file not found! Please ensure 'phishing_model.json' is in the same folder as this script.")
    st.stop()

# --- 3. FEATURE EXTRACTION ENGINE ---
# This recreates the exact mathematical features the XGBoost model was trained on
def calculate_entropy(url):
    if not url: return 0
    p, lns = Counter(url), float(len(url))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())

def extract_features(url):
        
    features = {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'has_https': 1 if url.startswith("https://") else 0,
        'entropy': calculate_entropy(url),
        'digits_count': sum(c.isdigit() for c in url),
        'special_char_count': len(re.findall(r'[^a-zA-Z0-9]', url)),
    }
    ordered_columns= ['url_length', 'num_dots', 'has_https', 'special_char_count', 'digits_count', 'entropy']
    df = pd.DataFrame([features], columns=ordered_columns)
    
    return df

# --- 4. THE USER INTERFACE ---
st.title("🛡️ Robust Phishing and Homoglyph Detection Engine")
st.markdown("Enter a suspicious URL below to analyze it for phishing indicators.")

# Search Bar
user_url = st.text_input("Target URL:", placeholder="e.g., https://secure-login.update-verification.com")

def check_safelist(url):
    # Add top trusted domains here
    trusted_domains = ['flipkart.com', 'google.com', 'amazon.in', 'youtube.com', 'github.com', 'microsoft.com']
    # Check if any trusted domain is inside the user's URL
    return any(domain in url.lower() for domain in trusted_domains)

# Analyze Button
if st.button("Analyze URL", type="primary"):
    if user_url:
        with st.spinner("Extracting features and analyzing..."):
            
            # 1. Extract the features
            live_features = extract_features(user_url)
            
            # 2. Make the Prediction 
            prediction = model.predict(live_features)[0]
            probability = model.predict_proba(live_features)[0][1]
                        
            st.markdown("---")
            
            # 3. Display the Results
            if prediction == 1:
                st.error(f"🚨 **PHISHING DETECTED** (Confidence: {probability * 100:.1f}%)")
                st.markdown("This URL exhibits strong indicators of a malicious attack.")
            else:
                st.success(f"✅ **SAFE DOMAIN** (Confidence: {(1 - probability) * 100:.1f}%)")
                st.markdown("No significant phishing indicators were detected.")
            
            # 4. Display the extracted math 
            st.markdown("### Internal Feature Analysis")
            st.dataframe(live_features, use_container_width=True)
            
    else:
        st.warning("Please enter a URL to analyze.")