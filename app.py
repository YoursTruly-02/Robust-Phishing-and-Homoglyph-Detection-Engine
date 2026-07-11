import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import math
import re
from collections import Counter
import pickle
import numpy as np



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


# Load TF-IDF Vectorizer
@st.cache_resource
def load_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

tfidf = load_vectorizer()

# --- 3. FEATURE EXTRACTION ENGINE ---


def extract_features(url):
        
    features = {}
    features["url_length"] = len(url)
    features["digit_count"] = sum(c.isdigit() for c in url)
    features["special_char_count"] = len(re.findall(r"[-@_.?=]", url))
    features["has_https"] = 1 if url.startswith("https") else 0
    features["has_ip"] = 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0
    
    return pd.DataFrame([features])

# --- 4. THE USER INTERFACE ---
st.title("🛡️ Robust Phishing and Homoglyph Detection Engine")
st.markdown("Enter a suspicious URL below to analyze it for phishing indicators.")

# Search Bar
user_url = st.text_input("Target URL:", placeholder="e.g., https://secure-login.update-verification.com")

# Analyze Button
if st.button("Analyze URL", type="primary"):
    if user_url:
        with st.spinner("Extracting features and analyzing..."):
            
            # 1. Get the 5 Math Features (Returns a DataFrame)
            math_features = extract_features(user_url)
            
            # 2. Get the 4,995 NLP Features from TF-IDF
            # .transform() expects a list, and .toarray() converts the sparse matrix to a flat grid
            nlp_features = tfidf.transform([user_url]).toarray()
            nlp_df = pd.DataFrame(nlp_features)
            
            # 3. Glue them together horizontally (axis=1) to get all 5,000 features
            live_features = pd.concat([nlp_df, math_features], axis=1)
            live_features.columns = live_features.columns.astype(str)
            # 4. Make the Prediction 
            prediction = model.predict(live_features)[0]
            probability = model.predict_proba(live_features)[0][1]
                        
            st.markdown("---")
            
            # 5. Display the Results
            if prediction == 1:
                st.error(f"🚨 **PHISHING DETECTED** (Confidence: {probability * 100:.1f}%)")
                st.markdown("This URL exhibits strong indicators of a malicious attack.")
            else:
                st.success(f"✅ **SAFE DOMAIN** (Confidence: {(1 - probability) * 100:.1f}%)")
                st.markdown("No significant phishing indicators were detected.")
            
            # 6. Display the extracted features
            st.markdown("### Internal Feature Analysis")
            st.dataframe(math_features, use_container_width=True)
        
    else:
        st.warning("Please enter a URL to analyze.")
