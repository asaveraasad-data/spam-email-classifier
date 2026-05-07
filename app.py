import pickle
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import re

# 1. Load Models
# Ensure these files exist in your directory
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 2. Define Functions
def get_top_spam_words(text, vectorizer, model, top_n=10):
    # Transform the text into the same TF-IDF space
    X = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    # Get indices of words actually present in this specific email
    nonzero_indices = X.nonzero()[1]
    
    if len(nonzero_indices) == 0:
        return pd.DataFrame()

    # Get log probabilities for both Ham (0) and Spam (1)
    log_prob_ham = model.feature_log_prob_[0]
    log_prob_spam = model.feature_log_prob_[1]
    
    # Calculate the "Spamminess" of each word present in the email
    # A higher (less negative) difference means the word is more likely in spam
    spam_influence = log_prob_spam - log_prob_ham
    
    # Filter to only the words present in the current email
    word_scores = []
    for i in nonzero_indices:
        word_scores.append({
            "Word": feature_names[i],
            "Influence Score": spam_influence[i]
        })
    # Convert to DataFrame and sort
    df_result = pd.DataFrame(word_scores)
    # Sort by influence (highest first) and take top_n
    df_result = df_result.sort_values(by="Influence Score", ascending=False).head(top_n)
    # Optional: Filter out words that actually lean towards "Ham" (Score < 0)
    df_result = df_result[df_result["Influence Score"] > 0]
    return df_result

def highlight_spam_text(original_text, spam_words):
    # Sort spam words by length (descending) so "credit card" is replaced before "credit"
    sorted_words = sorted(spam_words, key=len, reverse=True)
    
    highlighted_text = original_text
    
    # CSS for the highlight effect
    style = "background-color: #ffcccc; border-bottom: 2px solid #ff4b4b; padding: 0 2px; border-radius: 3px; color: #990000;"

    for word in sorted_words:
        # Use regex to replace whole words only (case insensitive)
        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span style="{style}">\\1</span>', highlighted_text)
        
    return highlighted_text.replace("\n", "<br>")

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    return text

# 3. Page Config & CSS
st.set_page_config(page_title="Spam Email Checker", layout="wide")

st.markdown("""
<style>
    /* Remove top space */
    section.main > div:first-child { margin-top: -25px !important; }
    .block-container { padding: 0 !important; margin: 0 !important; }
    header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
    .stApp { padding-top: 0 !important; margin-top: 0 !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stVerticalBlock"] { padding-top: 0 !important; }
    .element-container { padding-top: 0 !important; }
    iframe { margin: 0 !important; padding: 0 !important; }
    
    /* Center the button */
    .stButton button {
        width: 100%;
        background-color: #0078ff;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #005fcc;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 4. Navbar
# 4. Navbar
components.html("""
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        .navbar { width: 100%; background-color: #ffffff; display: flex; justify-content: center; align-items: center; border-bottom: 1px solid #e6e6e6; font-family: 'Segoe UI', sans-serif; height: 60px; }
        .navbar-container { width: 100%; max-width: 1200px; padding: 0 20px; display: flex; justify-content: space-between; align-items: center; }
        .navbar-left { display: flex; align-items: center; gap: 10px; flex: 1; }
        .navbar-left img { height: 24px; }
        .navbar-left span { font-size: 21px; font-weight: 700; color: #1a1a1a; white-space: nowrap; }
        .navbar-center { display: flex; gap: 25px; justify-content: center; flex: 2; }
        .navbar-center a { text-decoration: none; color: #444; font-weight: 500; font-size: 14px; transition: color 0.2s; }
        .navbar-center a:hover { color: #0078ff; }
        .navbar-right { display: flex; align-items: center; gap: 20px; justify-content: flex-end; flex: 1; }
        .navbar-right a { text-decoration: none; color: #333; font-weight: 600; font-size: 14px; }
        .get-started { background-color: #0078ff; color: white !important; padding: 8px 18px; border-radius: 6px; font-weight: 600; text-decoration: none; }
        @media (max-width: 850px) { .navbar-center { display: none; } }
    </style>
    <div class="navbar">
        <div class="navbar-container">
            <div class="navbar-left">
                <img src="https://cdn-icons-png.flaticon.com/512/561/561127.png" alt="Logo">
                <span>Spam Email Checker</span>
            </div>
            <div class="navbar-center">
                <a href="#">Services</a><a href="#">Solutions</a><a href="#">Pricing</a>
            </div>
            <div class="navbar-right">
                <a href="#">LOG IN</a><a href="#" class="get-started">SIGN IN</a>
            </div>
        </div>
    </div>
""", height=50)
# 5. Hero Section
components.html("""
    <style>
        .hero-section { width: 100%; padding: 70px 20px; text-align: center; background: linear-gradient(to bottom, #dff0ff, #ffffff); font-family: 'Segoe UI', sans-serif; }
        .hero-title { font-size: 48px; font-weight: 800; color: #111; margin-bottom: 20px; }
        .hero-subtitle { font-size: 24px; font-weight: 600; color: #0078ff; margin-bottom: 20px; }
        .hero-description { max-width: 800px; margin: auto; font-size: 16px; color: #444; line-height: 1.6; }
    </style>
    <div class="hero-section">
        <div class="hero-title">Email Spam Checker</div>
        <div class="hero-subtitle">Check the quality of an E-mail message</div>
        <div class="hero-description">
            Our free Email Spam Checker tool helps prevent spam, save resources, 
            enhance security, and improve deliverability by quickly verifying that 
            emails are real, valid, and free of typos or disposable domains.
        </div>
    </div>
""", height=300)

# -------------------------------------------------------
# MAIN INPUT SECTION (CENTERED & SINGLE BOX)
# -------------------------------------------------------

# Create 3 columns. The middle one (col2) is wider (ratio 4) to hold the content.
col1, col2, col3 = st.columns([1, 4, 1])

with col2:

    # Only ONE input box (Text Area)
    message_input = st.text_area("Paste your E-mail message here:", height=200)
    
    # Button inside the centered column
    predict_clicked = st.button("Predict")

    if predict_clicked:
        if not message_input:
            st.warning("Please paste an E-mail message to check.")
        else:
            # 1. Prepare text
            email_text = clean_text(message_input)

            # 2. Predict spam probability for a new email
            email_vector = vectorizer.transform([email_text])
            prediction = model.predict(email_vector)[0]
            prob = model.predict_proba(email_vector)[0][1]

            # 3. Display Results (Your original code)
            if prediction == 1:
                st.error("⚠️ This is a Spam Email")
            else:
                st.success("✅ This is a Ham Email")

            st.markdown(f"### 📊 Spam Probability: **{prob*100:.2f}%**")

            # 4. Explanation (Your original code)
            st.markdown("#### 🔍 Words Influencing This Prediction")
            explanation_df = get_top_spam_words(email_text, vectorizer, model, top_n=10)

            # --- START OF HIGHLIGHTING ADDITION ---
            if not explanation_df.empty:
                # Extract word list from your existing explanation_df
                spam_list = explanation_df['Word'].tolist()
                highlighted_html = highlight_spam_text(message_input, spam_list)
                
                st.markdown(f"""
                    <div style="background-color: #f9f9f9; border: 1px solid #ddd; padding: 20px; border-radius: 10px; line-height: 1.8; font-size: 16px; color: #333;">
                        {highlighted_html}
                    </div>
                """, unsafe_allow_html=True)
            # --- END OF HIGHLIGHTING ADDITION ---

            if explanation_df.empty:
                st.info("No strong Spam-indicating words found.")
            else:
                st.dataframe(explanation_df, use_container_width=True)