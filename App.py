import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
from io import BytesIO

# -------------------------
# 🔹 Load the dataset
# -------------------------
st.set_page_config(page_title="Disease Risk Predictor", layout="wide")
st.title("🧠 Disease Risk Predictor – Chronic & Contagious")

@st.cache_data
def load_data():
    df = pd.read_csv("Diseases_Symptoms.csv")
    df["Symptoms"] = df["Symptoms"].fillna("").str.lower()
    df["Treatments"] = df["Treatments"].fillna("Not available")
    return df

df = load_data()

# -------------------------
# 🔹 Feature Engineering
# -------------------------
vectorizer = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(",")])
X_symptoms = vectorizer.fit_transform(df["Symptoms"])
X_symptom_df = pd.DataFrame(X_symptoms.toarray(), columns=vectorizer.get_feature_names_out())

# Targets
Y = df[["Chronic", "Contagious"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_symptom_df, Y, test_size=0.2, random_state=42, stratify=Y["Chronic"].astype(str) + Y["Contagious"].astype(str))

# -------------------------
# 🔹 Train the Model
# -------------------------
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
model.fit(X_train, y_train)

# -------------------------
# 🔹 User Input
# -------------------------
st.subheader("🔍 Enter Symptoms (comma-separated):")
user_input = st.text_input("E.g. fever, cough, shortness of breath")

if user_input:
    input_features = vectorizer.transform([user_input.lower()])
    input_df = pd.DataFrame(input_features.toarray(), columns=vectorizer.get_feature_names_out())

    for col in X_symptom_df.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X_symptom_df.columns]

    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)
    chronic_prob = probs[0][0][1]
    contagious_prob = probs[1][0][1]

    # -------------------------
    # 🔹 Results
    # -------------------------
    st.markdown("---")
    st.subheader("🧬 Prediction Results")

    st.write(f"**Chronic:** {'✅ Yes' if prediction[0] else '❌ No'} (Confidence: {chronic_prob:.2f})")
    st.write(f"**Contagious:** {'✅ Yes' if prediction[1] else '❌ No'} (Confidence: {contagious_prob:.2f})")

    # 📊 Chart
    st.subheader("📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(["Chronic", "Contagious"], [chronic_prob, contagious_prob], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # -------------------------
    # 🔹 Most Similar Diseases & Treatments
    # -------------------------
    st.markdown("---")
    st.subheader("💊 Top 3 Closest Disease Matches & Treatments")
    similarities = cosine_similarity(input_features, X_symptoms)
    top_indices = similarities[0].argsort()[-3:][::-1]

    for idx in top_indices:
        matched_disease = df.iloc[idx]
        st.markdown(f"**Disease:** {matched_disease['Name']}")
        st.markdown(f"**Symptoms:** {matched_disease['Symptoms']}")
        st.markdown(f"**Treatments:** {matched_disease['Treatments']}")
        st.markdown("---")

    # -------------------------
    # 🔹 Export to PDF
    # -------------------------
    st.subheader("📄 Export Report")

    def create_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Disease Risk Prediction Report", ln=1, align="C")
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=f"User Input Symptoms: {user_input}")
        pdf.ln(5)
        pdf.cell(0, 10, txt=f"Chronic: {'Yes' if prediction[0] else 'No'} (Confidence: {chronic_prob:.2f})", ln=1)
        pdf.cell(0, 10, txt=f"Contagious: {'Yes' if prediction[1] else 'No'} (Confidence: {contagious_prob:.2f})", ln=1)
        pdf.ln(5)

        for idx in top_indices:
            dis = df.iloc[idx]
            pdf.multi_cell(0, 10, txt=f"Disease: {dis['Name']}\nSymptoms: {dis['Symptoms']}\nTreatments: {dis['Treatments']}\n")
            pdf.ln(2)

        return pdf.output(dest="S").encode("latin1")

    if st.button("📥 Download PDF Report"):
        pdf_bytes = create_pdf()
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="report.pdf">📄 Click here to download your report</a>'
        st.markdown(href, unsafe_allow_html=True)

    # -------------------------
    # 🔹 Notes
    # -------------------------
    st.markdown("---")
    st.subheader("📌 Important Notes")
    st.info("This tool is for research and educational use only. Not a diagnostic system.")

# -------------------------
# 🔹 Footer
# -------------------------
st.markdown("""
---
Built by Yanet Niguse · Powered by Streamlit & Scikit-learn  
Contact for research collaboration | © 2025
""")
