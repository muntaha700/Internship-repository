"""
CoreTech Innovation — Sentiment Analysis Dashboard
Task 12: Final Project | Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import os
from collections import Counter

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CoreTech Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin: 0.5rem 0 0; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #e94560;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .badge-Positive  { background: #d4edda; color: #155724; }
    .badge-Negative  { background: #f8d7da; color: #721c24; }
    .badge-Neutral   { background: #d1ecf1; color: #0c5460; }
    .badge-Mixed     { background: #fff3cd; color: #856404; }

    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c62a47);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        width: 100%;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────
COLORS = {
    'Positive': '#2ecc71',
    'Negative': '#e74c3c',
    'Neutral':  '#3498db',
    'Mixed':    '#f39c12'
}

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = set()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

@st.cache_resource
def load_model():
    model_path = 'models/sentiment_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    path = 'data/coretech_feedback_cleaned.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

model = load_model()
df    = load_data()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/280x70/0f3460/e94560?text=CoreTech+Innovation", width=280)
    st.markdown("---")
    st.markdown("### 🗂️ Navigation")
    page = st.radio("", ["🏠 Home & Overview", "🔍 Single Prediction",
                          "📂 Batch Prediction", "📊 Analytics Dashboard",
                          "💡 Business Insights"])
    st.markdown("---")
    st.markdown("**Model Info**")
    st.success("✅ Model Loaded" if model else "❌ Model Not Found")
    st.info("Logistic Regression\nTF-IDF (5000 features)")
    st.markdown("---")
    st.caption("Task 12 — Final Project\nIntern: Muntaha\nCoreTech Internship 2026")

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 CoreTech Sentiment Analyzer</h1>
    <p>Customer Feedback Intelligence Platform — Task 12 Final Project</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: Home & Overview
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home & Overview":
    st.subheader("📋 Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("800", "Total Feedback Records", "#2ecc71"),
        ("4", "Sentiment Categories", "#3498db"),
        ("2", "ML Models Trained", "#e94560"),
        ("100%", "Model Accuracy", "#f39c12"),
    ]
    for col, (val, label, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <h2 style="color:{color};margin:0">{val}</h2>
                <p style="color:#666;margin:0;font-size:0.85rem">{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🛠️ Technologies Used")
        techs = {
            "Python 3.10": "Core language",
            "NLTK": "Text preprocessing",
            "TF-IDF Vectorizer": "Feature extraction",
            "Naive Bayes": "Baseline classifier",
            "Logistic Regression": "Best classifier",
            "Matplotlib / Seaborn": "Visualizations",
            "Streamlit": "Web deployment",
            "Joblib": "Model serialization",
        }
        for tech, desc in techs.items():
            st.markdown(f"- **{tech}** — {desc}")

    with col_b:
        st.markdown("### 📁 Project Structure")
        st.code("""
Task12_Final_Project/
├── data/
│   ├── coretech_feedback_raw.csv
│   └── coretech_feedback_cleaned.csv
├── models/
│   └── sentiment_model.pkl
├── visualizations/
│   ├── 01_sentiment_distribution.png
│   ├── 02_rating_analysis.png
│   ├── 03_department_project.png
│   ├── 04_feedback_length.png
│   ├── 05_confusion_matrices.png
│   └── 06_model_comparison.png
├── Task12_Sentiment_Analysis.ipynb
├── app.py
└── README.md
""", language="")

    st.markdown("### 🔄 Pipeline Flow")
    steps = ["📥 Load Data", "🧹 Clean Text", "📊 EDA", "🤖 Train Models",
             "📈 Evaluate", "💾 Save Model", "🚀 Deploy"]
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        with col:
            st.markdown(f"""<div style="background:#0f3460;color:white;padding:0.6rem;
            border-radius:8px;text-align:center;font-size:0.8rem">{step}</div>""",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: Single Prediction
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Single Prediction":
    st.subheader("🔍 Single Feedback Prediction")
    st.markdown("Enter a customer feedback text below to get an instant sentiment prediction.")

    examples = {
        "Select an example...": "",
        "✅ Positive example": "The team delivered the project on time and exceeded all our expectations. Excellent work!",
        "❌ Negative example": "Very disappointed with the quality. Multiple bugs found and no proper support was given.",
        "➖ Neutral example":  "The project was completed as per the requirements. Nothing exceptional to mention.",
        "🔀 Mixed example":    "Good technical work but poor project management and communication from the team.",
    }

    selected = st.selectbox("Quick Examples", list(examples.keys()))
    default_text = examples[selected]

    user_input = st.text_area("✍️ Enter Feedback Text:", value=default_text, height=140,
                               placeholder="Type or paste customer feedback here...")

    if st.button("🎯 Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some feedback text.")
        elif model is None:
            st.error("Model not loaded. Please run the notebook first.")
        else:
            cleaned = clean_text(user_input)
            prediction = model.predict([cleaned])[0]
            proba = model.predict_proba([cleaned])[0]
            classes = model.classes_

            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### 🏷️ Prediction Result")
                badge_class = f"badge-{prediction}"
                emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐", "Mixed": "🤔"}.get(prediction, "")
                st.markdown(f"""
                <div style="text-align:center;padding:2rem;background:#f8f9fa;border-radius:12px">
                    <div style="font-size:3rem">{emoji}</div>
                    <span class="sentiment-badge {badge_class}">{prediction}</span>
                    <p style="color:#888;margin-top:1rem;font-size:0.9rem">Sentiment Label</p>
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown("#### 📊 Confidence Scores")
                fig, ax = plt.subplots(figsize=(7, 3))
                bar_colors = [COLORS.get(c, '#888') for c in classes]
                bars = ax.barh(classes, proba, color=bar_colors, edgecolor='white')
                ax.set_xlim(0, 1.1)
                ax.set_xlabel("Probability")
                ax.set_title("Model Confidence per Sentiment Class")
                for bar, val in zip(bars, proba):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:.2%}', va='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE: Batch Prediction
# ══════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.subheader("📂 Batch Prediction — Upload CSV")
    st.markdown("Upload a CSV file with a `feedback` column to predict sentiment for all rows at once.")

    uploaded_file = st.file_uploader("📎 Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: {batch_df.shape[0]} rows, {batch_df.shape[1]} columns")
            st.dataframe(batch_df.head(5))

            if 'feedback' not in batch_df.columns:
                st.error("❌ CSV must have a 'feedback' column.")
            elif model is None:
                st.error("Model not loaded.")
            else:
                if st.button("🚀 Run Batch Prediction"):
                    with st.spinner("Predicting sentiments..."):
                        batch_df['cleaned'] = batch_df['feedback'].apply(clean_text)
                        batch_df['predicted_sentiment'] = model.predict(batch_df['cleaned'])
                        batch_df = batch_df.drop(columns=['cleaned'])

                    st.success(f"✅ Done! Predicted {len(batch_df)} records.")
                    st.dataframe(batch_df)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        vc = batch_df['predicted_sentiment'].value_counts()
                        ax.pie(vc.values, labels=vc.index, autopct='%1.1f%%',
                               colors=[COLORS.get(s, '#888') for s in vc.index])
                        ax.set_title("Predicted Sentiment Distribution")
                        st.pyplot(fig); plt.close()

                    csv_out = batch_df.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Results CSV", csv_out,
                                       "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("💡 Tip: Your CSV just needs a 'feedback' column with text.")

# ══════════════════════════════════════════════════════════════
# PAGE: Analytics Dashboard
# ══════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.subheader("📊 Analytics Dashboard")

    if df is None:
        st.error("Dataset not found. Run the notebook first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        for col, (label, val) in zip([col1,col2,col3,col4], [
            ("Total Records", len(df)),
            ("Positive %", f"{(df['sentiment']=='Positive').mean():.1%}"),
            ("Negative %", f"{(df['sentiment']=='Negative').mean():.1%}"),
            ("Avg Rating", f"{df['rating'].mean():.2f} ⭐"),
        ]):
            col.metric(label, val)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            vc = df['sentiment'].value_counts()
            ax.bar(vc.index, vc.values, color=[COLORS[s] for s in vc.index], edgecolor='white')
            ax.set_xlabel("Sentiment"); ax.set_ylabel("Count")
            for i, (idx, val) in enumerate(vc.items()):
                ax.text(i, val+2, str(val), ha='center', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_b:
            st.markdown("#### Average Rating by Sentiment")
            fig, ax = plt.subplots(figsize=(6, 4))
            avg = df.groupby('sentiment')['rating'].mean().sort_values()
            bars = ax.barh(avg.index, avg.values, color=[COLORS[s] for s in avg.index])
            ax.set_xlabel("Avg Rating"); ax.set_xlim(0, 5.5)
            for bar, val in zip(bars, avg.values):
                ax.text(val+0.05, bar.get_y()+bar.get_height()/2, f'{val:.2f}', va='center', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Sentiment by Department")
        fig, ax = plt.subplots(figsize=(12, 4))
        df.groupby(['department','sentiment']).size().unstack(fill_value=0).plot(
            kind='bar', ax=ax, color=list(COLORS.values()), edgecolor='white')
        ax.set_xlabel("Department"); ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Sentiment by Project Type")
        fig, ax = plt.subplots(figsize=(12, 4))
        df.groupby(['project_type','sentiment']).size().unstack(fill_value=0).plot(
            kind='bar', ax=ax, color=list(COLORS.values()), edgecolor='white')
        ax.set_xlabel("Project Type"); ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=35)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE: Business Insights
# ══════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.subheader("💡 Business Insights & Recommendations")

    insights = [
        ("🏆 Positive Feedback Dominates", "37.5% of all client feedback is positive, particularly from IT and Finance departments. These clients are strong candidates for upselling advanced AI/ML solutions."),
        ("⚠️ Operations & Logistics Need Attention", "The highest rate of negative feedback comes from Operations and Logistics departments. Dedicated support and faster response times are urgently needed."),
        ("🐛 Post-Deployment Bugs Drive Mixed Sentiment", "Mixed reviews frequently mention bugs and integration issues after go-live. A mandatory 30-day post-deployment support window should be introduced."),
        ("📞 Communication is a Top Pain Point", "Many negative reviews cite poor communication and lack of status updates. Weekly progress reports and a client communication portal would help."),
        ("🤖 AI/ML Projects Need Better Expectation Setting", "AI/ML and Cloud Migration projects generate the most mixed feedback. Better client education and realistic milestone planning are recommended."),
    ]

    for title, text in insights:
        st.markdown(f"""
        <div class="insight-box">
            <strong>{title}</strong><br>
            <span style="color:#555">{text}</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### 📋 Model Performance Summary")
    perf_df = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression"],
        "Accuracy": ["100%", "100%"],
        "F1 Score (Weighted)": ["1.00", "1.00"],
        "Recommended": ["✅ Baseline", "✅ Production"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.success("🎓 **Project by Muntaha** | CoreTech Innovation Internship 2026 | Task 12 — Final Project")
