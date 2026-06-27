# 🎯 Task 12: Final Project — CoreTech Customer Feedback Sentiment Analysis

**Intern:** Muntaha  
**Repository:** [muntaha700/Internship-repository](https://github.com/muntaha700/Internship-repository)  
**Task:** 12 (Final Project)  
**Date:** June 2026  

---

## 📋 Problem Statement

CoreTech Innovation collects customer feedback across multiple departments and project types. Manual review of hundreds of feedback entries is time-consuming and inconsistent. This project builds an **automated NLP-based sentiment classification system** that categorizes client feedback into four categories — *Positive, Negative, Neutral, and Mixed* — and provides actionable business intelligence to help CoreTech improve client satisfaction.

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| NLP | NLTK (stopwords, tokenization) |
| Feature Engineering | TF-IDF Vectorizer (bigrams, 5000 features) |
| ML Models | Multinomial Naive Bayes, Logistic Regression |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Environment | Jupyter Notebook / Google Colab |
| Model Export | Joblib (Pickle) |

---

## 🧹 Data Cleaning Steps

1. **Loaded** raw CSV with 800 client feedback records
2. **Lowercased** all text for uniformity
3. **Removed** special characters, punctuation, and digits using regex
4. **Removed** English stopwords using NLTK
5. **Filtered** tokens shorter than 3 characters
6. **Dropped** any rows with null values in `feedback` or `sentiment`
7. **Saved** cleaned dataset to `data/coretech_feedback_cleaned.csv`

---

## 🤖 Model Used

### Pipeline Architecture
```
Raw Text → clean_text() → TF-IDF Vectorizer → Classifier → Sentiment Label
```

### Models Trained
| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| Multinomial Naive Bayes | 100% | 1.00 | Fast, good baseline |
| **Logistic Regression** | **100%** | **1.00** | **Selected for deployment** |

**TF-IDF Settings:** `max_features=5000`, `ngram_range=(1,2)` (unigrams + bigrams)  
**Train/Test Split:** 80% / 20% with stratification

---

## 📊 Model Evaluation

- **Accuracy:** 100% on test set (160 samples)
- **Precision, Recall, F1:** 1.00 for all four classes
- **Confusion Matrix:** Zero misclassifications across all categories
- **Cross-validation:** Consistent performance across folds

### Sentiment Classes
| Label | Description |
|-------|-------------|
| Positive | Satisfied, exceeded expectations, excellent service |
| Negative | Disappointed, bugs, delays, poor communication |
| Neutral | Average, met requirements, nothing special |
| Mixed | Good in some areas, poor in others |

---

## 💡 Business Insights

1. **37.5% Positive Feedback** — IT and Finance departments are CoreTech's strongest performing areas and ideal targets for upselling.
2. **25% Negative Feedback** — Operations and Logistics need immediate attention; dedicated support reps recommended.
3. **Post-deployment bugs** are the #1 driver of Mixed sentiment — a 30-day post-launch support window should be standard.
4. **Communication** is the top complaint in Negative reviews — weekly client reports and a project portal are recommended.
5. **AI/ML and Cloud Migration** projects generate the most Mixed reviews — better expectation setting during onboarding is needed.

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/muntaha700/Internship-repository.git
cd Internship-repository/Task12_Final_Project
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib streamlit
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Task12_Sentiment_Analysis.ipynb
```
> Or open in Google Colab and upload the CSV files from the `data/` folder.

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
Task12_Final_Project/
├── data/
│   ├── coretech_feedback_raw.csv          # Original 800-row dataset
│   └── coretech_feedback_cleaned.csv      # Preprocessed dataset
├── models/
│   └── sentiment_model.pkl               # Trained Logistic Regression model
├── visualizations/
│   ├── 01_sentiment_distribution.png
│   ├── 02_rating_analysis.png
│   ├── 03_department_project.png
│   ├── 04_feedback_length.png
│   ├── 05_confusion_matrices.png
│   └── 06_model_comparison.png
├── Task12_Sentiment_Analysis.ipynb       # Main notebook (full pipeline)
├── app.py                                 # Streamlit web application
└── README.md                              # This file
```

---

## 🎥 Demo Video

> Record a short screen recording of the Streamlit app running:
> 1. Open the app → show Home page
> 2. Go to Single Prediction → type feedback → show result
> 3. Go to Analytics Dashboard → show charts
> 4. Go to Business Insights → show recommendations

Upload to Google Drive or YouTube and paste the link here.

**Demo Link:** `[Add your demo video link here]`

---

## ✅ Task 12 Checklist

- [x] Complete dataset (800 rows, 7 features)
- [x] Cleaned data saved as CSV
- [x] EDA with 6 visualizations
- [x] ML model (Naive Bayes + Logistic Regression)
- [x] Model evaluation (accuracy, F1, confusion matrix)
- [x] Business insights and recommendations for CoreTech
- [x] Streamlit app for dashboard and prediction
- [x] README with all required sections
- [ ] Demo video link (record and add)
- [ ] Submit GitHub repository link

---

*CoreTech Innovation Internship Program 2026 — Task 12 Final Project*
