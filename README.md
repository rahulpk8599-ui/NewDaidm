# 🏙️ LifePass Dubai — Analytics Dashboard

**One Subscription · Every Service · One City**

A data analytics project for the GMBA programme at SP Jain School of Global Management, Dubai 2026.

## 📊 What This App Does

An interactive Streamlit dashboard with four analytics layers:

| Tab | Analytics Type | What it shows |
|-----|---------------|---------------|
| 📊 Descriptive | Who are the respondents? | Demographics, spend, service usage, segment profiles |
| 🔍 Diagnostic | Why do they subscribe or not? | Correlations, frustrations, pain drivers, trust barriers |
| 🤖 Predictive | Who will subscribe? | KNN, Decision Tree, Random Forest, Gradient Boosting with ROC curves |
| 🎯 Prescriptive | What should LifePass do? | Tier recommender, go-to-market priority, action plan |

## 🚀 Deploy on Streamlit Cloud

1. Fork or upload this repository to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app**
4. Select your GitHub repo
5. Set **Main file path** to `app.py`
6. Click **Deploy**

The dataset (`lifepass_survey_1000.csv`) is included in the repo and loaded automatically.

## 📁 File Structure

```
lifepass_app/
├── app.py                        # Main Streamlit application
├── generate_data.py              # Synthetic dataset generator
├── lifepass_survey_1000.csv      # Pre-generated dataset (1,000 respondents)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔬 Dataset Details

- **1,000 synthetic respondents** mimicking Dubai's diverse population
- **4 latent customer segments**: Busy Professional, Nuclear Family, Budget Expat, Premium Resident
- **53 variables** covering demographics, behaviour, service usage, pain points, and WTP
- **Realistic distributions**: right-skewed income and spend, bimodal pain scores
- **5% noise**, **3% outliers**, **2.5% missing values** for realistic ML training conditions
- **Target variable**: `would_subscribe` (68.1% yes / 31.9% no)

## 📈 Expected ML Performance

| Model | Test Accuracy | AUC-ROC |
|-------|-------------|---------|
| KNN | ~85% | ~0.90 |
| Decision Tree | ~87% | ~0.91 |
| Random Forest | ~91% | ~0.96 |
| Gradient Boosting | ~92% | ~0.97 |

## 🛠️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---
*SP Jain School of Global Management · GMBA · Dubai 2026*
