# Sentiment-Analysis-of-Social-Media
A Python-based web application that performs real-time sentiment analysis and topic modeling on social media data (Twitter/X) and product reviews. It tracks brand perception over time using VADER sentiment analysis, LDA topic modeling, interactive visualizations, and a user-friendly Streamlit dashboard.

Key Capabilities:

Analyzes text sentiment (Positive/Neutral/Negative)
Identifies main discussion topics
Visualizes sentiment trends over time
Generates word clouds and performance metrics

Ideal for brand monitoring, marketing insights, and data science demonstrations.


## ✨ Features

- **Real-time Sentiment Analysis** using VADER + TextBlob
- **Topic Modeling** with LDA (Latent Dirichlet Allocation)
- **Interactive Time-Series Visualization**
- **Word Cloud** generation
- **Streamlit Web Dashboard** (fully interactive)
- Supports CSV upload or live data
- Brand perception tracking over time

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Analysis**: VADER Sentiment, TextBlob, scikit-learn (LDA)
- **Visualization**: Plotly, Matplotlib, WordCloud
- **Data**: Pandas

## 📁 Project Structure

sentiment-app/
├── sentiment_dashboard.py      # Main Streamlit App
├── sentiment_analysis.py       # Core analysis functions
├── requirements.txt
├── README.md
├── data/                       # (optional) sample CSVs
└── results/                    # Output files
