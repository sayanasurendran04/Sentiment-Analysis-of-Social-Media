import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
from sentiment_analysis import (
    preprocess_text, get_vader_sentiment, 
    categorize_sentiment, perform_topic_modeling
)

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("📊 Social Media Sentiment Analysis Dashboard")
st.markdown("**Track Brand Perception on X/Twitter & Reviews**")

# Sidebar
st.sidebar.header("Controls")
brand = st.sidebar.text_input("Brand/Query", "Tesla")
uploaded_file = st.sidebar.file_uploader("Upload CSV (columns: text, date)", type=["csv"])
date_range = st.sidebar.date_input("Date Range", 
    [datetime(2025, 1, 1).date(), datetime.now().date()])

# Load Data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        # Sample Data
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=500),
            'text': ["Love the innovation!"]*150 + ["Very disappointed."]*150 + ["Average product."]*200,
            'brand': ["Tesla"]*500
        })
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data(uploaded_file)

# Processing
df['clean_text'] = df['text'].apply(preprocess_text)
df['vader_score'] = df['clean_text'].apply(get_vader_sentiment)
df['sentiment'] = df['vader_score'].apply(categorize_sentiment)

# Date Filter
df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]

# Dashboard
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts", len(df))
col2.metric("Avg Sentiment", f"{df['vader_score'].mean():.2f}")
col3.metric("Positive", (df['sentiment'] == 'Positive').sum())
col4.metric("Negative", (df['sentiment'] == 'Negative').sum())

# Trend
st.subheader("Sentiment Trend Over Time")
weekly = df.resample('W', on='date')['vader_score'].mean().reset_index()
fig = px.line(weekly, x='date', y='vader_score', markers=True, title="Weekly Sentiment Trend")
st.plotly_chart(fig, use_container_width=True)

# Charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sentiment Distribution")
    st.plotly_chart(px.pie(df, names='sentiment'), use_container_width=True)

with col2:
    st.subheader("Word Cloud")
    if len(df) > 0:
        wc = WordCloud(background_color='black', width=700, height=400).generate(" ".join(df['clean_text']))
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis('off')
        st.pyplot(fig)

# Topics
if st.button("Show Topics"):
    with st.spinner("Analyzing topics..."):
        topics, _ = perform_topic_modeling(df['clean_text'])
        for topic in topics:
            st.write(topic)

if st.checkbox("Show Data"):
    st.dataframe(df)

st.caption("Built with Streamlit + VADER + LDA")