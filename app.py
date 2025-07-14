import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Function to clean text
def clean_text(tweet):
    tweet = re.sub(r'@[A-Za-z0-9_]+','', tweet)
    tweet = re.sub(r'#','', tweet)
    tweet = re.sub(r'https?:\/\/\S+','', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+','', tweet)
    tweet = re.sub(r'[^\w\s]','', tweet)
    return tweet.strip()

# Function to get sentiment
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        return 'Positive'
    elif polarity < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# App title
st.title("🌍 Real-World Tweet Sentiment Analyzer")
st.markdown("Analyze people's mood based on tweet content 📊")

# File upload
uploaded_file = st.file_uploader("📥 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Show preview
    st.subheader("🔍 Sample Tweets")
    st.write(df.head())

    # Process text
    df["CleanedTweet"] = df["Tweet"].apply(clean_text)
    df["Sentiment"] = df["CleanedTweet"].apply(get_sentiment)

    # Display sentiment count
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    # Pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    ax1.axis("equal")
    st.pyplot(fig1)

    # Word cloud
    st.subheader("☁️ Common Words WordCloud")
    all_words = ' '.join([text for text in df['CleanedTweet']])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)

    # Download processed file
    st.subheader("📥 Download Processed Data")
    st.download_button("Download CSV", df.to_csv(index=False), file_name='sentiment_result.csv', mime='text/csv')
    