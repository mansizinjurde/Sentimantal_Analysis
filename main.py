import pandas as pd
import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
import cleantext

st.header('Sentiment Analysis')


# Function to analyze sentiment and return sentiment label, polarity, and subjectivity
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)

    # Determine sentiment label based on polarity
    if polarity > 0.5:
        sentiment_label = 'Positive'
    elif polarity < -0.5:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    return sentiment_label, polarity, subjectivity


with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        sentiment_label, polarity, subjectivity = analyze_sentiment(text)
        st.write('Sentiment:', sentiment_label)
        st.write('Polarity:', polarity)
        st.write('Subjectivity:', subjectivity)

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                 stopwords=True, lowercase=True, numbers=True, punct=True))

with st.expander('Analyze CSV/TXT'):
    upl = st.file_uploader('Upload file', type=['csv', 'txt'])  # Allow CSV and TXT file uploads

    if upl:
        content = upl.getvalue().decode("utf-8")
        if upl.type == 'text/plain':  # Check if the uploaded file is a TXT file
            # Perform sentiment analysis for each line in the text file
            lines = content.split('\n')
            results = []
            for line in lines:
                sentiment_label, polarity, subjectivity = analyze_sentiment(line)
                results.append(
                    {'Text': line, 'Sentiment': sentiment_label, 'Polarity': polarity, 'Subjectivity': subjectivity})
            df_results = pd.DataFrame(results)
            st.write(df_results)

            # Option to download sentiment analysis results for text file
            st.download_button(
                label="Download Sentiment Analysis Results",
                data=df_results.to_csv().encode('utf-8'),
                file_name='sentiment_analysis_results.csv',
                mime='text/csv',
            )
        else:
            df = pd.read_excel(upl, engine='openpyxl')
            del df['Unnamed: 0']
            df['Sentiment'], df['Polarity'], df['Subjectivity'] = zip(*df['tweets'].apply(analyze_sentiment))
            st.write(df.head(10))

            fig, ax = plt.subplots()
            df['Sentiment'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            ax.set_title('Sentiment Analysis')
            st.pyplot(fig)


            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')


            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )
