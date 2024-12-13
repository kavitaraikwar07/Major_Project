import streamlit as st
import pickle
import string
import nltk
from nltk.stem import PorterStemmer
import re

ps = PorterStemmer()
from nltk.corpus import stopwords


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    # clean = re.compile('<.*?>')
    # text = re.sub(clean,'',text)k
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

   

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit Frontend Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
        position: relative;
        height: 100vh;
    }
   
    .stApp {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4, #ffdde1);
        background-size: 400% 400%;
        #animation: gradientBG 10s ease infinite; 
        }
         @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

      .result {
            background-color: #f0f2f6; /* Light background color */
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-top: 80px;
        }
 
    .title {
        font-size: 2.7em;
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True
)

countVect = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
st.title("sentiment analysis")
input_sms = str(st.text_input("enter the message 😊😡🤓"))

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = countVect.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    # Display results with animations and styling
    st.markdown(
        f"""
        <div class="result">
            <h2 class="title">Sentiment Analysis Result:</h2>
            <h3 style="color: {'#4CAF50' if result == 1 else '#F44336'}">
                {"Positive 😊😃" if result == 1 else "Negative 😞😡"}
            </h3>
        </div>
        """, unsafe_allow_html=True
    )

    # if result == 1:
    #     st.header("positive")
    # else:
    #     st.header("negative")
