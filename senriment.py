import streamlit as st
import joblib
import pandas as pd
import re
import nltk
import ssl
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

import setiment

ssl._create_default_https_context = ssl._create_unverified_context
#Downloadind NLTK Resourcess
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

# Title of the web app
st.title("Sentiment Analysis Web App")


def load_data():
    df = pd.read_csv("/Users/sakibhussen/Downloads/IMDB Dataset.csv")

    return df.sample(10000)


df = load_data()
df['sentiment'].replace({'positive': 1, 'negative': 0}, inplace=True)
st.dataframe(df.head(10))


def remove_special_charecters(text):
    patern = re.compile('<.*?>!')
    return re.sub(patern, '', text)


df['review'] = df['review'].apply(remove_special_charecters)

def convert_lower(text):
    return text.lower()

df['review'] = df['review'].apply(convert_lower)

def remove_special(text):
    x=''
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x+ ' '
    return x
df['review'] = df['review'].apply(remove_special)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


df['review'] = df['review'].apply(remove_stopwords)
def stem_words(text):
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in text.split()]
    return ' '.join(stemmed_words)

df['review'] = df['review'].apply(stem_words)




st.title("After process the text data ")
st.dataframe(df['review'].head(10))





cv = CountVectorizer()

cv.fit(df['review'])

X=cv.transform(df['review']).toarray()


y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
clf1= GaussianNB()
clf2= MultinomialNB()
clf3= BernoulliNB()
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
y_pred1=clf1.predict(X_test)
y_pred2=clf2.predict(X_test)
y_pred3=clf3.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)
accuracy3 = accuracy_score(y_test, y_pred3)

st.write("Gaussian Naive Bayes Accuracy:", accuracy1)
st.write("Multinomial Naive Bayes Accuracy:", accuracy2)
st.write("Bernoulli Naive Bayes Accuracy:", accuracy3)

# Create a bar chart to visualize the accuracy scores
chart_data = {"Model": ["Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes"],
              "Accuracy": [accuracy1, accuracy2, accuracy3]}
chart_df = pd.DataFrame(chart_data)

st.bar_chart(chart_df.set_index("Model"))
user_text_area=st.text_area("Enter your review here:")

if st.button("Predict"):
    if not user_text_area:
        st.warning("Please enter some text for sentiment analysis.")
    else:
        x=setiment.preprocess_review(user_text_area,cv)
        predicted_sentiment = clf1.predict(x)
        predicted_sentiment1 = clf2.predict(x)
        predicted_sentiment2 = clf3.predict(x)
        st.success(f"The sentiment of the text in Gaussian : {predicted_sentiment}")
        st.success(f"The sentiment of the text in Multinomial : {predicted_sentiment1}")
        st.success(f"The sentiment of the text in Bernoulli : {predicted_sentiment2}")

