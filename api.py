import streamlit as st
import joblib
import pandas as pd
import re
#removing stop words like preposition, article,conjunction
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

nltk.download('stopwords')
st.title("This is a Machine learning Model")
df=pd.read_csv("/Users/sakibhussen/Downloads/IMDB Dataset.csv")

df=df.sample(10000)
df['sentiment'].replace({'positive':1,'negative':0},inplace=True)

def remove_special_charecters(text):
    patern=re.compile('<.*?>!')
    return re.sub(patern,'',text)


def convert_lower(text):
    return text.lower()
#converting special charecters
def remove_special(text):
    x=''
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x+ ' '
    return x

def remove_stopwords(text):
    x=[]
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y


y = []


def stem_words(text):
    for i in text:
        ps = PorterStemmer()
        y.append(ps.stem(i))
    z = y[:]
    y.clear()
    return z
#join back
def join_back(list_input):
    return " ".join(list_input)

cv.fit(df['review'])

X=cv.transform(df['review']).toarray()
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
print(accuracy_score(y_test,y_pred1))
print(accuracy_score(y_test,y_pred2))
print(accuracy_score(y_test,y_pred3))

st.dataframe(df.head())