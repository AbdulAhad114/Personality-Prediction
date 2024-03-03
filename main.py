# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
import streamlit as st

# Load the dataset (replace 'your_dataset.csv' with the actual file name)
data = pd.read_csv('./mbti_1.csv')

# Making the whole data in lower case
data['posts'] = data['posts'].str.lower()
#Define the text from which you want to replace the url with "".
def remove_URL(text):
    """Remove URLs from a text string"""
    return re.sub(r"http\S+", "", text)

# Removing URL's in the reviews
data['posts'] = data['posts'].apply(lambda x:remove_URL(x))

# removing Punctuations
punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)"""
data['posts'] = data['posts'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))


# Converting MBTI personality (or target or Y feature) into numerical form using Label Encoding
# encoding personality type
label_encoder  = preprocessing.LabelEncoder()
data['type of encoding'] = label_encoder.fit_transform(data['type'])
target = data['type of encoding']

count_vectorizer = CountVectorizer()
# Converting posts (or training or X feature) into numerical form by count vectorization
X = count_vectorizer.fit_transform(data["posts"])
# y = data['type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data["type of encoding"], test_size=0.2, stratify=target, random_state=42)

# Logistic Regression
# lr_model=LogisticRegression()
# lr_model.fit(X_train,y_train)
# lr_pred = lr_model.predict(X_test)
# lr_accuracy = accuracy_score(y_test,lr_pred)
# print("Logistics Regression Accuracy: ",lr_accuracy)

# Random Forest Classifier
# rfc_model = RandomForestClassifier()
# rfc_model.fit(X_train,y_train)
# rfc_pred = rfc_model.predict(X_test)
# rfc_accuracy = accuracy_score(y_test,rfc_pred)
# print("Random Forest Classifier: ",rfc_accuracy)

# Support Vector Machine (SVM) Classifier
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)
# svm_pred = svm_model.predict(X_test)
# svm_accuracy = accuracy_score(y_test, svm_pred)
# print("SVM Accuracy:", svm_accuracy)


# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
# xgb_pred = xgb_model.predict(X_test)
# xgb_accuracy = accuracy_score(y_test, xgb_pred)
# print("XGBoost Accuracy:", xgb_accuracy)

# Personality details dictionary
personality_details = {
    0: 'ISTJ: Introverted, Sensing, Thinking, Judging',
    1: 'ISFJ: Introverted, Sensing, Feeling, Judging',
    2: 'INFJ: Introverted, Intuitive, Feeling, Judging',
    3: 'INTJ: Introverted, Intuitive, Thinking, Judging',
    4: 'ISTP: Introverted, Sensing, Thinking, Perceiving',
    5: 'ISFP: Introverted, Sensing, Feeling, Perceiving',
    6: 'INFP: Introverted, Intuitive, Feeling, Perceiving',
    7: 'INTP: Introverted, Intuitive, Thinking, Perceiving',
    8: 'ESTP: Extroverted, Sensing, Thinking, Perceiving',
    9: 'ESFP: Extroverted, Sensing, Feeling, Perceiving',
    10: 'ENFP: Extroverted, Intuitive, Feeling, Perceiving',
    11: 'ENTP: Extroverted, Intuitive, Thinking, Perceiving',
    12: 'ESTJ: Extroverted, Sensing, Thinking, Judging',
    13: 'ESFJ: Extroverted, Sensing, Feeling, Judging',
    14: 'ENFJ: Extroverted, Intuitive, Feeling, Judging',
    15: 'ENTJ: Extroverted, Intuitive, Thinking, Judging'
}


# Streamlit app
st.title('Personality Prediction')

user_input = st.text_area('Enter your text here')

if st.button('Predict'):
    # Preprocess the user input
    user_input = user_input.lower()
    user_input = remove_URL(user_input)
    user_input = re.sub('[%s]' % re.escape(string.punctuation), '', user_input)
    
    # Vectorize the user input
    user_input_vectorized = count_vectorizer.transform([user_input])
    
    # Predict the personality type
    prediction = xgb_model.predict(user_input_vectorized)
    predicted_personality = label_encoder.inverse_transform(prediction)
    
    # Display personality type
    st.write('Predicted Personality Type:', predicted_personality[0])
    
    # Display additional details 
    if predicted_personality in personality_details:
        st.write('Personality Details:', personality_details[predicted_personality])
    else:
        st.write('Personality details not available for this type.')