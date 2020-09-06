import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from flask import Flask, render_template, url_for, request
import joblib

app = Flask(__name__)
@app.route('/')

def home():
    return render_template('home.html')
@app.route('/predict', methods = ['POST'])
def predict():
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
    data['label'] = data['v1'].map({'ham': 0, 'spam':1})
    X  = data['v2']
    y = data['label']
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
    clf= MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    #joblib.dump(clf, 'NB_spam_model.pkl')
    #NB_spam_model = open('NB_spam_model.pkl', 'rb')
    #clf = joblib.open(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        predictions = clf.predict(vect)
    return render_template('result.html', prediction = predictions)
if __name__ == '__main__':
    app.run(debug=True)




