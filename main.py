
import pandas as pd
from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Final_Datframe.csv')
data['label'] = data['label'].replace({'not_spam': 0, 'spam': 1})
# Split the data into features (X) and labels (y)
X = data['text']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Count Vectorizer to convert text data into numerical features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_spam', methods=['POST'])
def check_spam():
    user_input = request.form['email_text']
    user_input_vectorized = vectorizer.transform([user_input])
    result = model.predict(user_input_vectorized)
    return render_template('display.html',user_imput=user_input_vectorized,result=result)

    # return f'The input is {"spam" if result == 1 else "not spam"}.'

