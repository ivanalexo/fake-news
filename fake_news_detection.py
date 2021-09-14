from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from google_trans_new import google_translator

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
loaded_model = pickle.load(open('fake_news_model_one.pkl', 'rb'))
data = pd.read_csv('news.csv')
translator = google_translator()

x = data['text']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

def fake_news_detection(news):
    print(news)
    txt = translator.translate(news, lang_src = 'es', lang_tgt = 'en')
    print(txt)
    news = txt
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_news = [news]
    vectorized_input_news = tfvect.transform(input_news)
    prediction = loaded_model.predict(vectorized_input_news)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = fake_news_detection(message)
        print(prediction)
        return render_template('index.html', prediction = prediction)
    else:
        return render_template('index.html', prediction = 'Error while predicting')

if __name__ == '__main__':
    app.run(debug=True)