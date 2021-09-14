import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
data = pd.read_csv('news.csv')
x = data['text']
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
y_train
tfvect = TfidfVectorizer(stop_words='english', max_df = 0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)
classifier = PassiveAggressiveClassifier(max_iter = 50)
classifier.fit(tfid_x_train, y_train)
y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')
cf = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
pickle.dump(classifier, open('fake_news_model_one.pkl', 'wb'))