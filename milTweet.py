import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn import metrics
import numpy as np

df_orig = pd.read_csv("/Users/a./Downloads/training.1600000.processed.noemoticon.csv",encoding="ISO-8859-1")

df_orig.columns = ["0.","1.","2.","3.","4.","5."]

df_orig = df_orig[["0.","5."]]

df_orig = df_orig.dropna()


y = df_orig["0."].astype('int32')

X = df_orig["5."]


X = X.str.lower()
X = X.str.replace(r'^br$', ' ')
X = X.str.replace(r'\s+br\s+',' ')
X = X.str.replace(r'\s+[a-z]\s+', ' ')
X = X.str.replace(r'^b\s+', '')
X = X.str.replace(r'\s+', ' ')



vectorizer = TfidfVectorizer(max_features=10000,stop_words=stopwords.words("english"))
X = vectorizer.fit_transform(X)

pickle.dump(vectorizer,open("best_vect.sav",'wb'))


X_train,X_test,y_train,y_test = train_test_split(X,y)

lg = LogisticRegression(max_iter=10000,verbose=2)
lg.fit(X_train,y_train)
print(lg.score(X_test,y_test))

pickle.dump(lg,open('best_twitter_model.sav','wb'))






