import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

MODEL_PATH = "/Users/a./Desktop/sentiment_analysis/best_twitter_model.sav" 
TRANSFORMER_PATH = "/Users/a./Desktop/sentiment_analysis/best_vect.sav"
DATA_PATH = "/Users/a./Desktop/sentiment_analysis/reddit_crypto_comments.sav"

model = pickle.load(open(MODEL_PATH,'rb'))

transformer = pickle.load(open(TRANSFORMER_PATH,'rb'))

data  = list(pickle.load(open(DATA_PATH,'rb')))


print(nltk.sent_tokenize(data[0]))

X = []
print(len(data))
for i in range(0,len(data)):
	if(i%100 == 0):
		print(i)
	comment = data[i]
	# print(comment)
	comment = comment.replace("\n", "")
	comment = comment.lower()
	comment = re.sub(r'\s{2,}', '', comment)
	comment = re.sub(r'^br$', ' ', comment)
	comment = re.sub(r'\s+br\s+',' ',comment)
	comment = re.sub(r'\s+[a-z]\s+', ' ',comment)
	comment = re.sub(r'^b\s+', '', comment)
	comment = re.sub(r'\s+', ' ', comment)
	sentences = nltk.sent_tokenize(comment)
	for j in range(0,len(sentences)):
		sentence = sentences[j]
		words = nltk.word_tokenize(sentence)
		new_words= []
		for k in range(0,len(words)):
			word = words[k]
			if word not in stopwords.words("english"):
				new_words.append(word)
		sentences[j] = ' '.join(new_words)
	# print(comment)
	# a()
	X.append(comment)


print("transformed_data")
transformed_data = transformer.transform(X)
ans = model.predict(transformed_data)

pos =0
neg = 0

for an in ans:
	if(an == 0):
		neg+=1
	else:
		pos+=1

X = ['Pos','NEG']
y = [pos,neg]
x_pos = [i for i, _ in enumerate(X)]

plt.bar(x_pos,y,color='green')
plt.xlabel("SENTIMENT")
plt.ylabel("VALUES")
plt.xticks(x_pos,X)
plt.show()
with open("results.txt",'a') as f:
	f.write(f"pos :{pos} neg: {neg} total: {len(data)}")








