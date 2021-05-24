import praw
import pickle
import time
SECRET = ""
CLIENT_ID = ""



reddit = praw.Reddit(client_id=CLIENT_ID,
	client_secret=SECRET,username='',
	password='',user_agent='')


comments = pickle.load(open("/Users/a./Desktop/sentiment_analysis/reddit_crypto_comments.sav",'rb'))

subreddit = reddit.subreddit("all")
for comment in subreddit.stream.comments(pause_after=2):
	try:
		if(comment == None):
			break
		body = str(comment.body).lower()
		# if "crypto" in  body or "coin" in body or "bit coin" in body or "ada" in body or "ethereum" in body or "bitcoin" in body:
		comments.add(comment.body)
		print(len(comments))


		pickle.dump(comments,open("news_comments.sav",'wb'))
	except Exception as e:
		print(e)

