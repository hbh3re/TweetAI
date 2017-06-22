import tweepy
import string
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

# could turn this into a do-while loops later to get rid of repetition
def get_tweets(user, api_key, api_secret, access_token, access_token_secret):
	# credit to Github user yanofsky for code in part of this function
	# https://gist.github.com/yanofsky/5436496

	# create authorization to access Twitter API through tweepy
	auth = tweepy.OAuthHandler(api_key, api_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)

	# initialize a list to hold all the tweets accessed through tweepy
	alltweets = []

	print("Grabbing 200 tweets at a time...")
	# make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(user, count=200)
	# save most recent tweets
	alltweets.extend(new_tweets)
	# save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	# print most recently fetched tweet
	# print("Last tweet fetched: ", alltweets[-1].text)

	# loop to grab the rest of the tweets possible
	# max you can grab is somewhere around 3000 due to API limitations
	while len(new_tweets) > 0:
		print("Getting tweets before id: " + str(oldest))
		new_tweets = api.user_timeline('realDonaldTrump', count=200, max_id=oldest)
    	#save most recent tweets
		alltweets.extend(new_tweets)
    	# update oldest so that we do not fetch repeats
		oldest = alltweets[-1].id - 1
		print(str(len(alltweets)), " tweets downloaded so far...")
		# print most recently fetched tweet
		# print("Last tweet fetched: ", alltweets[-1].text)

	print("All possible tweets have been retrieved")
	return alltweets

def preprocess_tweets(tweets):
	# remove retweets while also converting tweets into string form
	non_retweets = [t.text for t in tweets if 'RT @' not in t.text]

	processed_tweets = []
	for tweet in non_retweets:
		new_tweet = ' '.join(word for word in tweet.split() if word[0:4] != 'http')
    	# changes '&amp;' to simply '&'
		# NEW - maybe replace with 'and' so it will learn how to spell 'and' rather have to learn where to place &
		# idk bru
		if '&amp;' in new_tweet:
			new_tweet = new_tweet.replace('&amp;', '&')
		# removes all emojis from tweets
		# technically removes all characters that are not letter, numbers, punctuation, or whitespace
		for c in list(new_tweet):
			if not c.isalnum() and c not in string.punctuation and c not in string.whitespace:
				new_tweet = new_tweet.replace(c, '')

		processed_tweets.append(new_tweet)

    	# remove extraneous links
		if 'http' in new_tweet:
			processed_tweets.remove(new_tweet)

	print("Number of tweets: ", len(processed_tweets))
	max_len = max(map(len, [tweet for tweet in processed_tweets]))
	print("Longest tweet in characters: ", max_len)
	min_len = min(map(len, [tweet for tweet in processed_tweets]))
	print("Shortest tweet in characters: ", min_len)

	return processed_tweets

# does this need to show/print/return anything?
def get_length_histogram(tweets):
	lengths = []
	for tweet in tweets:
		lengths.append(len(tweet))

	bin_width = 20
	plt.hist(lengths, bins=range(min(lengths), max(lengths)+bin_width, bin_width))
	plt.title("Lengths of tweets")
	plt.xlabel("Length")
	plt.ylabel("Frequency")

def get_indices(chars):
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	return (char_indices, indices_char)

def get_train_data(tweets, maxlen, step, chars):
	# cut the text in semi-redundant sequences of maxlen characters
	# idea for later cut text relative to size of tweet aka maxlen = len(tweet)/3

	sentences = []
	next_chars = []
	for tweet in tweets:
		if len(tweet) > maxlen:
			for i in range(0, len(tweet) - maxlen, step):
				sentences.append(tweet[i: i + maxlen])
				next_chars.append(tweet[i + maxlen])
	    # two options here:
	    # could iterate with a range like above, using lengths of 10 (min tweet length)
	    # or could just get one sentence from each tweet based on length of each tweet
		else:
			sentences.append(tweet[len(tweet)-2])
			next_chars.append(tweet[len(tweet)-1])
	print('nb sequences:', len(sentences))

	# might need to pad shorter sentences? jk apparently not
	# creating 3D 'one-hot' matrices for sentences and next_chars
	print('Vectorization...')

	char_idx = get_indices(chars)[0]

	X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
	y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			X[i, t, char_idx[char]] = 1
		y[i, char_idx[next_chars[i]]] = 1

	return (X, y, sentences)

def build_model(maxlen, chars):
	# Building the model
	print('Building model...')
	model = Sequential()
	model.add(LSTM(128, input_shape=(maxlen, len(chars)))) # might have problems if fed a shorter tweet than maxlen?
	model.add(Dense(len(chars)))
	model.add(Activation('softmax'))

	optimizer = RMSprop(lr=0.01) # experiment by adding ReduceLROnPlateau
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)

	return model

# essentially a softmax function with an option to change the degree of variability
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train_model(model, X, y, sentences, maxlen, chars, num_epochs):

	# train the model, output generated text after each iteration
	for iteration in range(1, num_epochs):
		print()
		print('-' * 50)
		print('Iteration', iteration)
		model.fit(X, y,
	              batch_size=128,
	              epochs=1)

		start_index = random.randint(0, len(sentences)) # choose index to get random tweet beginning

		char_idx, idx_char = get_indices(chars)
		for diversity in [0.2, 0.5, 1.0, 1.2]:
			print()
			print('----- diversity:', diversity)

			generated = ''
			sentence = sentences[start_index] # get the tweet using the random index
			generated += sentence
			print('----- Generating with seed: "' + sentence + '"')
			sys.stdout.write(generated)

			for i in range(140-maxlen): # max length of a tweet is 140 characters
				x = np.zeros((1, maxlen, len(chars)))
				for t, char in enumerate(sentence):
					x[0, t, char_idx[char]] = 1.

				preds = model.predict(x, verbose=0)[0]
				next_index = sample(preds, diversity)
				next_char = idx_char[next_index]

				generated += next_char
				sentence = sentence[1:] + next_char

				sys.stdout.write(next_char)
				sys.stdout.flush()
			print()

if __name__ == '__main__':
	# pass the user you wish to learn as the only command line parameter (string)
	user = sys.argv[1]
	api_key = 'OX6EZZ9kea7e3QgICtbN5fep0'
	api_secret = 'F5IYMC7pEUo4ntf2mSU9vAuO4Z4tGa3pb2yKgnxR4cEZItJrWa'
	access_token = '360249871-fMMBIO5myYfsFfDE35FYak1EU1p3rACWCBP0BZga'
	access_token_secret = 'kqZz9dzHEPsMkEZWXCbWlucdW6bYAoP0nYTh6g5QI5vkC'

	# length of our seeds
	MAXLEN = 30

	raw_tweets = get_tweets(user, api_key, api_secret, access_token, access_token_secret)

	tweets = preprocess_tweets(raw_tweets)
	get_length_histogram(tweets)

	chars = sorted(list(set(' '.join(tweets))))

	X, y, seeds = get_train_data(tweets, maxlen=MAXLEN, step=3, chars=chars)

	model = build_model(maxlen=MAXLEN, chars=chars)
	train_model(model, X, y, sentences=seeds, maxlen=MAXLEN, chars=chars, num_epochs=30)
