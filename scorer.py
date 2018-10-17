import numpy as np
import pandas as pd
import ast
import util




def get_simple_projection(in_phrase,words,embeddings): 
	# word embeddings is assumed to be a dict of {word: embedding} pairs of type {string: np array}
	# words is assumed to be a list of lists of form [word, freq], types - [str, int]
	# if it is a string then it is converted to the [[w1,f1],[w2,f2],...] form

#	words = ast.literal_eval(words)
	if isinstance(words,str):
		words = util.word_freq(util.tokenize(words))  # -----> tokenizing takes out NEs by default

	a = np.zeros(len(embeddings['example'])) # initialize vector to same dimension as preloaded embeddings
	for word in util.tokenize(in_phrase):
		a = a + embeddings[word]
	a = a/np.sqrt(np.dot(a,a)) # normalize by "magnitude" of input phrase
	
	b = np.zeros(len(embeddings['example']))
	for [word, freq] in words:
		if word in embeddings.keys():
			b = b + freq*embeddings[word] # weights 
	b = b/np.sqrt(np.dot(b,b)) # normalize 

	score = np.dot(a,b) # weighted inner product by frequency
	return score

## add functionality for "negative" keywords but don't subtract them without adding a multiplier with value < 1
## how to penalize? has to be from the b vector



