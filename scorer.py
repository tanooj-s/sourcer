import numpy as np
import pandas as pd
import ast
import util
import math

def get_similarity(in_phrase,words,embeddings): 
	# word embeddings is assumed to be a dict of {word: embedding} pairs of type {string: np array}
	# words is assumed to be a list of lists of form [word, freq], types - [str, int]
	# if it is a string then it is converted to the [[w1,f1],[w2,f2],...] form

#	words = ast.literal_eval(words)
	if isinstance(words,str):
		words = util.word_freq(util.tokenize(words))  # -----> tokenizing takes out NEs by default

	a = np.zeros(len(embeddings['example'])) # initialize vector to same dimension as preloaded embeddings
	for word in util.tokenize(in_phrase):
		a = a + embeddings[word]	# -----> this should also be weighted more by thing than category
	
	a = a/np.sqrt(np.dot(a,a))	# -----> normalize

	b = np.zeros(len(embeddings['example']))
	for [word, freq] in words:
		if word in embeddings.keys():
			b = b + freq*embeddings[word] # weights 
	
	b = b/np.sqrt(np.dot(b,b))	# -----> normalize

	similarity = np.dot(a,b) # normalized weighted inner product by frequency
	return similarity



def get_distance(in_phrase,words,embeddings): 
	# word embeddings is assumed to be a dict of {word: embedding} pairs of type {string: np array}
	# words is assumed to be a list of lists of form [word, freq], types - [str, int]
	# if it is a string then it is converted to the [[w1,f1],[w2,f2],...] form

#	words = ast.literal_eval(words)
	if isinstance(words,str):
		words = util.word_freq(util.tokenize(words))  # -----> tokenizing takes out NEs by default
	a = np.zeros(len(embeddings['example'])) # initialize vector to same dimension as preloaded embeddings
	for word in util.tokenize(in_phrase):
		a = a + embeddings[word]	
	a = a/np.sqrt(np.dot(a,a))
	b = np.zeros(len(embeddings['example']))
	for [word, freq] in words:
		if word in embeddings.keys():
			b = b + freq*embeddings[word] # weights 
	b = b/np.sqrt(np.dot(b,b))
	
	distance = np.sqrt(np.sum(np.square(a - b)))  # euclidean distance between weighted vectors
	distance = (2 - distance)/2 # -------> transform so that more relevant targets get a higher score from this function
	return distance



def inclusive_mean(lst):
	"""
	lst is a list/series/array of numbers (int, float etc) that might include NaNs. Find the mean of non-NaNs.
	"""
	deno = 0
	nume = 0
	for ob in lst:
		if (ob is not None) and (math.isnan(ob) == False):
			nume += ob
			deno += 1
	return float(nume/deno)




def final_score(s1,s2):
	"""
	s1,s2 are float inputs or None. Return the mean of the two scores unless one is significantly higher than the other.
	(threshold may be a function of each dataset, need to check)
	"""
	if isinstance(s1,float) == False:
		return s2
	if isinstance(s2,float) == False:
		return s1
	if (s1 - s2 > 0.25):
		return (2*s1 + s2)/3
	elif (s2 - s1 > 0.25):
		return (2*s2 + s1)/3
	else:
		return (s1+s2)/2

