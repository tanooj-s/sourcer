import numpy as np
import pandas as pd
import spacy

bad_pos = ['PUNCT','SPACE','SYM','PRON','DET','NUM','X','INTJ','PROPN','ADV','ADP']
language_model = spacy.load('en_core_web_sm') # should be a larger model when this is finally deployed

# There should be a pretrained .txt word vector embedding file in the current working
# directory with 300d vector representations of words in the English language.
def load_word_embeddings(input_file):
	# returns a dictionary with words as keys and embeddings as values
	model = {}
	with open(input_file,'r') as f:
		for line in f:
			word = line.split()[0]
			embedding = np.array([float(val) for val in line.split()[1:]])
			model[word] = embedding
		return model

def tokenize(text):
	"""
	From input string, add words to the list of tokens if they are not stop words and their parts-of-speech
	are not identified as spaces, symbols or numbers. Language model should be a spacy language model object
	that can automatically parse an input string with POS tags.
	"""
	tokens = []
	doc = language_model(text)
	for token in doc:
		if token.is_digit:
			continue
		if len(token.text) == 1:
			continue
		elif token.is_stop:
			continue
		elif token.lemma_ == '-PRON-':
			continue
		elif token.pos_ in bad_pos:
			continue
		else:
			tokens.append(token.lemma_)
	return tokens

# not being used yet but we could use things with MONEY/PERCENT/TIME/DATE tags potentially to score sites
def get_entities(text):
	"""
	From input string, return a list of named entities as recongized by spaCy. These should be a list of lists
	with the format [ne.text, ne.tag] of type [str, str].
	"""
	entities = []
	doc = language_model(text)
	for ent in doc.ents:
		entities.append([ent.text, ent.label_])
	return entities

def get_freq(item):
	"""
	Helper function for word_freq which allows that list of 2-element lists
	to be sorted by the 2nd element in each sublist.
	"""
	return item[1]


def word_freq(tokens):
	"""
	Given a list of tokens (words) in which there are assumed to be many repetitions,
	returns a list of 2-element lists of the form [token, frequency], sorted by frequency.
	"""
	## this should return a list of dictionaries instead
	word_list = []
	frequency_list = []
	indi_words = set(tokens)
	for this_word in indi_words:
		word_list.append(this_word)
		count = 0
		for token in tokens:
			if token == this_word:
				count += 1
		frequency_list.append(count)

	wf_pairs = [[w,f] for w,f in zip(word_list,frequency_list)]
	return sorted(wf_pairs,key=get_freq,reverse=True)


def filter_competitor_list(csv_file,traffic_threshold):
	"""
	Takes in csv_file as a string and traffic threshold as specified by user,
	returns a dataframe object with domains filtered out based on input. 
	"""
	df = pd.read_csv(csv_file)
	df = df[df['Organic Traffic'] <= traffic_threshold]
	n_rows = df.shape[0]
	thresh = 0.00001
	while n_rows > 1000:
		df = df[df['Competitor Relevance'] > thresh] 
		thresh += 0.000001
		n_rows = df.shape[0]
	return df


#	elif n_rows <= 1000:
#		df = df[df['Competitor Relevance'] > 0.01]
#		return df
#	else:
#		df = df[df['Competitor Relevance'] > 0.025]
#		return df


def is_job_board(soup):
	"""
	This function returns 1 if the input website is determined to have a job board or not. This is determined first by checking if there is
	an element on the page with text like "post job"/"advertise job"/"submit job" i.e. a place for an employer to post jobs. "browse jobs"/"search jobs"/"navigate jobs"
	throws up too many false positives. If such an element is found then the script returns a 1. Otherwise this returns a 0.
	"""
	if soup is None:
		return 0
	else:
		element_texts = set()
		for a in soup.select('a'):
			element_texts.add(a.text)
		element_texts = list(element_texts)
		evaluation = 0
		for et in element_texts:
			if (('job' in et.lower()) and (('post' in et.lower()) or ('advertise' in et.lower()) or ('submit' in et.lower()))):
				evaluation = 1
		return evaluation

