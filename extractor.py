
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
import eventlet
import idna
import util
from bs4 import BeautifulSoup


def log_error(e):
	"""
	This functions prints out errors found when the script is run.
	"""
	print(e)


def get_html_content(url):
	"""
	Attempts to get the content at 'url' by making an HTTP GET request. If the content-type
	of response is some kind of HTML/XML and is not None, return as a BeautifulSoup object.
	"""
	eventlet.monkey_patch()
	try:
		with(eventlet.Timeout(45)):
			try:
				with closing(get('http://' + str(url), stream=True, timeout=15)) as resp:
					if resp.status_code == 200 and resp.content != None:
						html_data = BeautifulSoup(resp.content, 'html.parser')
						print("Found HTML content for " + str(url))
						return html_data
					else:
						print("No content returned from request to " + str(url))
						return None
			except RequestException as e:
				log_error('Error during requests to {0}: {1}'.format(url, str(e)))
				return None
			except idna.IDNAError as e:
				log_error('IDNAError during requests to {0}: {1}'.format(url, str(e)))
				return None
			except NotImplementedError as e:
				log_error('Error during requests to {0}: {1}'.format(url, str(e)))
				return None
			except ValueError as e:
				log_error('Error during requests to {0}: {1}'.format(url, str(e)))
				return None
	except eventlet.timeout.Timeout as e:
		log_error('Request to {0} timed out.'.format(url))
		return None


def get_meta_contents(soup):
	"""
	Returns a dictionary meta_contents of text data from HTML meta elements that can be further processed.
	"""
	if soup is None:
		return None
	meta_contents = {
			'descriptions': [],
			'keywords': [],
			'titles': [],
			'types': []
			}	
	elements = soup.select('meta')
	attributes = []
	for e in elements:
		attributes.append(e.attrs)
	for meta_dict in attributes:
		if ('name' in meta_dict.keys()) and ('content' in meta_dict.keys()):
			if 'description' in meta_dict['name'].lower():
				meta_contents['descriptions'].append(meta_dict['content'].lower())
			if 'title' in meta_dict['name'].lower():
				meta_contents['titles'].append(meta_dict['content'].lower())
			if 'keywords' in meta_dict['name'].lower():
				meta_contents['keywords'].append(meta_dict['content'].lower())
			if 'abstract' in meta_dict['name'].lower():
				meta_contents['descriptions'].append(meta_dict['content'].lower())
		elif ('property' in meta_dict.keys()) and ('content' in meta_dict.keys()):
			if 'description' in meta_dict['property'].lower():
				meta_contents['descriptions'].append(meta_dict['content'].lower())
			if 'title' in meta_dict['property'].lower():
				meta_contents['titles'].append(meta_dict['content'].lower())
			if 'type' in meta_dict['property'].lower():
				meta_contents['types'].append(meta_dict['content'].lower())
		elif ('itemprop' in meta_dict.keys()) and ('content' in meta_dict.keys()):
			if 'description' in meta_dict['itemprop'].lower():
				meta_contents['descriptions'].append(meta_dict['content'].lower())
			if 'type' in meta_dict['itemprop'].lower():
				meta_contents['types'].append(meta_dict['content'].lower())
		continue
	return meta_contents


def get_meta_description(meta_dict):
	"""
	Get a human-readable description of the website that will be output to the dataframe.
	"""
	if meta_dict is None:
		return None
	if len(meta_dict['descriptions']) == 0:
		if len(meta_dict['titles']) == 0:
			return ''
		else:
			return meta_dict['titles'][0]
	return meta_dict['descriptions'][0]


def get_meta_type(meta_dict):
	"""
	If present, get the 'property':'og:type' attribute of the website.
	"""
	if meta_dict is None:
		return None
	if len(meta_dict['types']) == 0:
		return ''
	return meta_dict['types'][0]


def get_meta_text(meta_dict):
	"""
	Returns a list of 2-element dictionaries of the form [word, freq] from the metadata in the HTML.
	"""
	if meta_dict is None:
		return None
	descriptions = list(set(meta_dict['descriptions']))
	titles = list(set(meta_dict['titles']))
	keywords = list(set(meta_dict['keywords']))
	text_data = []
	if len(descriptions) > 0:
		for d in descriptions:
			text_data.extend(util.tokenize(d))
	if len(titles) > 0:
		for t in titles:
			text_data.extend(util.tokenize(t))
	if len(keywords) > 0:
		for k in keywords:
			text_data.extend(util.tokenize(k))
	return util.word_freq(text_data)


def get_homepage_keywords(soup):
	"""
	Returns a list of 2-element dictionaries of the form [word, freq] from the body of the HTML.
	"""
	if soup is None:
		return None

	element_types = ['h1','h2','h3','h4','p','li','h5','h6','div','article',
					'header','footer','blockquote','figcaption','menuitem']  
	# there may be other element types that you can add here
	element_tokens = []
	for element_type in element_types:
		for e in soup.find_all(element_type):
			if e is not None:
				if e.string is not None:
					element_tokens.append(util.tokenize(str(e.string)))
		# element_words is a list of lists of words, grouped together by which HTML element type they were
		# they need to be unwrapped
	flattened_tokens = [token for sublist in element_tokens for token in sublist]
	return util.word_freq(flattened_tokens)
