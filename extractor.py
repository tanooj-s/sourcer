
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


def get_homepage_keywords(soup):
	"""
	Returns a list of 2-element dictionaries of the form [word, freq]. If the soup object is None then returns an empty list.
	"""
	if soup is not None:
		element_types = ['h1','h2','h3','h4','p','li','h5','h6','div',
						'article','header','footer','blockquote','figcaption',
						'menuitem']  # there may be other element types that you can add here
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

	else:
		return []


def get_meta(soup):
	description = ''
	if soup is not None:
		meta_elements = []
		for element in soup.select('meta'):
			meta_elements.append(element.attrs)
			# meta_elements should be a list of dictionaries here

		for meta_dict in meta_elements:
			if ('name' in meta_dict.keys()) and ('content' in meta_dict.keys()):
				if (meta_dict['name'] == 'description'):
					description = meta_dict['content']
	return description


