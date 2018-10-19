"""
This script takes in a folder that contains SEMrush competitor list files (.csv files), filters out ones that are not relevant enough and have too much traffic
deletes duplicates, and extracts metadata for the websites where it exists and outputs a consolidated list of the data, sorted by relevance to input phrase
(assumed for the moment to be some kind of e-commerce/SaaS niche but this should be generalizable).
"""

import os
import csv
import glob
import datetime
import argparse


import numpy as np
import pandas as pd

from tqdm import tqdm

import spacy
import util
import scorer
import extractor


now = datetime.datetime.now().strftime('%Y%m%d')

parser = argparse.ArgumentParser()
parser.add_argument('-output_file', action='store', dest='output_file', help='Output csv file', default='output_' + now + '.csv')
args = parser.parse_args()
tqdm.pandas(desc='Progress bar')

in_phrase = input("Enter a phrase for the niche you are interested in (e.g. 'job board software', 'resume builders'): ")
job_board_search = input("Do you want to check if these websites are job boards? (Y/N): ")
traffic_threshold = int(input("Please enter the upper organic traffic threshold to filter results by (default 100000): "))
#exclusions = input("Are there any words/niches you don't want? Enter as keywords (e.g. 'music writing'). If no, type N: ")

print("Loading language model...")
language_model = util.language_model # once this is ready to deploy should be using a larger model

print("Loading word embeddings...")
word_embeddings = util.load_word_embeddings('word_embeddings/glove.6B.300d.txt') # use 42B common crawl when ready for prod, possibly just import directly from spaCy
print("Word embeddings and language model loaded!")

competitor_list_files = []
competitor_dataframes = []
path = "competitor_lists/"

for filename in glob.glob(os.path.join(path, '*.csv')):
	competitor_list_files.append(filename)
print ("There are " + str(len(competitor_list_files)) + " competitor list files to extract information from.")
for csv_file in competitor_list_files:
	competitor_dataframes.append(util.filter_competitor_list(csv_file, traffic_threshold))
print ("Filtered out sites based on traffic and competitor relevance!")

df = pd.concat(competitor_dataframes)
df.drop_duplicates(subset=['Domain'],inplace=True)
print ("Deleted duplicates!")

# this block should hopefully not be necessary running on a remote machine with high RAM (at least 8GB)
print("Increasing competitor relevance filter to reduce table size (avoid oom-killer).")
n_rows = df.shape[0]
filter_score = 0.01
while n_rows > 3000:
	df = df[df['Competitor Relevance'] > filter_score]
	filter_score += 0.001
	n_rows = df.shape[0]
print("Final competitor relevance filter threshold - " + str(filter_score))

df.drop(labels=['Competitor Relevance','Common Keywords'], axis=1, inplace=True) # these will be different for a single site that appears in two separate competitor lists, since this data is not independent of the competitor list source we can get rid of it
df.sort_values(by='Domain',inplace=True)
print (str(n_rows) + " websites to mine data from.")

#websites_reached = 0

df['soup'] = df['Domain'].progress_apply(extractor.get_html_content) # this line takes about 2/3 of the total running time
n_unreached = df['soup'].isnull().sum() 
print("Got HTML content from " + str(n_rows - n_unreached) + "/" + str(n_rows) + " websites.")

print("Getting metadata of websites...")
df['metadata'] = df['soup'].progress_apply(extractor.get_meta)
print("Got meta descriptions of all websites!")

print("Getting homepage keywords...")
df['homepage keywords'] = df['soup'].progress_apply(extractor.get_homepage_keywords) # this line takes most of the remainder of the running time
print("Got homepage keywords of all websites!")

if(job_board_search.lower() == 'y'):
	print("Determining whether sites are job boards based on text in href elements")
	df['is job board'] = df['soup'].progress_apply(util.is_job_board)

df.drop(labels='soup', axis=1, inplace=True)


print("Calculating homepage keywords scores...")
df['score_homepage'] = df['homepage keywords'].progress_apply(lambda x: scorer.get_simple_projection(in_phrase=in_phrase,words=x,embeddings=word_embeddings))
print("Calculating website metadata scores...")
df['score_metadata'] = df['metadata'].progress_apply(lambda x: scorer.get_simple_projection(in_phrase=in_phrase,words=x,embeddings=word_embeddings))
print("Calculating net relevance scores of websites...")
df['score_net'] = df[['score_homepage','score_metadata']].progress_apply(lambda row: (row['score_homepage'] + 0.5*row['score_metadata'])/1.5, axis=1)


print('Ranking websites...')
df.sort_values(by='score_net',ascending=False).to_csv(args.output_file)
print("Targets sourced, sorted by relevance to input phrase.")






