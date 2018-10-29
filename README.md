This tool is meant to identify websites that are related to a particular niche. Currently it is used primarily by IntVentures to find potential acquisition prospects
for their clients in specific SaaS/eCommerce verticals but it should be generalizable across topics.

When using this tool there should also be two subfolders (word_embeddings/ and competitor_lists/) in the working directory besides the Python scripts.  


------DEPENDENCIES------

In order to use this tool, the user must have the NLP library spaCy already installed as well as the 'en_core_web_sm' language model already installed. spaCy can be 
installed using either 'pip install -U spacy' or 'conda install -c conda-forge spacy'. Once the library is installed, the language model can be downloaded with 
'python -m spacy download en_core_web_sm' (although larger language models will be used once this is ready to be deployed). Other Python libraries that this tool uses
 - requests, numpy, pandas, ast, tqdm, bs4, eventlet, idna.  

The user must add competitor list csv files from SEMrush to the competitor_lists/ directory before running the tool. These competitor lists are the primary (and 
currently only) source of data this tool uses. However in the future this step can likely be automated as well.

There must also be a pretrained txt file with word embeddings in the word_embeddings/ directory. Currently this model uses GloVe word embeddings, which can be downloaded 
from nlp.stanford.edu/projects/glove/. 


------PIPELINE------

This tool performs three main tasks - first, it consolidates all the websites in the various competitor lists by deleting duplicates and getting rid of totally irrelevant
webaites. Second, it makes an HTTP request to each website in the consolidated list and if there is a good response extracts text data from each website. Finally, from
the extracted text data, and using the pre-trained word embeddings, for each website it creates a 'website vector' that is a n-dimensional representation of what that
website is about (dimensionality specified by whichever word embeddings you are using). Using a similar representation for the input phrase entered by the user, the 
scorer.py script then uses a combination of cosine similarity and Euclidean distance to provide a real valued score between -1 and 1 for how relevant each website is to the input phrase. The websites are then
sorted from highest to lowest score. Once this is completed the tool outputs a csv file with the text data, scores and whatever data was provided from SEMrush for each website.


------USAGE------

The tool can be run by entering the command 'python main.py' in the command line (make sure you are using Python 3). The script will first load the language model
and the word vectors (will take about 2 minutes), then will ask the user for 3 prompts - an input phrase, a traffic filter threshold, and whether or not the user 
wants to check if the websites have job boards. For the tool to be effective, the input phrase should ideally be 2-5 words long (although there is no upper limit). 
Once the user has entered the inputs they can leave their computer and come back in a couple of hours to check results. On average, the script takes about 1hr/1000
websites to extract information from and evaluate. Currently (for memory reasons) it limits the number of websites to extract information from to 3000 (by gradually
cranking up the SEMrush competitor relevance filter threshold until there are less than 3000 left) but again, once this is ready to deploy it can likely work on an 
arbitrary number of websites. 

So far the tool has been tested on AWS EC2 instances. Performance will likely improve once there is a dedicated machine with high RAM. 
