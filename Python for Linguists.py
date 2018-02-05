
# coding: utf-8

# # Python for Linguists
# I'm going to try something novel: giving this talk from a Jupyter notebook so I can run code on the fly.

# ## In this talk...
# * Who stands to gain the most
# * Why consider _programming,_ generally?
# * Why consider _Python,_ specifically?
# * Some demos

# ## Who this talk is for
# * Anyone who deals with _data:_ people interested in corpus work, sociolinguistics, natural language processing, digital/online language, etc.
# * Anyone interested in _computational social science_ (CSS): i.e. general social science approaches leveraging large datasets and computational horsepower.
#   * CSS is currently exploding, and is a hugely important avenue for applied social science research.
#   * CSS is also massively interdisciplinary: programming, statistics, machine learning, AI, network analysis, linguistics, sociology, psychology, etc all combine to make CSS happen.
# * If you deal mostly with theory, or are primarily an experimentalist, you probably stand to gain less from this talk.

# ## What does programming offer?
# * (Quite literally) infinite control over your data processing: you're not limited by the features someone else decided to code into their program--you can change your code up to do anything you want.
# * Scalability and automation of your data work
#   * Work with literally millions of documents and billions of words with relative ease.
#   * Automate steps from data collection through final analysis.
# * Marketable skills: even a little bit of Python, Java, or any other language can open doors in the job market.
# * You'll feel like a badass.

# # What does _Python_ offer?
# * Free (as in speech, not beer.  But also as in beer), open-source, royalty-free.  No licenses to sign, no royalties to pay, and _essentially no restrictions_ on what you can and can't do with it.  (the [Python Software Foundation license](https://docs.python.org/3/license.html) is an extremely permissive BSD-type license)
# 
# * Huge userbase that's big into Open Source and Free Software--so it's easy to find help or sample code.
# 
# * Rapidly becoming _the_ language for data science, displacing even R in most applications.  (R is still dominant for raw statistics, though Python has plenty of packages that implement common statistical tests).
#   * Though, keep an eye on a different language--Julia--over the next few years.  It is truly a worthy contender, but has yet to hit version 1.0 as of this talk.
# 
# * Easy-to-learn language.
#   * Great documentation and stupid amounts of free, high-quality learning resources.
#   * Among its core ideas:
#     * Code is read far more than it is written, so the language should be _human-readable._
#     * "There should be one, and preferably only one, obvious way to do it."  I.e., the most straightforward approach is _usually_ the best.  (This results in a lot of people writing straightforward, fairly easy-to-follow code).
#   * Commonly taught as a first programming language, so there are LOTS of materials for eveyone from beginning programmers to seasoned professionals; the Python community is also very welcoming of newcomers.
# 
# * General purpose language: can do (almost) everything you want to make it to.
#   * Compare to R, which is great for statistics, and a pain for a lot of other stuff.
#   * Or Matlab, which is great for being a broken, slow, difficult software environment, and isn't so good at being, well, good.
#     * (this has been your mandatory "Matlab is bad" comment)
# 
# * **For linguists**: a _huge_ array of language processing functionality and libraries.
#   * [spaCy](https://spacy.io), basically a Python version of Stanford's CoreNLP toolkit (lemmatization, tokenization, dependency parsing, POS tagging, and more).
#   * [Gensim](https://radimrehurek.com/gensim/), full of topic models and pretty bleeding-edge NLP tools.
#   * [Natural Language Toolkit (NLTK)](http://www.nltk.org/), a _massive_ library that's designed to teach a lot of NLP concepts (but can be used for some serious production work too).
#   * [Tensorflow](https://www.tensorflow.org/)+[Keras](https://keras.io/), for quickly and easily building neural networks.
#   * [PyTorch](http://pytorch.org/), an up-and-coming (but extremely exciting) neural network library.
#   * [Pandas](http://pandas.pydata.org/) for R-like dataframes, statistics, and general tabular data management.
#   * [Matplotlib](https://matplotlib.org/) (and others like [Seaborn](https://seaborn.pydata.org/), [PyGal](http://www.pygal.org/en/stable/), [Bokeh](https://bokeh.pydata.org/en/latest/), ...) for high-quality, powerful data visualization.
#   * [scikit-learn](http://scikit-learn.org/stable/index.html) for non-neural machine learning (support vector machines, random forests, and a few text features like basic preprocessing)
#     * Side note, the scikit-learn [User Guides](http://scikit-learn.org/stable/user_guide.html) are an _excellent_ technical crash course in machine learning, even if you're not too interested in Python.
#   * [Networkx](https://networkx.github.io/) for performing network analysis.
#   * And dozens more.

# # Demo time.

# These demos will focus on text analysis and natural language processing, which is the sort of work that I'm most familiar with--though there are plenty of other applications out there.  We'll be using the text of each episode of [Engines of Our Ingenuity,](http://uh.edu/engines/epiindex.htm) as a toy dataset.  Engines of Our Ingenuity is a long-running, 3-5 minute daily radio segment broadcast on Houston's NPR station, and focuses broadly on the people involved in history of science and technology.  (It's also available as a podcast, and I highly recommend it)
# 
# We'll look at some common data prep approaches:
# * Tokenization
# * Stemming and lemmatization
# * POS tagging
# * Dependency parsing
# * Noun chunk and named entity detection
# 
# We'll go through a few different approaches to cleaning the data, and preparing it for some simple analyses:
# * Topic modeling (LSA, LDA, author-topic models)
# * Word and document vectorizations/embeddings with Word2Vec
# * Identifying authors using n-grams and word/document embeddings.

# In[176]:


# First step: read in the data (provided in XML format) and extract the text into
# a Pandas dataframe.

# Modules from the standard library
import csv
import os

# Third-party module--needs to be installed separately
from tqdm import tqdm

def scan_recursive(directory):
    """
    Recursively scan a directory to extract all files.
    """
    for i in os.scandir(directory):
        if os.path.isfile(i): 
            yield i.path
        elif os.path.isdir(i):
            yield from scan_recursive(i.path)

DIR = "engines/textonly"
files = list(scan_recursive(DIR))
files


# Now, we need to open each of these files and do some preprocessing to isolate the body of the transcript from the front and back matter.  We'll split it into a dictionary in the form `{"Author":author, "Transcript":transcript of episode}`.

# In[177]:


# Standard library imports
from string import punctuation
import re

def parse_episode(infile):
    """
    Parse the input file to extract the author and
    episode transcript.  Return it as a JSON so
    we can easily make a dataframe out of a list 
    of parse_episode() calls.
    """
    transcript = open(infile, "r", encoding="utf8").read()
    
    # use a regular expression to split at the "click here for audio"
    # link, which ALWAYS comes between the author byline and the 
    # main transcript text.
    transcript = re.split(
        r"Click here for audio of Episode [0-9]+\.?|\(theme music\)", 
        transcript,
        flags=re.IGNORECASE|re.MULTILINE
    )
    
    # we should have three chunks: [front matter, text, back matter]
    # If we don't, something's gone wrong, and for this demo we'll
    # just ignore it
    if len(transcript) != 3:
        return 0,0
    author = [
        i 
        for i in transcript[0].split("\n")
        if i.strip() != ""
    ][-1].strip()
    # sometimes this string has "John Lienhard presents (guest) so-and-so"
    if "presents guest" in author:
        author = author.split("presents guest ")[-1]
    elif "presents" in author:
        author = author.split("presents ")[-1]
    # remove leading "by" and "essayist"
    author = author         .replace("by: ", "")         .replace("by ", "")         .replace("essayist, ", "")         .replace("essayist ", "")
    author = author.strip()
    
    # Combine some alternate spellings together
    if author == "Andrew Boyd": 
        author = "Andy Boyd"
    if author in ("John H. H. Lienhard", "John H. John H. Lienhard",
                    'John H. Lienhard', 'John H.Lienhard'):
        author = "John Lienhard"
    if author in ("Kreso Josic",  "Kre o Josić"):
        author = "Krešo Josić"
    if author == "Kre imir Josić":
        author = "Krešimir Josić"
    if author in ("the Rev. John Price", "REV. John W. Price"):
        author = "John Price"
    if author in ("Richard H. Armstrong", "Richard Armstong"):
        author = "Richard Armstrong"
        
    # Get the episode number out of the file name
    episode_num = int(infile[:-4].rsplit("epi")[-1])
    
    return {
        "Author":author, 
        "Transcript":re.sub(r"\s+", " ", transcript[1]).strip(),
        "Episode":episode_num
    }


# In[192]:


import pandas as pd

episode_data = [parse_episode(i) for i in tqdm(files, desc="Parsing episode data")]
# remove errors, if any
episode_data = [i for i in episode_data if i != (0,0)]
episode_data = pd.DataFrame(episode_data)
# .reset_index() is just because I want to have a nice 
episode_data = episode_data.sort_values("Episode").reset_index(drop=True)
episode_data


# In[179]:


# Some basic preprocessing in Gensim.
from gensim.parsing import preprocessing

sample_text = list(episode_data["Transcript"])[0]
sample_text


# `preprocess_string()` does the following steps:
# * Converts text to lowercase
# * Removes accents from letters
# * Strips all whitespace
# * Filters tokens by length
# * Removes stopwords (Gensim has a pretty aggressive stopword list)
# * Removes numbers (i.e. digits 0-9)
# * Runs the Porter Stemmer for English on the text.
# 
# It returns a list of processed words in the original text; usin ``" ".join()` we can convert this back to a more human-readable string.

# In[180]:


" ".join(preprocessing.preprocess_string(sample_text))


# To a linguist this output should look like nonsense.  Word forms have been mutilated, a lot of vocabulary has been stripped, and things are generally a mess.  But, this reduces the complexity of the data enough that we can work with it in reasonable ways--it's a _good enough for purpose_ representation of the original data.
# 
# The Porter Stemmer's results look a bit weird because the stemmer is a "dumb" tool--it only looks at the characters that comprise the end of a word.  In cases where the results need to be human-readable, the Porter stemmer isn't the best choice (e.g.: _today_ became _todai_).  For that, we would use a _lemmatizer,_ which preserves human-readable forms, but at the expense of being considerably slower.
# 
# spaCy's language models have a lemmatizer that's dead-easy to use.

# In[181]:


import spacy
# create the NLP parser.  We'll use the small model--
# faster, but less accurate, then others in the library--
# for the purposes of this demo.
print("Loading NLP model, may take a moment...")
nlp = spacy.load("en_core_web_sm")
doc = nlp(sample_text)
" ".join(
    i.lemma_ 
    for i in doc
    if i.is_stop == False
    and i.is_punct == False
)


# This looks a lot more readable to a human.  It doesn't come across for a small example like this, but this processing approach is actually _considerably_ more time-intensive than the simple Gensim version, because a pre-trained model has to be consulted.
# 
# While we've got a spaCy-parsed document, we can also look at some of its other features, like named entity recognition (which is far from perfect, but has pretty decent recall):

# In[182]:


list(doc.ents)


# Noun chunk (roughly equivalent to NP) identification:

# In[183]:


list(doc.noun_chunks)


# Dependency relations:

# In[184]:


for token in doc:
    print(f"{token.text:<25s} :: {token.dep_:<15s} :: {token.head}")


# Sentence detection:

# In[185]:


list(doc.sents)


# POS tags, using both fine-grained and coarse-grained annotation schemes:

# In[186]:


for token in doc:
    print(f"{token.text:<25s} :: {token.pos_:<15s} :: {token.tag_}")


# Though, for a more visual picture, we can use the the displacy tool to visualize this dependency parse:

# In[187]:


from spacy import displacy
displacy.render(
    nlp("Today, we see what guns and steam engines have to do with each other."), 
    style="dep",
    jupyter=True # to make this render correctly in the Jupyter notebook
)


# We'll use spaCy to do some our preprocessing, just because it's lemmatization leaves things much more readable to a human interpreter.  This will be important later.

# In[ ]:


# re-load the NLP model with some features disabled to make it go faster.
from time import time
nlp = spacy.load(
    "en_core_web_sm",
    disable=["tagger", "ner"]
)
# spaCy has built-in multithreading! Woohoo!
# Note that the pipeline doesn't actually start doing any
# processing until you iterate through it, though.
docs = nlp.pipe(
    tqdm(list(episode_data["Transcript"])), # seems to break unless we use list(), no clue why
    n_threads=3,
    batch_size=100
)

# Remove stopwords and punctuation
docs = [
    [
        i.lemma_ for i in j
        if i.is_stop == False
        and i.is_punct == False
    ]
    for j in docs
]
docs


# In[ ]:


print("test")

