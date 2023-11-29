import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup
from nltk import pos_tag, word_tokenize

messi_file_path = 'data/messi.json'
lebron_file_path = 'data/lebron.json'

with open(messi_file_path, 'r') as file:
    messidata = json.load(file)

with open(lebron_file_path, 'r') as file:
    lebrondata = json.load(file)

messistring = ' '.join(messidata)
lebronstring = ' '.join(lebrondata)





soup = BeautifulSoup(page.content, "html.parser")

for p in soup.find_all("p"):
    fulltext.appned(p.get_text())

print(fulltext)
                    





messitokens = nltk.word_tokenize(messistring)
lebrontokens = nltk.word_tokenize(lebronstring)











## REMOVE STOP WORDS                                                                                

stoplist = stopwords.words('english')
stoplist.extend([".", ",", "?", "could", "would", "“", "”", "’", ";", "!","much", "like", "one", "many", "though", "withzout", "upon"])
nostops = []
for tl in tokenlists:
    nostopwords = [w for w in tl if w.lower() not in stoplist]
    nostops.append(nostopwords)


## LEMMATIZE                                                                                                    
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# create a new list of tokens, alllemmas by lemmatizing allcontenttokens                                        
lemmatized = []
for tl in nostops:
    all_lemmas = [lemmatizer.lemmatize(w) for w in tl]
    lemmatized.append(all_lemmas)


## WORD CLOUDS FOR INDIVIDUAL PAGES                                                                                 
# The wordcloud library expects a string not a list,                                                            
# so join the list back together with spaces   
counter = 1
for all_lemmas in lemmatized:
    figurename = "wordcloud" + str(counter) + ".png"
    counter += 1
    text = " ".join(all_lemmas)

    # Generate a word cloud image                                                                               
    wordcloud = WordCloud().generate(text)

    # Display the generated image:                                                                              
    # the matplotlib way:                                                                                       

    # lower max_font_size                                                                                       
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(figurename)


## WORD CLOUD FOR ALL PAGES  
text = [" ".join(t) for t in lemmatized]
alltext = " ".join(text)

wordcloud = WordCloud(max_font_size=40).generate(alltext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("allwordcloud.png")


#!/usr/bin/env python
# coding: utf-8

# # Problem Set 5: n-grams, collocations, TF-IDF
# 
# For most this problem set, you'll be repurposing [the code from class 9.2 demonstrating how to use the nltk library to get n-grams and collocations](https://github.com/CSCI-2349-F23/sample_code/blob/main/class9.2/Class_9.2_ngrams.ipynb). You'll need to do a little of your own work for reading in files, writing functions, storing data properly, but nearly all the code you need is in the Class9.2 notebook. 
# 
# At the end of the problem set, you will paste in the TF-IDF code from the [sample code for Class 13.1](https://github.com/CSCI-2349-F23/sample_code/blob/main/class13.1/tfidf.ipynb) and edit it so that it works with your data.
# 
# Instructions for how to submit are at the end of the README. 
# 
# Both PS5 and PS6 are due Monday, November 27, at 11:59pm EST.
# 
# 
# ## Part 0: Install the necessary libraries.
# 
# You should have most of these installed already, but you can double check. The last two or three might be new for you.
# 
# ```
# python3 -m pip install jupyter
# python3 -m pip install nltk
# python3 -m pip install numpy
# python3 -m pip install matplotlib
# python3 -m pip install scikit-learn
# python3 -m pip install pandas
# python3 -m pip install altair
# ```

# 
# ## Part 1: Make sure you have the data
# 
# If you have not done so already, please follow the directions from the REAMDE to get your data. You should have a directory corresponding to one of your Wikipedia category pages, which contains one text file per Wikipedia page in that category.

# ## Part 2: Read in and tokenize the text files
# 
# The comments in the code block below will tell you what you need to do. 

# In[1]:


# Import statements
import re
import nltk
import glob
from nltk.corpus import stopwords

# Get the nlkt stopword list and improve it, as in prior problem sets.
stoplist = stopwords.words('english')
stoplist.extend([")", "(", ".", ",", "?", "could", "would", "“", "”", "’", ":",";", "!","much", "like", "one", "many", "though", "without", "upon"])

stoplist.extend([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
    "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", 
    "few", "for", "from", "further", 
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", 
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", 
    "me", "more", "most", "mustn't", "my", "myself", 
    "no", "nor", "not", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    ".", ","
])

# adding more punctuation
stoplist.extend(["''", '``'])



# In[2]:


# Write a function that reads the contents of a directory of
# Wikipedia text files into a list of lists of tokens,
# with one list per txt file.
# Argument: a string that is the name of the directory
# Returns: a list of lists of tokens, one per text file

# this list is to get the name of the files for the output of another part
files = []

def read_and_tokenize(directory_name):

    listoftokens = []
    # file in your code here!
    for filename in glob.glob(directory_name):

        files.append(filename)
        
        # open the file, read it in, replace newlines with space
        f = open(filename, encoding="utf=8")
        fulltext = f.read()
        alltext = re.sub("\n", " ", fulltext)
    
        # call nltk.word_tokenize
        alltokens = nltk.word_tokenize(alltext)

        listoftokens.append(alltokens)
    
    return listoftokens


# In[3]:


# Call the function on your directory
# and save what it returns as pagelist

# Example I would use for my Wiki category
# pagelist = read_and_tokenize("Endangered_animals")

# Write you code here
pagelist = read_and_tokenize('Actors_awarded_knighthoods/*')


# ## Part 3: Bigrams
# 
# Now you will count and print out bigrams in each of the lists of tokens. Make sure to print out the bigrams so that they look nice and not like Python tuples or lists. Here's an example of some output for my data. This is nice human-readable text. Try to make your output look like this!
# 
# ```
# Endangered_animals/Wattled_curassow.txt
# ** Most frequent bigrams **
# . The
# of the
# , and
# wattled curassow
# in the
# , the
# , but
# curassow (
# . It
# the wattled
# 
# ** Most frequent bigrams with no stop words **
# wattled curassow
# red-billed curassow
# black curassow
# C. globulosa
# black plumage
# ```
# 

# In[4]:


# Write a function that counts the bigrams for a string
# of tokens and then does the following:
# (1) prints out the 10 most frequent bigrams.
# (2) all bigrams in the top 50 where neither token is a stop word
# Argument: a list of tokens


def print_common_bigrams(tokenlist):
    
    # fill in your code here!

    # create the bigrams
    bigrams = nltk.ngrams(tokenlist, 2)
    bigramlist = list(bigrams)
    
    # print out most frequent bigrams
    bigramfreq = nltk.FreqDist(bigramlist)
    top10bigrams = bigramfreq.most_common(10)
    top50bigrams = bigramfreq.most_common(50)

    # part a)
    print('** Most frequent bigrams **')
    for x in top10bigrams:
        print(x[0][0], x[0][1])

    
    # part b)
    print('\n** Most frequent bigrams with no stop words **')
    for x in top50bigrams:
        if x[0][0].lower() not in stoplist and x[0][1].lower() not in stoplist:
            print(x[0][0], x[0][1])



# In[5]:


# Call your function on each token list in your list of lists
# from the Wikipedia directory but limit yourself to pages where 
# here the number of tokens is greater than 1000.

for x in range(len(pagelist)):
    if len(pagelist[x]) > 1000:
        print(files[x])
        print_common_bigrams(pagelist[x])
        print('')



        



# ## Part 4: Collocations
# 
# Now you are going to print out the most common collocations for each list of tokens. Use **PMI** to rank your collocations, and use a **frequency filter of 2**. Start with collocations where the **two words are adjacent** (the default behavior for collocations in nltk). 
# 
# As above, this code is all available in the [sample code for Class 9.2](https://github.com/CSCI-2349-F23/sample_code/blob/main/class9.2/Class_9.2_ngrams.ipynb).
# 
# **Continue to make the output look nice!** Here is an example of my output:
# 
# ```
# Endangered_animals/Wattled_curassow.txt
# ** Common Collocations **
# Development Reserve
# Mamirauá Sustainable
# Sustainable Development
# von Spix
# A captive
# well studied
# a. alector
# Adult male
# ancient lineage
# habitat destruction
# ```

# In[6]:


# import statement
from nltk.collocations import *

# Create the object you need to get collocations.
bigram_measures = nltk.collocations.BigramAssocMeasures()

# Write a function that prints out the top 10 collocations in
# a list of tokens using PMI for the ranking metric and 
# a frequency filter of 2.
# Argument: a list of tokens

def print_collocations(tokenlist):

    finder = BigramCollocationFinder.from_words(tokenlist)
    finder.apply_freq_filter(2)
    print('** Common Collocations **')
    for c in finder.nbest(bigram_measures.pmi, 10):
        print(" ".join(c))




# In[7]:


# Call your function on each token list in your list of lists
# from the first directory where the number of tokens is greater
# than 1000.

for x in range(len(pagelist)):
    if len(pagelist[x]) > 1000:
        print(files[x])
        print_collocations(pagelist[x])
        print('')


# ## Part 5: TF-IDF
# 
# TF-IDF stands for "term frequency - inverse document frequency". It's a way of measuring how common a word is in a document ("term frequency") relative to how common that word is in all your documents ("inverse document frequency"). This metric allows you to discover the words that are central to a particular document and make that document special or unique compared to other documents. 
# 
# For example, in my endangered animals example, all of the documents (of course) contain the same stop words, but they are also likely to contain many of the same content words -- *animal, endangered, threated, species*. Applying TF-IDF will allow the words that are particularly important for a particular document to be highlighted. 
# 
# This technique will be useful to your in your projects, when you try to highlight for your audience how different examples of the same kind of document (e.g., song lyrics, earnings calls, literature) are different from one another (e.g., over time, produced by a different company, written in a different genre).
# 
# In the sample code for class 13.1, you will find a notebook and my directory of Wikipedia pages for the Endangered Animals category. Run that code with that dataset so you can see how it works. Then come back here, and paste it in so that it works with your dataset. Of course you will have to change anything that has been hard-coded (e.g., how I look up information about the term "bird" or the document "African_bush_elephant").

# In[8]:


## Paste your code here!

# some import statements

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import pandas as pd
import glob

# Changed the directory name
directoryname = "Actors_awarded_knighthoods"

# Then run this code to get the files in that directory and their names.
text_files = glob.glob(directoryname + "/*.txt")
file_names = [Path(text).stem for text in text_files]

# This code here does all the tf-idf counting for you.
tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english')
tfidf_vector = tfidf_vectorizer.fit_transform(text_files)


# In[9]:


# This converts the results to a pandas dataframe, which makes it easier to
# process and visualize
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=file_names, columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.stack().reset_index()
tfidf_df = tfidf_df.stack().reset_index()
tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'term', 'level_2': 'term'})
tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)


# In[10]:


# This line of code just saves the above output to a variable so that you can query it.

top_tfidf = tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)

# This says "find all the documents that have bates in their top 10.

top_tfidf[top_tfidf['term'].str.contains('bates')]


# In[11]:


# This says "find the top ten words in the Tony_Robinson document

top_tfidf[top_tfidf['document'].str.contains('Tony_Robinson')]


# In[12]:


# this code will create the heatmap 

import altair as alt
import numpy as np


# adding a little randomness to break ties in term ranking
top_tfidf_plusRand = top_tfidf.copy()
top_tfidf_plusRand['tfidf'] = top_tfidf_plusRand['tfidf'] + np.random.rand(top_tfidf.shape[0])*0.0001

# base for all visualizations, with rank calculation
base = alt.Chart(top_tfidf_plusRand).encode(
    x = 'rank:O',
    y = 'document:N'
).transform_window(
    rank = "rank()",
    sort = [alt.SortField("tfidf", order="descending")],
    groupby = ["document"],
)

# heatmap specification
heatmap = base.mark_rect().encode(
    color = 'tfidf:Q'
)

# text labels, white for darker heatmap colors
text = base.mark_text(baseline='middle').encode(
    text = 'term:N',
    color = alt.condition(alt.datum.tfidf >= 0.23, alt.value('white'), alt.value('black'))
)

# display the three superimposed visualizations
(heatmap + text).properties(width = 600)


# ## Submission
# Complete this jupyter notebook. This will be your submission for PS5. Instructions for how to submit are provided in the README.
