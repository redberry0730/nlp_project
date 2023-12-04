#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[2]:


import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
import string


# In[3]:


# load messi data into single string
messi_file_path = 'data/messi.json'

with open(messi_file_path, 'r') as file:
    messidata = json.load(file)

messitotalstring = ' '.join(messidata)

soup = BeautifulSoup(messitotalstring, 'html.parser')

p_tags = soup.find_all('p')

messistring=''
for tag in p_tags:
    messistring+=tag.get_text()+' '

messistring = re.sub("\n", " ", messistring)
messistring = re.sub("\s+", " ", messistring)
messistring = re.sub("\[.*?\]", " ", messistring)
messistring = messistring.lower()


# In[4]:


# load trout data into single string
trout_file_path = 'data/trout.json'

with open(trout_file_path, 'r') as file:
    troutdata = json.load(file)

trouttotalstring = ' '.join(troutdata)

soup = BeautifulSoup(trouttotalstring, 'html.parser')

p_tags = soup.find_all('p')

troutstring=''
for tag in p_tags:
    troutstring+=tag.get_text()+' '

troutstring = re.sub("\n", " ", troutstring)
troutstring = re.sub("\s+", " ", troutstring)
troutstring = re.sub("\[.*?\]", " ", troutstring)
troutstring = troutstring.lower()


# In[5]:


# load lebron data into single string
lebron_file_path = 'data/lebron.json'

with open(lebron_file_path, 'r') as file:
    lebrondata = json.load(file)

lebrontotalstring = ' '.join(lebrondata)

soup = BeautifulSoup(lebrontotalstring, 'html.parser')

p_tags = soup.find_all('p')

lebronstring=''
for tag in p_tags:
    lebronstring+=tag.get_text()+' '

lebronstring = re.sub("\n", " ", lebronstring)
lebronstring = re.sub("\s+", " ", lebronstring)
lebronstring = re.sub("\[.*?\]", " ", lebronstring)
lebronstring = lebronstring.lower()


# In[6]:


# load mahomes data into single string
mahomes_file_path = 'data/mahomes.json'

with open(mahomes_file_path, 'r') as file:
    mahomesdata = json.load(file)

mahomestotalstring = ' '.join(mahomesdata)

soup = BeautifulSoup(mahomestotalstring, 'html.parser')

p_tags = soup.find_all('p')

mahomesstring=''
for tag in p_tags:
    mahomesstring+=tag.get_text()+' '

mahomesstring = re.sub("\n", " ", mahomesstring)
mahomesstring = re.sub("\s+", " ", mahomesstring)
mahomesstring = re.sub("\[.*?\]", " ", mahomesstring)
mahomesstring = mahomesstring.lower()


# In[7]:


# load crosby data into single string
crosby_file_path = 'data/crosby.json'

with open(crosby_file_path, 'r') as file:
    crosbydata = json.load(file)

crosbytotalstring = ' '.join(crosbydata)

soup = BeautifulSoup(crosbytotalstring, 'html.parser')

p_tags = soup.find_all('p')

crosbystring=''
for tag in p_tags:
    crosbystring+=tag.get_text()+' '

crosbystring = re.sub("\n", " ", crosbystring)
crosbystring = re.sub("\s+", " ", crosbystring)
crosbystring = re.sub("\[.*?\]", " ", crosbystring)
crosbystring = crosbystring.lower()


# In[8]:


# tokenize all the athlete strings

messitokens = nltk.word_tokenize(messistring)
lebrontokens = nltk.word_tokenize(lebronstring)
trouttokens = nltk.word_tokenize(troutstring)
mahomestokens = nltk.word_tokenize(mahomesstring)
crosbytokens = nltk.word_tokenize(crosbystring)


# In[9]:


# compute the counts of the individual tokenlists and total number of tokens

print('Length of Messi Tokens:', len(messitokens))
print('Length of Lebron Tokens:', len(lebrontokens))
print('Length of Trout Tokens:', len(trouttokens))
print('Length of Mahomes Tokens:', len(mahomestokens))
print('Length of Crosby Tokens:', len(crosbytokens))

total = len(messitokens)+len(lebrontokens)+len(trouttokens)+len(mahomestokens)+len(crosbytokens)
print('Length of All Tokens:', total)


# In[10]:


# compiling list of stopwords

punctuation_stop_list = string.punctuation

# originial list of stop words
stoplist = stopwords.words("english")

# adding more stop words
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
    ".", ",", "'s", "--", "n't", "ha", "wa"
])

# adding punctuation
stoplist.extend(punctuation_stop_list)

# adding more punctuation
stoplist.extend(["''", '``'])

stopwords = set(stoplist)


# In[11]:


import spacy

# load the spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1200000

index=['Messi', 'Lebron', 'Trout', 'Crosby', 'Mahomes']

# create string from tokens
texts = [" ".join(messitokens), " ".join(lebrontokens), " ".join(trouttokens), " ".join(crosbytokens), " ".join(mahomestokens)]

# function to extract adjectives from a single document
def extract_adjectives(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == "ADJ"]

# extract adjectives from each text
adjectives_per_text = [extract_adjectives(text) for text in texts]

# printing 50 adjectives of each player
for x in range(len(index)):
    print(index[x])
    print(adjectives_per_text[x][:50])


# In[12]:


# implement lemmatization

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

messi_lemmatized = [lemmatizer.lemmatize(w) for w in messitokens]
lebron_lemmatized = [lemmatizer.lemmatize(w) for w in lebrontokens]
mahomes_lemmatized = [lemmatizer.lemmatize(w) for w in mahomestokens]
crosby_lemmatized = [lemmatizer.lemmatize(w) for w in crosbytokens]
trout_lemmatized = [lemmatizer.lemmatize(w) for w in trouttokens]

messi_nostopwords = [w for w in messi_lemmatized if w not in stopwords]
lebron_nostopwords = [w for w in lebron_lemmatized if w not in stopwords]
mahomes_nostopwords = [w for w in mahomes_lemmatized if w not in stopwords]
crosby_nostopwords = [w for w in crosby_lemmatized if w not in stopwords]
trout_nostopwords = [w for w in trout_lemmatized if w not in stopwords]


# In[13]:


# building word clouds w stopwords

print('Messi Word Cloud')

fdist = nltk.FreqDist(messi_lemmatized)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

messitext = " ".join(messi_lemmatized)                                                                                                                                                                
wordcloud = WordCloud(max_font_size=40).generate(messitext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/messi.png')
plt.show()

print('Lebron Word Cloud')

fdist = nltk.FreqDist(lebron_lemmatized)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

lebrontext = " ".join(lebron_lemmatized)                                                                                                                                                             
wordcloud = WordCloud(max_font_size=40).generate(lebrontext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/lebron.png')
plt.show()

print('Mahomes Word Cloud')

fdist = nltk.FreqDist(mahomes_lemmatized)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

mahomestext = " ".join(mahomes_lemmatized)                                                                                                                                                             
wordcloud = WordCloud(max_font_size=40).generate(mahomestext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/mahomes.png')
plt.show()

print('Crosby Word Cloud')

fdist = nltk.FreqDist(crosby_lemmatized)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

crosbytext = " ".join(crosby_lemmatized)                                                                                                                                                             
wordcloud = WordCloud(max_font_size=40).generate(crosbytext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/crosby.png')
plt.show()

print('Trout Word Cloud')

fdist = nltk.FreqDist(trout_lemmatized)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

trouttext = " ".join(trout_lemmatized)                                                                                                                                                             
wordcloud = WordCloud(max_font_size=40).generate(trouttext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/trout.png')
plt.show()


# In[14]:


# building word clouds w/o stopwords

print('Messi Word Cloud')

fdist = nltk.FreqDist(messi_nostopwords)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

messitext = " ".join(messi_nostopwords)                                                                                                                                                                
wordcloud = WordCloud(max_font_size=40).generate(messitext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/messi.png')
plt.show()

print('Lebron Word Cloud')

fdist = nltk.FreqDist(lebron_nostopwords)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

lebrontext = " ".join(lebron_nostopwords)                                                                                                                                                             
wordcloud = WordCloud(max_font_size=40).generate(lebrontext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/lebron.png')
plt.show()

print('Mahomes Word Cloud')

fdist = nltk.FreqDist(mahomes_nostopwords)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

mahomestext = " ".join(mahomes_nostopwords)                                                                                                                                                             
wordcloud = WordCloud(max_font_size=40).generate(mahomestext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/mahomes.png')
plt.show()

print('Crosby Word Cloud')

fdist = nltk.FreqDist(crosby_nostopwords)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

crosbytext = " ".join(crosby_nostopwords)                                                                                                                                                             
wordcloud = WordCloud(max_font_size=40).generate(crosbytext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/crosby.png')
plt.show()

print('Trout Word Cloud')

fdist = nltk.FreqDist(trout_nostopwords)
print("\nFrequency Distribution")
print(fdist.most_common(20))
print(" ")

trouttext = " ".join(trout_nostopwords)                                                                                                                                                             
wordcloud = WordCloud(max_font_size=40).generate(trouttext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig('img/trout.png')
plt.show()


# In[15]:


# creating bigrams

def print_common_bigrams(tokenlist):
    
    bigrams = nltk.ngrams(tokenlist, 2)
    bigramlist = list(bigrams)
    
    # print out most frequent bigrams
    bigramfreq = nltk.FreqDist(bigramlist)
    top10bigrams = bigramfreq.most_common(10)
    top50bigrams = bigramfreq.most_common(50)

    print('** Most frequent bigrams **')
    for x in top10bigrams:
        print(x[0][0], x[0][1])

    print('\n** Most frequent bigrams with no stop words **')
    for x in top50bigrams:
        if x[0][0].lower() not in stoplist and x[0][1].lower() not in stoplist:
            print(x[0][0], x[0][1])


print('Messi Bigrams')
print_common_bigrams(messi_lemmatized)
print('\nLebron Bigrams')
print_common_bigrams(lebron_lemmatized)
print('\nTrout Bigrams')
print_common_bigrams(trout_lemmatized)
print('\nCrosby Bigrams')
print_common_bigrams(crosby_lemmatized)
print('\nMahomes Bigrams')
print_common_bigrams(mahomes_lemmatized)


# In[16]:


# creating collocations

from nltk.collocations import *

# create the object for collocations.
bigram_measures = nltk.collocations.BigramAssocMeasures()

def print_collocations(tokenlist):

    finder = BigramCollocationFinder.from_words(tokenlist)
    finder.apply_freq_filter(2)
    print('** Common Collocations **')
    for c in finder.nbest(bigram_measures.pmi, 10):
        print(" ".join(c))

print('Messi Collocations')
print_collocations(messi_lemmatized)
print('\nLebron Collocations')
print_collocations(lebron_lemmatized)
print('\nTrout Collocations')
print_collocations(trout_lemmatized)
print('\nCrosby Collocations')
print_collocations(crosby_lemmatized)
print('\nMahomes Collocations')
print_collocations(mahomes_lemmatized)


# In[17]:


# creating tf-idf vectors for all lemmatized words

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import pandas as pd
import glob

index=['Messi', 'Lebron', 'Trout', 'Crosby', 'Mahomes']
alltexts = [" ".join(messi_nostopwords), " ".join(lebron_nostopwords), " ".join(trout_nostopwords), " ".join(crosby_nostopwords), " ".join(mahomes_nostopwords),]

tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')
tfidf_vector = tfidf_vectorizer.fit_transform(alltexts)

tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=index, columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df = tfidf_df.stack().reset_index()
tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document', 'level_1': 'term'})
top_terms = tfidf_df.sort_values(by=['document', 'tfidf'], ascending=[True, False]).groupby(['document']).head(10)

top_terms


# In[18]:


# creating heatmap for tf-idf vectors for all lemmatized words

import altair as alt
import numpy as np


# adding a little randomness to break ties in term ranking
top_tfidf_plusRand = top_terms.copy()
top_tfidf_plusRand['tfidf'] = top_tfidf_plusRand['tfidf'] + np.random.rand(top_terms.shape[0])*0.0001

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


# In[19]:


# creating tf-idf vectors for adjectives

index=['Messi', 'Lebron', 'Trout', 'Crosby', 'Mahomes']
alltexts = [" ".join(adjectives_per_text[0]), " ".join(adjectives_per_text[1]), " ".join(adjectives_per_text[2]), " ".join(adjectives_per_text[3]), " ".join(adjectives_per_text[4]),]

tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')
tfidf_vector = tfidf_vectorizer.fit_transform(alltexts)

tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=index, columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df = tfidf_df.stack().reset_index()
tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document', 'level_1': 'term'})
top_terms = tfidf_df.sort_values(by=['document', 'tfidf'], ascending=[True, False]).groupby(['document']).head(10)

top_terms


# In[20]:


# creating heatmap for tf-idf vectors for adjectives

# adding a little randomness to break ties in term ranking
top_tfidf_plusRand = top_terms.copy()
top_tfidf_plusRand['tfidf'] = top_tfidf_plusRand['tfidf'] + np.random.rand(top_terms.shape[0])*0.0001

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

