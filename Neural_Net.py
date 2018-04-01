## Question Answering
# Mengying Zhang, Alicia Ge
## Neural Network Method

##### Largely inspired from https://www.oreilly.com/ideas/question-answering-with-tensorflow

# Library
import json
import pprint
import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import operator
import csv
import copy
from nltk.parse.stanford import StanfordParser
import random
import pickle

random.seed(2018)
np.random.seed(2018)


# global variables listed here
glove_wordmap = {}



# -------------
# Read in data
# -------------
def read_data(filename, overide = False):
    """ Read json formatted file, and return a dictionary """

    with open(filename) as json_data:
        doc = json.load(json_data)
        return doc


# -------------
# Load Stanford’s Global Vectors(Glove)
# -------------
def load_Glove(filename, overide = False):
    """ 
      Input: filename, if overide, load from pickle
      Output: modify glove_wordmap
    """
    global glove_wordmap

    if overide:
        glove_wordmap = pickle.load(open("Glove.p","rb"))
        return glove_wordmap

    else:
        with open(glove_vectors_file, "r", encoding="utf8") as glove:
            for line in glove:
                name, vector = tuple(line.split(" ", 1))
                glove_wordmap[name] = np.fromstring(vector, sep=" ")
        
        pickle.dump(glove_wordmap, open("Glove.p","wb+"))
        return glove_wordmap


def generate_new(unk):
    """ 
      Input: unknown word
      Output: generate a random word embedding from multi-nomial distribution and add to glove_wordmap
    """
    global glove_wordmap, Gmean, Gvar

    RS = np.random.RandomState()
    glove_wordmap[unk] = RS.multivariate_normal(Gmean,np.diag(Gvar))
    
    return glove_wordmap[unk]



#-------------
if __name__ == "__main__":

    # Step1: load data

    
    # Step2: load Glove and gather the distribution hyperparameters
    glove_wordmap = load_Glove("/Users/Mengying/Desktop/SI630 NLP/FinalProject/glove.6B/glove.6B.50d.txt")

    wvecs = []
    for item in glove_wordmap.items():
        wvecs.append(item[1])
    s = np.vstack(wvecs)
 
    Gvar = np.var(s,0) 
    Gmean = np.mean(s,0)




















