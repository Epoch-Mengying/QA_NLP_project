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
import tensorflow as tf

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


# -------------
# Prepare data for Neural Network
# -------------


def generate_new(unk):
    """ 
      Input: unknown word
      Output: generate a random word embedding from multi-nomial distribution and add to glove_wordmap
    """
    global glove_wordmap, Gmean, Gvar

    RS = np.random.RandomState()
    glove_wordmap[unk] = RS.multivariate_normal(Gmean,np.diag(Gvar))
    
    return glove_wordmap[unk]




def sentence2sequence(sentence):
    """
    - Turns an input paragraph into an (m,d) matrix, 
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.

      Input: sentence, string
      Output: embedding vector stacked row by row, in a matrix. And corresponding words(list)

    """
    tokens = sentence.strip('"(),-').lower().split(" ")  # The characters to be removed from beginning or end of the string.
    rows = []
    words = []

    
    #Greedy search for tokens: if it did not find the whole word in glove, then truncate tail to find
    for token in tokens:
        i = len(token)
        while len(token) > 0:
            word = token[:i]                 # shallow copy
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)               # reset to 0 
                continue
            else:
                i = i-1
            if i == 0:
                # word OOV
                # https://arxiv.org/pdf/1611.01436.pdf
                rows.append(generate_new(token))
                words.append(token)
                break
    return np.array(rows), words











#-------------
if __name__ == "__main__":

## Step1: load data

    
## Step2: load Glove and gather the distribution hyperparameters
    glove_wordmap = load_Glove("/Users/Mengying/Desktop/SI630 NLP/FinalProject/glove.6B/glove.6B.50d.txt")

    wvecs = []
    for item in glove_wordmap.items():
        wvecs.append(item[1])
    s = np.vstack(wvecs)
 
    Gvar = np.var(s,0) 
    Gmean = np.mean(s,0)



## Step3: Tensor Flow
    tf.reset_default_graph()   # clear the graph, so that we can always run again if we need to change anything


    #### Hyperparameters ####

    # The number of dimensions used to store data passed between recurrent layers in the network.
    recurrent_cell_size = 128

    # The number of dimensions in our word vectorizations.
    D = 50 

    # How quickly the network learns. Too high, and we may run into numeric instability 
    # or other issues.
    learning_rate = 0.005

    # Dropout probabilities. For a description of dropout and what these probabilities are, 
    # see Entailment with TensorFlow.
    input_p, output_p = 0.5, 0.5

    # How many questions we train on at a time.
    batch_size = 128

    # Number of passes in episodic memory. We'll get to this later.
    passes = 4

    # Feed Forward layer sizes: the number of dimensions used to store data passed from feed-forward layers.
    ff_hidden_size = 256

    weight_decay = 0.00000001
    # The strength of our regularization. Increase to encourage sparsity in episodic memory, 
    # but makes training slower. Don't make this larger than leraning_rate.

    training_iterations_count = 400000
    # How many questions the network trains on each time it is trained. 
    # Some questions are counted multiple times.

    display_step = 100
    # How many iterations of training occur before each validation check.






















