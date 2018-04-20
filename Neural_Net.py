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
from string import punctuation

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
    sentence = ''.join(c for c in sentence if c not in punctuation)
    #punctuation: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
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

    # Dropout probabilities. 
    input_p, output_p = 0.5, 0.5

    # How many questions we train on at a time.
    batch_size = 128

    # Number of passes in episodic memory. 
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


    
    #### Dynamic Memory Networks ####

########
# Input Module
#######
    # Context: A [batch_size, maximum_context_length, word_vectorization_dimensions] tensor 
    # that contains all the context information.
    context = tf.placeholder(tf.float32, [None, None, D], "context")  
    context_placeholder = context # I use context as a variable name later on

    # input_sentence_endings: A [batch_size, maximum_sentence_count, 2] tensor that 
    # contains the locations of the ends of sentences. 
    input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")

    # recurrent_cell_size: the number of hidden units in recurrent layers.
    input_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

    # input_p: The probability of maintaining a specific hidden input unit.
    # Likewise, output_p is the probability of maintaining a specific hidden output unit.
    gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru, input_p, output_p)

    # dynamic_rnn also returns the final internal state. We don't need that, and can
    # ignore the corresponding output (_). 
    input_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, context, dtype=tf.float32, scope = "input_module")

    # cs: the facts gathered from the context.
    cs = tf.gather_nd(input_module_outputs, input_sentence_endings)
    # to use every word as a fact, useful for tasks with one-sentence contexts
    s = input_module_outputs





########
# Question Module
#######
    # query: A [batch_size, maximum_question_length, word_vectorization_dimensions] tensor 
    #  that contains all of the questions.

    query = tf.placeholder(tf.float32, [None, None, D], "query")

    # input_query_lengths: A [batch_size, 2] tensor that contains question length information. 
    # input_query_lengths[:,1] has the actual lengths; input_query_lengths[:,0] is a simple range() 
    # so that it plays nice with gather_nd.
    input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")

    question_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, query, dtype=tf.float32, 
                                               scope = tf.VariableScope(True, "input_module"))  # returns(output, state)

    # q: the question states. A [batch_size, recurrent_cell_size] tensor.
    q = tf.gather_nd(question_module_outputs, input_query_lengths)



########
# Episodic Memory Module
#######

    # make sure the current memory (i.e. the question vector) is broadcasted along the facts dimension
    size = tf.stack([tf.constant(1),tf.shape(cs)[1], tf.constant(1)])
    re_q = tf.tile(tf.reshape(q,[-1,1,recurrent_cell_size]),size)  # use -1 to flatten


    # Final output for attention, needs to be 1 in order to create a mask
    output_size = 1 

    # Weights and biases
    attend_init = tf.random_normal_initializer(stddev=0.1)
    w_1 = tf.get_variable("attend_w1", [1,recurrent_cell_size*7, recurrent_cell_size], 
                          tf.float32, initializer = attend_init)
    w_2 = tf.get_variable("attend_w2", [1,recurrent_cell_size, output_size], 
                          tf.float32, initializer = attend_init)

    b_1 = tf.get_variable("attend_b1", [1, recurrent_cell_size], 
                          tf.float32, initializer = attend_init)
    b_2 = tf.get_variable("attend_b2", [1, output_size], 
                          tf.float32, initializer = attend_init)

    # Regulate all the weights and biases
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_2))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_2))

    def attention(c, mem, existing_facts):
        """
        Custom attention mechanism.
        c: A [batch_size, maximum_sentence_count, recurrent_cell_size] tensor 
            that contains all the facts from the contexts.
        mem: A [batch_size, maximum_sentence_count, recurrent_cell_size] tensor that 
            contains the current memory. It should be the same memory for all facts for accurate results.
        existing_facts: A [batch_size, maximum_sentence_count, 1] tensor that 
            acts as a binary mask for which facts exist and which do not.

        """
        with tf.variable_scope("attending") as scope:
            # attending: The metrics by which we decide what to attend to.
            attending = tf.concat([c, mem, re_q, c * re_q,  c * mem, (c-re_q)**2, (c-mem)**2], 2)

            # m1: First layer of multiplied weights for the feed-forward network. 
            #     We tile the weights in order to manually broadcast, since tf.matmul does not
            #     automatically broadcast batch matrix multiplication as of TensorFlow 1.2.
            m1 = tf.matmul(attending * existing_facts, 
                           tf.tile(w_1, tf.stack([tf.shape(attending)[0],1,1]))) * existing_facts
            # bias_1: A masked version of the first feed-forward layer's bias
            #     over only existing facts.

            bias_1 = b_1 * existing_facts

            # tnhan: First nonlinearity. In the original paper, this is a tanh nonlinearity; 
            #        choosing relu was a design choice intended to avoid issues with 
            #        low gradient magnitude when the tanh returned values close to 1 or -1. 
            tnhan = tf.nn.relu(m1 + bias_1)

            # m2: Second layer of multiplied weights for the feed-forward network. 
            #     Still tiling weights for the same reason described in m1's comments.
            m2 = tf.matmul(tnhan, tf.tile(w_2, tf.stack([tf.shape(attending)[0],1,1])))

            # bias_2: A masked version of the second feed-forward layer's bias.
            bias_2 = b_2 * existing_facts

            # norm_m2: A normalized version of the second layer of weights, which is used 
            #     to help make sure the softmax nonlinearity doesn't saturate.
            norm_m2 = tf.nn.l2_normalize(m2 + bias_2, -1)

            # softmaxable: A hack in order to use sparse_softmax on an otherwise dense tensor. 
            #     We make norm_m2 a sparse tensor, then make it dense again after the operation.
            softmax_idx = tf.where(tf.not_equal(norm_m2, 0))[:,:-1]
            softmax_gather = tf.gather_nd(norm_m2[...,0], softmax_idx)
            softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
            softmaxable = tf.SparseTensor(softmax_idx, softmax_gather, softmax_shape)
            return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)),-1)

    # facts_0s: a [batch_size, max_facts_length, 1] tensor 
    #     whose values are 1 if the corresponding fact exists and 0 if not.
    facts_0s = tf.cast(tf.count_nonzero(input_sentence_endings[:,:,-1:],-1,keep_dims=True),tf.float32)


    with tf.variable_scope("Episodes") as scope:
        attention_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

        # memory: A list of all tensors that are the (current or past) memory state 
        #   of the attention mechanism.
        memory = [q]

        # attends: A list of all tensors that represent what the network attends to.
        attends = []
        for a in range(passes):
            # attention mask
            attend_to = attention(cs, tf.tile(tf.reshape(memory[-1],[-1,1,recurrent_cell_size]),size),
                                  facts_0s)

            # Inverse attention mask, for what's retained in the state.
            retain = 1-attend_to

            # GRU pass over the facts, according to the attention mask.
            while_valid_index = (lambda state, index: index < tf.shape(cs)[1])
            update_state = (lambda state, index: (attend_to[:,index,:] * 
                                                     attention_gru(cs[:,index,:], state)[0] + 
                                                     retain[:,index,:] * state))
            # start loop with most recent memory and at the first index
            memory.append(tuple(tf.while_loop(while_valid_index,
                              (lambda state, index: (update_state(state,index),index+1)),
                               loop_vars = [memory[-1], 0]))[0]) 

            attends.append(attend_to)

            # Reuse variables so the GRU pass uses the same variables every pass.
            scope.reuse_variables()


########
# Answer Module
#######











