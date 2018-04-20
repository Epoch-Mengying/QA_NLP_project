import tensorflow as tf
import numpy as np


#-------
# Answer module
#-------

# answer_word: [batch_size, max_answer_length, D = 50]
answer_word = tf.placeholder(tf.int32, [None, None, D], "answer_word")
# answer_word_length
answer_word_length = tf.placeholder(tf.int32, [1], "answer_word_length")

## Answer RNN defined here
answer_gru = tf.contrib.rnn.GRUCell(D)
answer_gru_drop = tf.contrib.rnn.DropoutWrapper(answer_gru, input_p, output_p) # our Cell

# Initial hidden state:
answer_init_state = tf.concat([memory[-1],q],-1) # initial memory state for Answer GRU

# Input:
q_reshaped = 
tf.layers.dense(input = )
q_expanded = q... # q: 128*1*256 --> 128*20*256
I = tf.concat([answer_word, q_expanded], -1)



answer_output,answer_final_state = tf.nn.dynamic_rnn(answer_gru_drop, I,
   initial_state= answer_init_state, dtype = tf.float32, scope = tf.VariableScope(True, "Answer_module"))






# Test

if answers[i] == np.zeros(50)

n=1
tf.restore()

for i in range(20):
answer_sets = sess.run([answer_output], feed_dict={data}) #128*

for i in range(batch_size):
    for j in range(max_answer_input):
        one_word = answer_sets[i,j,:]  # 50*1
        
        
    


  







