import numpy as np
import tensorflow as tf

X=[1,2]
state=[0.0,0.0]
w_cell_state=np.asarray([[0.1,0.2],[0.3,0.4]])
w_cell_input=np.asarray([0.5,0.6])
b_cell=np.asarray([0.1,-0.1])
#用于定义输出的全连接层参数
w_output=np.asarray([[1.0],[2.0]])
b_output=0.1
for i in range(len(X)):
    before_activation=np.dot(state,w_cell_state)+X[i]*w_cell_input+b_cell
    state=np.tanh(before_activation)
    final_output=np.dot(state,w_output)+b_output
    print('before_activation:',before_activation)
    print('state:',state)
    print('output:',final_output)

'''
以上为numpy实现，
下面是tensorflow实现
'''
lstm=rnn_cell.BasicLSTMCell(lstm_hidden_size)
state=lstm.zero_state(batch_size,tf.float32)
loss=0.0
for i in range(num_steps):
    if i>0:tf.get_variable_scope().reuse_variables()
    lstm_output,state=lstm(current_input,state)
    final_output=fully_connectd(lstm_output)
    loss+=calc_loss(final_output,expected_output)

'''
深层循环网络
'''
lstm=rnn_cell.BasicLSTMCell(lsem_size)
stacked_lstm=rnn_cell.MultiRNNCell([lstm]*number_of_layers)#每个时刻的前向传播过程有多少层
state=stacked_lstm.zero_state(batch_size,tf.float32)
for i in range(len(num_steps)):
    if i>0:tf.get_variable_scope().reuse_variables()
    stacked_lstm_output,state=stacked_lstm(current_input,state)
    final_output=fully_connectd(stacked_lstm_output)
    loss+=calc_loss(final_output,expected_output)
