Python 3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.

= RESTART: C:\Users\benhe\Documents\.MastersDissertation\Program\_sequential_lstm_rnn_ben_hesketh\sequential_bi_lstm.py

Welcome! Importing Libraries, please wait a moment...

pygame 2.1.2 (SDL 2.0.18, Python 3.10.6)
Hello from the pygame community. https://www.pygame.org/contribute.html
Number of Graphic Processing Units found by CUDA:  1

Road Accident Predictions in UK regions using a LSTM_RNN network.
By Ben Hesketh.


Successfully imported sklearn module and pyplot.

For example: 'fatalOrSerious2010to2019.xlsx'
Enter the filename of the first dataset that will train the network (requires the extension): fatalOrSerious2010to2019.xlsx
Recommended is 100! Increase to improve prediction accuracy affected by dropout after each pass.

How many times would you like the model to go over the whole dataset provided? Please enter number: 5
The epochs value is now:  5

Batch size is recommended as 32! Increase to speed up trainng but use more resource.

What would you like to set the batch size to? Please enter a number: 32
The batch size value is now:  32

Beginning training, this may take some time...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 60, 128)           66560     
                                                                 
 dropout (Dropout)           (None, 60, 128)           0         
                                                                 
 bidirectional (Bidirectiona  (None, 60, 128)          98816     
 l)                                                              
                                                                 
 dropout_1 (Dropout)         (None, 60, 128)           0         
                                                                 
 bidirectional_1 (Bidirectio  (None, 60, 128)          98816     
 nal)                                                            
                                                                 
 dropout_2 (Dropout)         (None, 60, 128)           0         
                                                                 
 bidirectional_2 (Bidirectio  (None, 128)              98816     
 nal)                                                            
                                                                 
 dropout_3 (Dropout)         (None, 128)               0         
                                                                 
 dense (Dense)               (None, 1)                 129       
                                                                 
=================================================================
Total params: 363,137
Trainable params: 363,137
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/5

Epoch 1: saving model to bi_lstm_training_fosa_18_21\cp.ckpt

Epoch 2/5

Epoch 2: saving model to bi_lstm_training_fosa_18_21\cp.ckpt

Epoch 3/5

Epoch 3: saving model to bi_lstm_training_fosa_18_21\cp.ckpt

Epoch 4/5

Epoch 4: saving model to bi_lstm_training_fosa_18_21\cp.ckpt

Epoch 5/5

Epoch 5: saving model to bi_lstm_training_fosa_18_21\cp.ckpt

Epoch 1/5

Epoch 2/5

Epoch 3/5

Epoch 4/5

Epoch 5/5


Warning (from warnings module):
  File "C:\Users\benhe\Documents\.MastersDissertation\Program\_sequential_lstm_rnn_ben_hesketh\sequential_bi_lstm.py", line 171
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values #to compare the two, keeping 60 timesteps in mind.
FutureWarning: The behavior of `series[i:j]` with an integer-dtype index is deprecated. In a future version, this will be treated as *label-based* indexing, consistent with e.g. `series[i]` lookups. To retain the old behavior, use `series.iloc[i:j]`. To get the future behavior, use `series.loc[i:j]`.
