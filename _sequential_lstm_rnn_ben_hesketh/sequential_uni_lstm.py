"""
18008836 - Ben Hesketh
Liverpool Hope University
______________________________________________________________________________________________________________________________________________________________________________
Installed Modules
-----------------

matplotlib-3.5.2
pandas-1.5.0
    numpy-1.22.1
    pytz-2021.3
    python-dateutil-2.8.2
    six-1.16.0
scikit-learn-1.0.2
    joblib-1.1.0
    threadpoolctl-3.0.0
    scipy-1.7.3
    numpy-1.22.1
opencv-python-4.6.0.66
pygame-2.1.2 < -- for music mixer
tensorflow-2.9.1-cp310-cp310-win_amd64.whl
    keras-2.10.0
pydot-3.0.7
    pydotplus-2.0.2
    graphviz-0.20.1
    <other dependancies...>

______________________________________________________________________________________________________________________________________________________________________________
"""
print("\nWelcome! Importing Libraries, please wait a moment...\n")

#LIBRARY IMPORTS______________________________________________________________________________________________________________________________________________________________
import numpy as np
#import seaborn as sns # <-- only if heatmaps are used.
import pandas as pd
import cv2
import tensorflow as tf # <-- only if using gpu.
import os # <-- to save weights to a directory easily.

#from keras import optimizers  <-- conflict between TensorFlow Keras and Keras API.
from pathlib import Path # <-- needed for finding files from user inputted filenames.
from pygame import mixer # for music mixer
from keras.utils.vis_utils import plot_model
#_____________________________________________________________________________________________________________________________________________________________________________


#CV2 & PYGAME MIXER MUSIC INIT________________________________________________________________________________________________________________________________________________
startImage = cv2.imread('C:/Users/benhe/Documents/.MastersDissertation/Program/images_cv2/starter.png')
trainImage = cv2.imread('C:/Users/benhe/Documents/.MastersDissertation/Program/images_cv2/training.png')

mixer.init() # mixer must be initialised!
mixer.music.load('C:/Users/benhe/Documents/.MastersDissertation/Program/music/elevator.mp3')
mixer.music.set_volume(0.8)

#_____________________________________________________________________________________________________________________________________________________________________________


#TRAIN WITH 20% DATASET_______________________________________________________________________________________________________________________________________________________
def begin_training():
    
    #Structured in 3 main parts (Data preperation, Keras, Evaluation).

    #The Dataset from generated report by Department for Transport - roadtraffic.dft.gov.uk
    dataset_train = pd.read_excel(datasetFile, sheet_name = '2018to2021')# full set
    training_set = dataset_train.iloc[:, 1:2].values #: is all the rows (first value), (first column is actually 0) <- CHANGE COLUMNS FOR CERTAIN YEARS?
    
    #Scale after dataset is loaded in (removes possible outliers, filters out impossible day numbers)
    sc = MinMaxScaler(feature_range = (0, 1)) #squish it down so its between 1 and 0.
    training_set_scaled = sc.fit_transform(training_set)

    #Structure with timesteps and a output from a range
    x_train = [] #Source of data from dataset.
    y_train = [] #Answers known so far.
    for i in range(60,2703): #total of 60 arrays (process all variables * 60 since theres a time sequence). 9569 lines in dataset - 60 = 9536.
        x_train.append(training_set_scaled[i-60:i, 0]) #-60 is actually to 0 (60 '- 60', range starting from 0 does not count. We want to know what 60 is.
        y_train.append(training_set_scaled[i, 0]) #filled with arrays from 0 to 59 on x array, 60 to 1258 on y array.
    x_train, y_train = np.array(x_train), np.array(y_train)#convert back to NumPy array.
    # two arrays, one filled with arrays 0 to 59, and one array which is just a value for 60 (timestep).

    #Reshape to format data correctly
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))#NumPy sees these as layers, requiring values (branches). Result is a new leaf (final value).

    #Keras modules
    from tensorflow.keras.models import Sequential #Data is sequential

    #Begin our recurrent neural network (RNN)
    bens_keras_rnn_model = Sequential()#Sequential model is used.

    #Three layers are the dense layer, the LSTM layer, and the dropout layer. 
    from tensorflow import keras
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout #Drops out data not needed.
    from tensorflow.keras.layers import Dense
    
    #NETWORK LAYERS_________________________________________________________________________________________________________________________________________________________________
    """
    The LSTM layers combined with dropout regularisation
    Each LSTM layer includes regularisation from the dropout layer
    This removes unneeded data before going through the network.

    Units are the positive integer and it is the dimensionality of the space of the output.
    This is inputted into the next layer.
    Data is sequenced data so 'return_sequences' must be set to true.
    Shape inputted must be assigned. NumPy sees shapes as layers. NumPy is required by Keras for this.
    Shape does not need to be known by other layers besides the first one. It is automatically understood.

    IMPORTANT -
    Dropout layer is vital to avoid providing too much information so the network does not predict anything. that is not in that realm.
    Dense layer is important to bring all of the sequence data from each layer's output as a single output for an answer at the end.
    Keras is used because it is not automatic to create layers with one line but it does provide flexibility and options of how layers
    interface and how data is inputted.
    """

    bens_keras_rnn_model.add(LSTM(units = 128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    bens_keras_rnn_model.add(Dropout(0.2)) #20 percent or the neurons are switched off. The neurons chosen are random each layer.

    bens_keras_rnn_model.add(LSTM(units = 128, activation = 'relu', return_sequences = True))
    bens_keras_rnn_model.add(Dropout(0.15))

    bens_keras_rnn_model.add(LSTM(units = 128, activation = 'relu', return_sequences = True))
    bens_keras_rnn_model.add(Dropout(0.25))

    bens_keras_rnn_model.add(LSTM(units = 128))
    bens_keras_rnn_model.add(Dropout(0.2))

    #ValueError: Found array with dim 3. Estimator expected <= 2 <-- caused by final layer having return_sequences = True. Don't do that.

    #final output 'Dense' layer. Used to bring all down to one output for an answer, not a sequence. 
    bens_keras_rnn_model.add(Dense(units = 1))
    #_______________________________________________________________________________________________________________________________________________________________________________

    #COMPILER WITH CUSTOM EPOCHS VALUE SET BY USER__________________________________________________________________________________________________________________________________

    checkpoint_path = "uni_lstm_training_fosa_18_21_rerun/cp.ckpt"
    bens_keras_rnn_model.load_weights(checkpoint_path)
    bens_keras_rnn_model.compile('adam', 'mean_squared_error', metrics = ['accuracy'])
    
    # during training, the loss is what ther loss is based on, how bad the error is.
    # the adam optimiser is used for the its different equations used.

    tf.keras.utils.plot_model(bens_keras_rnn_model, to_file='diagram_uni_gen.png', show_shapes=True, show_layer_names=True)

    print(bens_keras_rnn_model.summary())
    """
    epochs - how many times the model goes over the whole dataset provided (each rows with timestep of 60).
    """

    #SAVE WEIGHTS AT EACH EPOCH INTERVAL AND END____________________________________________________________________________________________________________________________________
    save_weight_path = "uni_lstm_training_fosa_18_21_rerun/cp.ckpt"
    save_weight_dir = os.path.dirname(save_weight_path)

    # Saving the weights of the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = save_weight_path, save_weights_only=True, verbose = 1)
    
    bens_keras_rnn_model.fit(x_train, y_train, epochs = setEpochs, batch_size = setBatch, callbacks = [cp_callback])# hyperparameters are inputted values from user from called functions.

    #_______________________________________________________________________________________________________________________________________________________________________________
#___________________________________________________________________________________________________________________________________________________________________________________

#CONACT FIRST TRAINING WITH FULL DATASET TO TEST PREDCITION ACCURACY________________________________________________________________________________________________________________
#def begin_concattraining(): <-- not sepearated yet.
    dataset_test = pd.read_excel('fatalOrSerious2010to2019.xlsx', sheet_name = '2020to2021') #First 20% as a test to see the rest of the missing 80% predicted this time. #1139 - 60
    real_set = dataset_test.iloc[:, 1:2].values

#concat them together for final plot figure.
#the end of the dataset (train one) is part of the data going in.
    dataset_total = pd.concat((dataset_train['FOSA'], dataset_test['FOSA']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values #to compare the two, keeping 60 timesteps in mind.
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 1076): #1625 - 60 = 1565
        X_test.append(inputs[i-60:i, 0]) #is actually 0 to 59! i, 0 is the open column.
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    predicted_accidents = bens_keras_rnn_model.predict(X_test)
    predicted_accidents = sc.inverse_transform(predicted_accidents) #MinMaxScaler(feature_range = (0, 1)). Is inversed to see the actual accidents numbers, not a float between 0 and 1.

    #Since they put together, it will be much faster the second time for network to train.

    plt.plot(real_set, color ='purple', label = 'Real North West Recorded Accidents ')
    plt.plot(predicted_accidents, color = 'green', label = 'Predicted 2020 North West Accidents')
    
    plt.title('2020 to 2021 North West Accident Predictions')
    plt.xlabel('Accident')
    plt.ylabel('Accident Fatal or Serious Adjusted')
    plt.legend()
    
    mixer.music.stop()
    plt.show()


#ASK FOR EPOCHS VALUE WITH RETRY LOOPER_____________________________________________________________________________________________________________________________________________
def ask_epochs():
    global setEpochs #global variable since begin_training() requires this value.
    print("Recommended is 100! Increase to improve prediction accuracy affected by dropout after each pass.\n")
    try:
        setEpochs = int(input("How many times would you like the model to go over the whole dataset provided? Please enter number: ")) #must be an integer like 100
        print("The epochs value is now: ", setEpochs)
        ask_batch()
    except ValueError:
        setEpochs = 100 # default value
        print("This is not an acceptable epochs value! It can only be a whole number.. (Recommended: 100.\n")
        ask_epochs()
#____________________________________________________________________________________________________________________________________________________________________________________


#ASK FOR BATCH SIZE WITH RETRY LOOPER________________________________________________________________________________________________________________________________________________        
def ask_batch():
    global setBatch
    print("\nBatch size is recommended as 32! Increase to speed up trainng but use more resource.\n")
    try:
        setBatch = int(input("What would you like to set the batch size to? Please enter a number: "))
        print("The batch size value is now: ", setBatch)
        print("\nBeginning training, this may take some time...")
        cv2.imshow('training', trainImage)
        mixer.music.play(-1) #-1 means indefinite loop
        cv2.waitKey(2000) #2 second pause.
        cv2.destroyWindow('training') #destroys the window showing image
        begin_training()     
    except ValueError:
        setBatch = 32
        print("This is not an acceptable batch size for the network! It can only be a whole number.. (Recommended: 32.\n")
        ask_batch()
#____________________________________________________________________________________________________________________________________________________________________________________
        

#ASK FOR 20% TRAINING DATASET WITH RETRY LOOPER______________________________________________________________________________________________________________________________________
def ask_dataset():
    global datasetFile #global variable since begin_training() requires this value.
    print("For example: 'fatalOrSerious2010to2019.xlsx'") #Full 100% dataset to train on.
    datasetFile = input("Enter the filename of the first dataset that will train the network (requires the extension): ")

    my_file = Path(datasetFile)
    if my_file.is_file(): # file exists
        ask_epochs()
    else: #file was not found..
        print("ERROR: Dataset file (.xlsx) was not found! Please try again. >_<\n")
        ask_dataset()
#_____________________________________________________________________________________________________________________________________________________________________________________


#BEGIN SCRIPT_________________________________________________________________________________________________________________________________________________________________________
cv2.imshow('begin', startImage)
cv2.waitKey(2000) #2 second pause.
cv2.destroyWindow('begin')#destroys the window showing image
print("Number of Graphic Processing Units found by CUDA: ", len(tf.config.list_physical_devices('GPU')))
print("\nRoad Accident Predictions in UK regions using a LSTM_RNN network.\nBy Ben Hesketh.\n")

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

print("\nSuccessfully imported sklearn module and pyplot.\n")
ask_dataset()

#_____________________________________________________________________________________________________________________________________________________________________________________

