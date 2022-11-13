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
import pandas as pd
import tensorflow as tf
import cv2
import os

from pygame import mixer # <- for music mixer
from pathlib import Path
from keras.utils.vis_utils import plot_model
#_____________________________________________________________________________________________________________________________________________________________________________

#CV2 & PYGAME MIXER MUSIC INIT________________________________________________________________________________________________________________________________________________
startImage = cv2.imread('C:/Users/benhe/Documents/.MastersDissertation/Program/images_cv2/starter.png')
trainImage = cv2.imread('C:/Users/benhe/Documents/.MastersDissertation/Program/images_cv2/training.png')

mixer.init() # mixer must be initialised!
mixer.music.load('C:/Users/benhe/Documents/.MastersDissertation/Program/music/elevator.mp3')
mixer.music.set_volume(0.8)
#_____________________________________________________________________________________________________________________________________________________________________________

#TRAINING WITH SPLIT DATASET__________________________________________________________________________________________________________________________________________________
def begin_training():
#ALL COLUMNS MUST MATCH DECIMAL PLACE FOR FLOAT VALUES. OR NAN_LOSS_ERROR (no values).
    dataset_train = pd.read_excel(datasetFile, usecols=['FOSA', 'Month','LA','Day','Road surface','Accident year'], sheet_name = '2016to2021') # <-- dataset is split with 'train_test_split' later. Give full DS.
    dataset_train.head()
    print(dataset_train)
    print(dataset_train.shape)
    dataset_train.info(verbose=True, memory_usage='deep')

#Memory issue solved - Pandas Dataframe becomes blank due to dataset being too large. usecols=[] did cause missing array, dtype error, but now fixed.

    def extract_ds(data):
        rc_out = data.pop(col1_user)
        rc_out = np.array(rc_out)
        
        la_out = data.pop(col2_user)
        la_out = np.array(la_out)

        return rc_out, la_out # output
        print(rc_out)
        print(la_out)

    # Split the data into train and test with 80 train / 20 test
    train_ds, test_ds = train_test_split(dataset_train, test_size=setTestSize, random_state = 0, shuffle = False, stratify = None) #76.9215169% is years unaffected by covid roughly. 100 - unaffected = 23.0784831%, rounded to 0.23

    # Getting the outputs of the train and test data 
    y_train = extract_ds(train_ds)
    y_test = extract_ds(test_ds)
    

    min_max=MinMaxScaler()

    X_train=min_max.fit_transform(train_ds)
    X_test=min_max.transform(test_ds)

    #NETWORK LAYERS_________________________________________________________________________________________________________________________________________________________________
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    #from tensorflow.keras.layers import LSTM
    #(Here we are using 2 hidden layers and one branched layer with 10 neurons each)

    #Input layer
    inp_layer = Input(shape=(4,),name='inp_layer')# number of columns - 2

    ##Hidden layers
    #LSTM_1 = LSTM(10, activation='tanh',recurrent_activation='sigmoid',name='LSTM_1')(inp_layer)
    Layer_1 = Dense(10, activation="relu",name='Layer_1')(inp_layer)
    Layer_2 = Dense(10, activation="relu",name='Layer_2')(Layer_1)
    
    ##Defining Branched layer
    Branched_layer_1 = Dense(15, activation="relu",name='Branched_layer_1')(Layer_2)

    Branched_layer_2 = Dense(15, activation="relu",name='Branched_layer_2')(Layer_2)
    
    ##Defining 2nd output layer
    la_out_layer = Dense(1, activation="linear",name='la_out_layer')(Branched_layer_1)

    rc_out_layer = Dense(1, activation="linear",name='rc_out_layer')(Branched_layer_2)

    ##Defining the model by specifying the input and output layers
    bens_functional_api_model = Model(inputs = inp_layer, outputs = [rc_out_layer,la_out_layer]) #<-- Heatmap from here?

    checkpoint_path = "api_training_fosa_LA_16_21/cp.ckpt"
    bens_functional_api_model.load_weights(checkpoint_path)
    
    #COMPILER WITH CUSTOM EPOCHS & BATCH VALUES SET BY USER_________________________________________________________________________________________________________________________
    bens_functional_api_model.compile(optimizer='adam', loss={'rc_out_layer':'mse','la_out_layer':'mse'})

    print(bens_functional_api_model.summary())
    tf.keras.utils.plot_model(bens_functional_api_model, to_file='diagram_gen_api_fosa_months.png', show_shapes=True, show_layer_names=True)

    #SAVE WEIGHTS AT EACH EPOCH INTERVAL AND END____________________________________________________________________________________________________________________________________
    save_weight_path = "api_training_fosa_LA_16_21/cp.ckpt"
    save_weight_dir = os.path.dirname(save_weight_path)

    # Saving the weights of the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = save_weight_path, save_weights_only=True, verbose = 1)

    bens_functional_api_model.fit(X_train,y_train, epochs = setEpochs, batch_size = setBatch, validation_data = (X_test,y_test), callbacks = [cp_callback])
    #_______________________________________________________________________________________________________________________________________________________________________________

    #PREDICTIONS____________________________________________________________________________________________________________________________________________________________________
    model_predictions = bens_functional_api_model.predict(X_test)
    predicted_rc = model_predictions[0]
    predicted_la = model_predictions[1]
    print('done..')
    mixer.music.stop() # <-- stops elevator music end of training.

    #PLOT SCATTER FIGURES____________________________________________________________________________________________________________________________________________________________
    plt.scatter(y_test[0],predicted_rc, c = 'red')
    plt.title('2016 to 2021 Accident Day')
    plt.xlabel('Actual FOSA')
    plt.ylabel('Predicted FOSA')
    plt.show()

    plt.scatter(y_test[1],predicted_la, c = 'blue') #y-test is the actual data for comparisons.
    plt.title('2016 to 2021 Reported Accidents in Local Authority')
    plt.xlabel('Actual Local Authority')
    plt.ylabel('Predicted Local Authority')
    plt.annotate('1.00 - Halton, 2.00 - Liverpool, 3.00 - St.Helens, 4.00 - Warrington, 5.00 - Wigan', xy=(0.05, 0.95), xycoords='axes fraction')
    plt.show()


    #PLOT PLOTTING FIGURES___________________________________________________________________________________________________________________________________________________________
    plt.plot(y_test[0], color = 'blue', label = 'Actual FOSA')
    plt.plot(predicted_rc, color = 'red', label = 'Predicted FOSA')
    plt.title('2016 to 2021')
    plt.xlabel('Actual FOSA')
    plt.ylabel('Predicted FOSA')
    plt.legend()
    plt.show()

    plt.plot(y_test[1], color = 'blue', label = 'Actual Local Authority')
    plt.plot(predicted_la, color = 'red', label = 'Predicted Local Authority')
    plt.title('2016 to 2021 Reported Accidents in Local Authority')
    plt.xlabel('Actual Local Authority')
    plt.ylabel('Predicted Local Authority')
    plt.legend()
    plt.annotate('1.00 - Halton, 2.00 - Liverpool, 3.00 - St.Helens, 4.00 - Warrington, 5.00 - Wigan', xy=(0.05, 0.95), xycoords='axes fraction')
    plt.show()
    
    # print accuracy scores?
#___________________________________________________________________________________________________________________________________________________________________________________


#ASK FOR COLUMN NAMES_______________________________________________________________________________________________________________________________________________________________
def ask_columns():
    global col1_user
    global col2_user
    print("\nWhat are the names of the two columns you wish to extract data from the dataset?\n")
    #try:
    col1_user = input("Please enter the first column name (case-sensitive): ")
    print("The first column is: ", col1_user)
    col2_user = input("Please enter the second column name (case-sensitive): ")
    print("The second column is: ", col2_user)
    ask_batch()
    #except ValueError:
        #print("\nTry again..\n")
        #ask_columns()


#ASK FOR EPOCHS VALUE WITH RETRY LOOPER_____________________________________________________________________________________________________________________________________________
def ask_epochs():
    global setEpochs #global variable since begin_training() requires this value.
    print("Recommended is 1000! Increase to improve prediction accuracy affected by dropout after each pass.\n")
    #try:
    setEpochs = int(input("How many times would you like the model to go over the whole dataset provided? Please enter number: ")) #must be an integer like 1000
    print("The epochs value is now: ", setEpochs)
    ask_testSize()
    #except ValueError:
        #setEpochs = 100 # default value
        #print("This is not an acceptable epochs value! It can only be a whole number.. (Recommended: 100.\n")
        #ask_epochs()
#____________________________________________________________________________________________________________________________________________________________________________________

def ask_testSize():
    global setTestSize #global variable since begin_training() requires this value.
    print("\nRecommended is 0.2 (20%)! The rest of the dataset will be the training data!\n")
    #try:
    setTestSize = float(input("How much of the chosen dataset would you like to have split as the test data (as a percentage)?: ")) #must be an float like 0.2
    if setTestSize >= 1:
        print("This is not an acceptable dataset test size value! It can only be a float value lower than 1.0 (Recommended: 0.2.\n")
        ask_testSize()
    if setTestSize <= 0:
        print("This is not an acceptable dataset test size value! It can only be a float value lower than 1.0 (Recommended: 0.2.\n")
        ask_testSize()
    else:
        print("Test data size is now: ", setTestSize)
        ask_columns()
    #except ValueError:
        #setTestSize = 0.2 # default value
        #print("\nSomething has gone wrong. Try again or restart the program.\n")
        #ask_testSize()
     # turned this into if statements combined with try loop incase user inputs anything besides float values.
#____________________________________________________________________________________________________________________________________________________________________________________



#ASK FOR BATCH SIZE WITH RETRY LOOPER________________________________________________________________________________________________________________________________________________        
def ask_batch():
    global setBatch
    print("\nBatch size is recommended as 128! Increase to speed up trainng but use more resource.\n")
    #try:
    setBatch = int(input("What would you like to set the batch size to? Please enter a number: "))
    print("The batch size value is now: ", setBatch)
    print("\nBeginning training, this may take some time...")
    cv2.imshow('training', trainImage)
    mixer.music.play(-1) #-1 means indefinite loop
    cv2.waitKey(2000) #2 second pause.
    cv2.destroyWindow('training') #destroys the window showing image
    begin_training()   
    #except ValueError:
        #setBatch = 128
        #print("This is not an acceptable batch size for the network! It can only be a whole number.. (Recommended: 32.\n")
        #ask_batch()
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
print("\nRoad Accident Predictions in UK regions using a Functional API network.\nBy Ben Hesketh.\n\nImporting sklearn modules, please wait..\n")

import matplotlib.pyplot as plt
#import seaborn as sns #<-- Heatmap practise

from sklearn.model_selection import train_test_split # <-- Needed to split the dataset (80% train, 20% test)
from sklearn.preprocessing import MinMaxScaler
from pylab import *

print("\nSuccessfully imported sklearn modules.\n")

print("Number of Graphic Processing Units found by CUDA: ", len(tf.config.list_physical_devices('GPU')))
ask_dataset()
#_____________________________________________________________________________________________________________________________________________________________________________________

