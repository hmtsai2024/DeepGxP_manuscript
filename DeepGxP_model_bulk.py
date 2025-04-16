import numpy as np
from keras import backend as K
from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Activation
from import_data import load_data
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy.stats import pearsonr
import os

dir_input='/home/CBBI/tsaih/data/'
dir_output='/home/CBBI/tsaih/Research/DeepGxP_model/'

if not os.path.exists(dir_output):
    os.mkdir(dir_output)

tmpName = 'DeepGxP'

def save_weight_to_pickle(model, file_name):
    print('saving weights')
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)

print('Reading input data......')
input_train, input_labels_train, sample_names_train, gene_names_train = load_data(
    dir_input+"X_data_batch_13995x6813_train_z.txt")
input_val, input_labels_val, sample_names_val, gene_names_val = load_data(
    dir_input+"X_data_batch_13995x6813_val_z.txt")
input_test, input_labels_test, sample_names_test, gene_names_test = load_data(
    dir_input+"X_data_batch_13995x6813_test_z.txt")

print('Reading output data......')
input_pro, data_labels_pro, sample_names_pro, gene_names_pro = load_data(
    dir_input+"Y_data_187x6813_train.txt")
input_pro_val, data_labels_pro_val, sample_names_pro_val, gene_names_pro_val = load_data(
    dir_input+"Y_data_187x6813_val.txt")
input_pro_test, data_labels_pro_test, sample_names_pro_test, gene_names_pro_test = load_data(
    dir_input+"Y_data_187x6813_test.txt")

# Reshape data to 1D shape
x_train = input_train.reshape(input_train.shape[0], 13995, 1)
x_val = input_val.reshape(input_val.shape[0], 13995, 1)
x_test = input_test.reshape(input_test.shape[0], 13995, 1)

y_train=pd.DataFrame(input_pro)
y_val=pd.DataFrame(input_pro_val)
y_test=pd.DataFrame(input_pro_test)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

K.clear_session()

# DeepGxP model
batch_size = 256
filters1 =1024
kernel = 50
stride = 50
dense1 = 512
act = 'linear'

mse = np.empty((5, 3))
params=[]

# Run one round as an example
for i in range(1):
    print('Round' + str(i+1))
    model= models.Sequential()
    #---first CNN layer---
    model.add(Conv1D(input_shape=(13995,1), filters=filters1, kernel_size= kernel, strides= stride, use_bias=True))
    model.add(Activation(act))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid'))
    #---flatten layer---
    model.add(Flatten())
    #---dirst dense layer---
    model.add(Dense(dense1, use_bias=True))
    model.add(Activation(act))
    #---output layer---
    model.add(Dense(input_pro.shape[1], use_bias=True))
    model.add(Activation(act))
    model.compile(optimizer='adam', loss='mse')
    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
    #model.compile(optimizer='adam', loss='mse', batch_size=batch_size)
    history_2=model.fit(x_train, input_pro,
            validation_data=(x_val, input_pro_val),
            batch_size=batch_size,
            epochs=500,
            shuffle=True,
            callbacks=[history],
            verbose=2)
    cost =model.evaluate(x_test, input_pro_test, verbose=0)
    params=model.count_params()
    print(model.summary())

    print('Drawing model loss......')
    plt.plot(history_2.history['loss'])
    plt.plot(history_2.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(dir_output+ tmpName + str(i + 1) + 'loss.png', format='png', bbox_inches="tight")
    plt.close()

    mse[i, 0] = history_2.history['loss'][history.stopped_epoch]
    mse[i, 1] = history_2.history['val_loss'][history.stopped_epoch]
    mse[i, 2] = cost

    #print(datetime.datetime.now() - time_start)

    # Predict from training/validation/testing sets
    print('Predicting....')
    train_pred = pd.DataFrame(model.predict(x_train, batch_size=batch_size, verbose=0))
    val_pred = pd.DataFrame(model.predict(x_val, batch_size=batch_size, verbose=0))
    test_pred = pd.DataFrame(model.predict(x_test, batch_size=batch_size, verbose=0))
    
    # Compute Pearson's correlation for each protein in training/validation/testing sets
    pearsonr_train = []
    pearsonp_train = []
    pearsonr_val = []
    pearsonp_val = []
    pearsonr_test = []
    pearsonp_test = []

    for j in range(input_pro.shape[1]):
        pearsonr_train.append(pearsonr(y_train.iloc[:, j], train_pred.iloc[:, j])[0])
        pearsonp_train.append(pearsonr(y_train.iloc[:, j], train_pred.iloc[:, j])[1])
        pearsonr_val.append(pearsonr(y_val.iloc[:, j], val_pred.iloc[:, j])[0])
        pearsonp_val.append(pearsonr(y_val.iloc[:, j], val_pred.iloc[:, j])[1])
        pearsonr_test.append(pearsonr(y_test.iloc[:, j], test_pred.iloc[:, j])[0])
        pearsonp_test.append(pearsonr(y_test.iloc[:, j], test_pred.iloc[:, j])[1])

    CorMatrix = pd.DataFrame({
        'rppa': gene_names_pro,
        'pearsonr_train': pearsonr_train,
        'pearsonp_train': pearsonp_train,
        'pearsonr_val': pearsonr_val,
        'pearsonp_val': pearsonp_val,
        'pearsonr_test': pearsonr_test,
        'pearsonp_test': pearsonp_test
    })
    
    # Save correlation results
    CorMatrix.to_csv(dir_output + tmpName + '_corr_results_' + str(i + 1) + '.txt', sep='\t', index=False)
    
    # Save prediction values
    train_pred.to_csv(dir_output + tmpName + '_predfromtrain_' + str(i + 1) + '.txt', sep='\t')
    val_pred.to_csv(dir_output + tmpName + '_predfromval_' + str(i + 1) + '.txt', sep='\t')
    test_pred.to_csv(dir_output + tmpName + '_predfromtest_' + str(i + 1) + '.txt', sep='\t')
    
    # Save model
    model.save(dir_output + tmpName + '_model_' + str(i + 1) + '.h5')
    
    # Save model weights
    save_weight_to_pickle(model, dir_output + tmpName + '_model_' + str(i + 1) + '.pickle')
    K.clear_session()
np.savetxt(dir_output + tmpName + '_mse_summary.txt', mse, delimiter='\t', fmt='%.4f')

