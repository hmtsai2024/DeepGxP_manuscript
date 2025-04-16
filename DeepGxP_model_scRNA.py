import scanpy as sc
from time import time
import pandas as pd
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pickle
#tf version 2.x
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=100, inter_op_parallelism_threads=100)))

def CNNmodel(x_train, x_val, x_test, y_train, y_val, y_test, dim):
    batch_size = 256
    filters1 = 256
    kernel = 100
    stride = 100
    dense1 = 1024
    actC = 'linear'
    actD = 'relu'

    x_train = x_train.to_numpy().reshape(x_train.shape[0], dim, 1)
    x_val = x_val.to_numpy().reshape(x_val.shape[0], dim, 1)
    x_test = x_test.to_numpy().reshape(x_test.shape[0], dim, 1)

    K.clear_session()
    start = time()
    layers = []
    main_input = Input(shape=(dim, 1), name='main_input')
    layers.append(
        Conv1D(filters=filters1, activation=actC, kernel_size=kernel, strides=stride, use_bias=True)(main_input))
    layers.append(MaxPooling1D(pool_size=2, strides=None, padding='same')(layers[-1]))
    layers.append(Flatten()(layers[-1]))
    layers.append(Dense(dense1, activation=actD, use_bias=True)(layers[-1]))
    output = Dense(y_train.shape[1], activation='linear', use_bias=True)(layers[-1])
    model = Model(inputs=main_input, outputs=output)

    model.compile(optimizer='adam', loss='mse')
    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min',
                            restore_best_weights=True)

    history_2 = model.fit(x_train, y_train,
                          validation_data=(x_val, y_val),
                          batch_size=batch_size,
                          epochs=500,
                          shuffle=True,
                          callbacks=[history],
                          verbose=2)
    print((time() - start) / 60)
    cost = model.evaluate(x_test, y_test, verbose=0)
    params = model.count_params()

    mse_train = history_2.history['loss'][history.stopped_epoch]
    mse_val = history_2.history['val_loss'][history.stopped_epoch]
    mse_test = cost
    epochs = history.stopped_epoch

    # print('Predicting....')
    pred_train = pd.DataFrame(model.predict(x_train, batch_size=batch_size, verbose=0))
    pred_val = pd.DataFrame(model.predict(x_val, batch_size=batch_size, verbose=0))
    pred_test = pd.DataFrame(model.predict(x_test, batch_size=batch_size, verbose=0))
    return mse_train, mse_val, mse_test, pred_train, pred_val, pred_test, epochs, model

def MSEresult(y_train, y_val, y_test, pred_train, pred_val, pred_test):
    mse_train= mean_squared_error(y_train, pred_train)
    mse_val = mean_squared_error(y_val, pred_val)
    mse_test = mean_squared_error(y_test, pred_test)
    return mse_train, mse_val, mse_test

def Pearsonresult(y_train, y_val, y_test, pred_train, pred_val, pred_test):
    pearsonr_train = []
    pearsonp_train = []
    pearsonr_val = []
    pearsonp_val = []
    pearsonr_test = []
    pearsonp_test = []

    for j in range(y_train.shape[1]):
        pearsonr_train.append(pearsonr(y_train.iloc[:, j], pred_train.iloc[:, j])[0])
        pearsonp_train.append(pearsonr(y_train.iloc[:, j], pred_train.iloc[:, j])[1])
        pearsonr_val.append(pearsonr(y_val.iloc[:, j], pred_val.iloc[:, j])[0])
        pearsonp_val.append(pearsonr(y_val.iloc[:, j], pred_val.iloc[:, j])[1])
        pearsonr_test.append(pearsonr(y_test.iloc[:, j], pred_test.iloc[:, j])[0])
        pearsonp_test.append(pearsonr(y_test.iloc[:, j], pred_test.iloc[:, j])[1])

    CorMatrix = pd.DataFrame({
        'rppa': y_train.columns,
        'pearsonr_train': pearsonr_train,
        'pearsonp_train': pearsonp_train,
        'pearsonr_val': pearsonr_val,
        'pearsonp_val': pearsonp_val,
        'pearsonr_test': pearsonr_test,
        'pearsonp_test': pearsonp_test
    })
    return CorMatrix

def save_weight_to_pickle(model, file_name):
    print('saving weights')
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)

dir_input='/home/CBBI/tsaih/data/10Xdata/sciPENN_data/ProcessedData/'
dir_output='/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS/'
from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

index=pd.read_csv(dir_input+'CellIndex_time0.txt', sep='\t')
index_train=index[index.DataSet=='train']['CellID'].tolist()
index_val=index[index.DataSet=='val']['CellID'].tolist()
index_test=index[index.DataSet=='test']['CellID'].tolist()

Ymethod='protein_norm_renorm'
Xmethod='magic'

print ('Reading Y data.....')
Y = sc.read_h5ad(dir_input + 'pbmc_time0_' + Ymethod + '.h5ad')
df_Y = Y.to_df()
df_Y=df_Y.drop(columns=['Rag-IgG2c','Rat-IgG1-1', 'Rat-IgG1-2', 'Rat-IgG2b'])
del Y

# Define per-fold score containers
tableCNN = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])
print ('Reading X data.....')
X = sc.read_h5ad(dir_input + 'pbmc_time0_' + Xmethod + '.h5ad')
df_X = X.to_df()
del X

print('Selecting samples.....')
df_X_train = df_X.loc[index_train, ]
df_X_val = df_X.loc[index_val, ]
df_X_test = df_X.loc[index_test, ]

df_Y_train = df_Y.loc[index_train, :]
df_Y_val = df_Y.loc[index_val, :]
df_Y_test = df_Y.loc[index_test, :]

for i in range(1):
    print('-----Round '+ str(i+1))
    tmpName='CNN_X_'+Xmethod+ '_Y_'+ Ymethod +'_NOgeneset'
    resultCNN = CNNmodel(df_X_train, df_X_val, df_X_test, df_Y_train, df_Y_val, df_Y_test, df_X_train.shape[1])
    pearsonCNN = Pearsonresult(df_Y_train, df_Y_val, df_Y_test, resultCNN[3], resultCNN[4], resultCNN[5])
    mseCNN = MSEresult(df_Y_train, df_Y_val, df_Y_test, resultCNN[3], resultCNN[4], resultCNN[5])

    tableCNN.loc[i, 'mse_train'] = resultCNN[0]
    tableCNN.loc[i, 'mse_val'] = resultCNN[1]
    tableCNN.loc[i, 'mse_test'] = resultCNN[2]
    tableCNN.loc[i, 'mse_train2'] = mseCNN[0]
    tableCNN.loc[i, 'mse_val2'] = mseCNN[1]
    tableCNN.loc[i, 'mse_test2'] = mseCNN[2]
    tableCNN.loc[i, 'cor_train'] = pearsonCNN['pearsonr_train'].mean()
    tableCNN.loc[i, 'cor_val'] = pearsonCNN['pearsonr_val'].mean()
    tableCNN.loc[i, 'cor_test'] = pearsonCNN['pearsonr_test'].mean()

    pearsonCNN.to_csv(dir_output + tmpName + '_corr_results_' + str(i + 1) + '.txt', sep='\t', index=False)
    resultCNN[3].to_csv(dir_output + tmpName + '_predfromtrain_' + str(i + 1) + '.txt', sep='\t')
    resultCNN[4].to_csv(dir_output + tmpName + '_predfromval_' + str(i + 1) + '.txt', sep='\t')
    resultCNN[5].to_csv(dir_output + tmpName + '_predfromtest_' + str(i + 1) + '.txt', sep='\t')
    resultCNN[7].save(dir_output + tmpName + '_model_' + str(i + 1) + '.h5')
    save_weight_to_pickle(resultCNN[7], dir_output + tmpName + '_model_' + str(i + 1) + '.pickle')
    K.clear_session()

tableCNN.to_csv(dir_output + tmpName + '_result_summary.txt', sep='\t')


