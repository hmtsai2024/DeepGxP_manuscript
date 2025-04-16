from keras import backend as K
from keras.callbacks import EarlyStopping
from keras import models
from keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Activation
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
import xgboost
from sklearn.multioutput import MultiOutputRegressor
from constant_variables import *

def CNNmodel(x_train, x_val, x_test, y_train, y_val, y_test):
    batch_size = 256
    filters1 = 1024
    kernel = 50
    stride = 50
    dense1 = 512
    act = 'linear'
    
    # Reshape input data to 1D structure
    x_train_np = x_train.to_numpy().reshape(x_train.shape[0], 13995, 1)
    x_val_np = x_val.to_numpy().reshape(x_val.shape[0], 13995, 1)
    x_test_np = x_test.to_numpy().reshape(x_test.shape[0], 13995, 1)

    K.clear_session()

    model = models.Sequential()
    # ---first CNN layer---
    model.add(Conv1D(input_shape=(13995, 1), filters=filters1, kernel_size=kernel, strides=stride, use_bias=True,
                     padding='same'))
    model.add(Activation(act))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='same'))  # padding='valid'
    # ---flatten layer---
    model.add(Flatten())
    # ---dirst dense layer---
    model.add(Dense(dense1, use_bias=True))
    model.add(Activation(act))
    # ---output layer---
    model.add(Dense(y_train.shape[1], use_bias=True))
    model.add(Activation(act))
    #compile model
    model.compile(optimizer='adam', loss='mse')
    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
    history_2 = model.fit(x_train_np, y_train,
                          validation_data=(x_val_np, y_val),
                          batch_size=batch_size,
                          epochs=500,
                          shuffle=True,
                          callbacks=[history],
                          verbose=0)
    cost = model.evaluate(x_test_np, y_test, verbose=0)

    mse_train = history_2.history['loss'][history.stopped_epoch]
    mse_val = history_2.history['val_loss'][history.stopped_epoch]
    mse_test = cost
    epochs = history.stopped_epoch

    pred_train = pd.DataFrame(model.predict(x_train_np, batch_size=batch_size, verbose=0))
    pred_val = pd.DataFrame(model.predict(x_val_np, batch_size=batch_size, verbose=0))
    pred_test = pd.DataFrame(model.predict(x_test_np, batch_size=batch_size, verbose=0))
    return mse_train, mse_val, mse_test, pred_train, pred_val, pred_test, epochs

def CCAmodel(x_train, x_val, x_test, y_train, y_val, y_test):
    cca = CCA(scale=False)
    cca.fit(x_train, y_train)
    X_c, Y_c = cca.transform(x_train, y_train)
    pred_train = pd.DataFrame(cca.predict(x_train))
    pred_val = pd.DataFrame(cca.predict(x_val))
    pred_test = pd.DataFrame(cca.predict(x_test))
    return pred_train, pred_val, pred_test

def mLRmodel(x_train, x_val, x_test, y_train, y_val, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    pred_train = pd.DataFrame(lr.predict(x_train))
    pred_val = pd.DataFrame(lr.predict(x_val))
    pred_test = pd.DataFrame(lr.predict(x_test))
    return pred_train, pred_val,  pred_test

def XGBmodel(x_train, x_val, x_test, y_train, y_val, y_test):
    xgb = MultiOutputRegressor(xgboost.XGBRegressor(max_depth=10, nthread=80))
    xgb.fit(x_train, y_train)
    pred_train = pd.DataFrame(xgb.predict(x_train))
    pred_val = pd.DataFrame(xgb.predict(x_val))
    pred_test = pd.DataFrame(xgb.predict(x_test))
    return pred_train, pred_val, pred_test

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

dir_input = "/home/CBBI/tsaih/data/"
dir_output='/home/CBBI/tsaih/Research/diffMLmethods_comparison/CrossValidation/'

from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

# Read input gene data
ori_exp1=pd.read_csv(dir_input + 'X_data_batch_13995x6813_train_z.txt', sep="\t", index_col=[0])
ori_exp2=pd.read_csv(dir_input + 'X_data_batch_13995x6813_val_z.txt', sep="\t", index_col=[0])
ori_exp3=pd.read_csv(dir_input + 'X_data_batch_13995x6813_test_z.txt', sep="\t", index_col=[0])
ori_exp=pd.merge(pd.merge(ori_exp1, ori_exp2, left_index=True, right_index=True), ori_exp3, left_index=True, right_index=True)
del ori_exp1, ori_exp2, ori_exp3

# Read output protein data
ori_pro1=pd.read_csv(dir_input + 'Y_data_187x6813_train.txt', sep="\t", index_col=[0])
ori_pro2=pd.read_csv(dir_input + 'Y_data_187x6813_val.txt', sep="\t", index_col=[0])
ori_pro3=pd.read_csv(dir_input + 'Y_data_187x6813_test.txt', sep="\t", index_col=[0])
ori_pro=pd.merge(pd.merge(ori_pro1, ori_pro2, left_index=True, right_index=True), ori_pro3, left_index=True, right_index=True)
del ori_pro1, ori_pro2, ori_pro3
ori_pro.index=ori_pro.index.str.upper()
ori_exp_T=ori_exp.T
ori_pro_T=ori_pro.T

# Read self-gene vs protein correlation data
pairing=CorrRNAnPro()

# Read sample index of pre-split 80/10/10 traning/validation/testing sets
index_train=pd.read_csv(dir_input+'RandomIndex_train_10xCV.txt', sep='\t')
index_val=pd.read_csv(dir_input+'RandomIndex_val_10xCV.txt', sep='\t')
index_test=pd.read_csv(dir_input+'RandomIndex_test_10xCV.txt', sep='\t')

# Define per-fold score containers
tableCNN = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])
tablemLR = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])
tableLR = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])
tableCCA = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])
tableXGB = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])

# Use one round as example
for i in range(1):
    print('Round' + str(i+1))

    # input_data
    x_train = ori_exp_T.iloc[index_train.iloc[:, i]]
    x_val = ori_exp_T.iloc[index_val.iloc[:, i]]
    x_test = ori_exp_T.iloc[index_test.iloc[:, i]]
    # output_data
    y_train = ori_pro_T.iloc[index_train.iloc[:, i]]
    y_val = ori_pro_T.iloc[index_val.iloc[:, i]]
    y_test = ori_pro_T.iloc[index_test.iloc[:, i]]

    # CNN
    print('.......CNN')
    resultCNN = CNNmodel(x_train, x_val, x_test, y_train, y_val, y_test)
    pearsonCNN=Pearsonresult(y_train, y_val, y_test, resultCNN[3], resultCNN[4], resultCNN[5])

    tableCNN.loc[i, 'mse_train']=resultCNN[0]
    tableCNN.loc[i, 'mse_val'] = resultCNN[1]
    tableCNN.loc[i, 'mse_test'] = resultCNN[2]
    tableCNN.loc[i, 'cor_train']= pearsonCNN['pearsonr_train'].mean()
    tableCNN.loc[i, 'cor_val'] = pearsonCNN['pearsonr_val'].mean()
    tableCNN.loc[i, 'cor_test'] = pearsonCNN['pearsonr_test'].mean()

    #CCA
    print('.......CCA')
    resultCCA = CCAmodel(x_train, x_val, x_test, y_train, y_val, y_test)
    pearsonCCA = Pearsonresult(y_train, y_val, y_test, resultCCA[0], resultCCA[1], resultCCA[2])
    mse_train, mse_val,  mse_test=MSEresult(y_train, y_val,  y_test, resultCCA[0], resultCCA[1], resultCCA[2])
    tableCCA.loc[i, 'mse_train'] = mse_train
    tableCCA.loc[i, 'mse_val'] = mse_val
    tableCCA.loc[i, 'mse_test'] = mse_test
    tableCCA.loc[i, 'cor_train'] = pearsonCCA['pearsonr_train'].mean()
    tableCCA.loc[i, 'cor_val'] = pearsonCCA['pearsonr_val'].mean()
    tableCCA.loc[i, 'cor_test'] = pearsonCCA['pearsonr_test'].mean()
    
    del resultCCA, pearsonCCA, mse_train, mse_val, mse_test
    #XGBoosting
    print('.......XGBoost')
    resultxgb = XGBmodel(x_train, x_val, x_test, y_train, y_val, y_test)
    pearsonxgb = Pearsonresult(y_train, y_val, y_test, resultxgb[0], resultxgb[1], resultxgb[2])
    mse_train, mse_val, mse_test = MSEresult(y_train, y_val, y_test, resultxgb[0], resultxgb[1], resultxgb[2])
    tableXGB.loc[i, 'mse_train'] = mse_train
    tableXGB.loc[i, 'mse_val'] = mse_val
    tableXGB.loc[i, 'mse_test'] = mse_test
    tableXGB.loc[i, 'cor_train'] = pearsonxgb['pearsonr_train'].mean()
    tableXGB.loc[i, 'cor_val'] = pearsonxgb['pearsonr_val'].mean()
    tableXGB.loc[i, 'cor_test'] = pearsonxgb['pearsonr_test'].mean()
    del resultxgb, pearsonxgb, mse_train, mse_val, mse_test
    
    # mLR
    print('.......mLR')
    resultmLR = mLRmodel(x_train, x_val, x_test, y_train, y_val, y_test)
    pearsonmLR = Pearsonresult(y_train, y_val, y_test, resultmLR[0], resultmLR[1], resultmLR[2])
    mse_train, mse_val, mse_test = MSEresult(y_train, y_val, y_test, resultmLR[0], resultmLR[1], resultmLR[2])
    tablemLR.loc[i, 'mse_train'] = mse_train
    tablemLR.loc[i, 'mse_val'] = mse_val
    tablemLR.loc[i, 'mse_test'] = mse_test
    tablemLR.loc[i, 'cor_train'] = pearsonmLR['pearsonr_train'].mean()
    tablemLR.loc[i, 'cor_val'] = pearsonmLR['pearsonr_val'].mean()
    tablemLR.loc[i, 'cor_test'] = pearsonmLR['pearsonr_test'].mean()
    del resultmLR, pearsonmLR, mse_train, mse_val, mse_test
    
    # LR
    print('......self-gene')
    result_train = []
    result_val = []
    result_test = []
    for j in range(y_train.shape[1]):
        pro = y_train.columns[j]
        gene = pairing[pairing['proName'] == pro]['geneName'].tolist()[0]
        if gene == 'C12orf5':
            gene='C12ORF5'
    
        # train data
        df_pro = pd.DataFrame(y_train[pro])
        df_gene = pd.DataFrame(x_train[gene])
    
        # val
        df_pro_val = pd.DataFrame(y_val[pro])
        df_gene_val= pd.DataFrame(x_val[gene])
    
        # test
        df_pro_test = pd.DataFrame(y_test[pro])
        df_gene_test = pd.DataFrame(x_test[gene])
    
        LRresult = mLRmodel(df_gene, df_gene_val, df_gene_test, df_pro, df_pro_val, df_pro_test)
        result_train.append(LRresult[0])
        result_val.append(LRresult[1])
        result_test.append(LRresult[2])
    
    
    result_train_pd = pd.concat(result_train, axis=1)
    result_val_pd = pd.concat(result_val, axis=1)
    result_test_pd = pd.concat(result_test, axis=1)
    
    pearsonLR = Pearsonresult(y_train, y_val, y_test, result_train_pd, result_val_pd, result_test_pd)
    mse_train, mse_val, mse_test = MSEresult(y_train, y_val, y_test, result_train_pd, result_val_pd, result_test_pd)
    
    tableLR.loc[i, 'mse_train'] = mse_train
    tableLR.loc[i, 'mse_val'] = mse_val
    tableLR.loc[i, 'mse_test'] = mse_test
    tableLR.loc[i, 'cor_train'] = pearsonLR['pearsonr_train'].mean()
    tableLR.loc[i, 'cor_val'] = pearsonLR['pearsonr_val'].mean()
    tableLR.loc[i, 'cor_test'] = pearsonLR['pearsonr_test'].mean()
    del result_train, result_val, result_test, result_train_pd, result_val_pd, result_test_pd, mse_train, mse_val, mse_test

tableCNN.to_csv(dir_output + 'Table_result_10foldCrossValidation_' + str(1) + '_CNN2.txt', sep='\t')
tableCCA.to_csv(dir_output + 'Table_result_10foldCrossValidation_' + str(1) + '_CCA.txt', sep='\t')
tablemLR.to_csv(dir_output + 'Table_result_10foldCrossValidation_' + str(1) + '_mLR.txt', sep='\t')
tableLR.to_csv(dir_output + 'Table_result_10foldCrossValidation_' + str(1) + '_LR.txt', sep='\t')
tableXGB.to_csv(dir_output + 'Table_result_10foldCrossValidation_' + str(1) + '_XGB.txt', sep='\t')

