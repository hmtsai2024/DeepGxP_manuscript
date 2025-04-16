
from import_data import load_data
from keras.models import load_model
from constant_variables import *
from sklearn.preprocessing import StandardScaler

#load model
dir_model='/home/CBBI/tsaih/Research/Model_Xnorm13995_new/'
tmpName = '1Layer1D_1LayerDense_conv1024_kernelstride50_dense512_batch254_normX_biasFalse'
model_saved=load_model(dir_model + tmpName + '_model_1'+ '.h5')

dir_output=dir_model+ 'Prediction_TCGAunknownsamples/'
from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

dir_input='/home/CBBI/tsaih/data/'

# Find samples that were not used in DeepGxP training process
df_exp=pd.read_csv(dir_input + 'EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena', sep='\t', index_col=[0])
df_exp.columns=df_exp.columns.str.replace('-', '.')
sampleNames=sampleTrain() + sampleVal() + sampleTest()
geneNames = pd.read_csv(dir_input + 'X_data_batch_13995x6813_val.txt', sep='\t', index_col=[0]).index.tolist()
sampleNames_notyet=list(set(df_exp.columns)-set(sampleNames)) #4256

# Organize mismatched genenames
geneNames_intersect=list(set(geneNames) & set(df_exp.index)) #13976
geneNames_notinlist=list(set(geneNames) - set(geneNames_intersect)) #19
print(len(geneNames_intersect))
print(len(geneNames_notinlist))

geneNames_new=['MARCH1', 'MARCH2', 'MARCH3', 'MARCH4', 'MARCH6', 'MARCH8', 'MARCH9',
                     'SEPT1', 'SEPT10', 'SEPT3', 'SEPT4', 'SEPT5', 'SEPT6', 'SEPT7', 'SEPT8', 'SEPT9', 'SEPT11']
geneNames_old=['1-Mar', '2-Mar', '3-Mar', '4-Mar', '6-Mar', '8-Mar', '9-Mar',
                     '1-Sep', '10-Sep', '3-Sep', '4-Sep', '5-Sep', '6-Sep', '7-Sep', '8-Sep', '9-Sep', '11-Sep']
geneNameNew2Old=dict(zip(geneNames_new, geneNames_old))

#rename a list of names to another list names
df_exp.rename(index=geneNameNew2Old, inplace=True)

# Step 1: Convert index to a list
index_list = list(df_exp.index)
# Step 2: Find positions of duplicated names
res = [i for i, name in enumerate(df_exp.index) if name == 'SLC35E2']
# Step 3: Rename specific entries by position
index_list[res[0]] = 'SLC35E2_1'
index_list[res[1]] = 'SLC35E2_2'
# Step 4: Assign the modified list back to the DataFrame index
df_exp.index = index_list

df_exp2=df_exp.loc[geneNames, sampleNames_notyet]
df_exp2.to_csv(dir_input + 'X_data_batch_13995x4256_unknowdata' + '.txt', sep='\t', index=True)

# Z-transform RNA data on unknown samples (apply same transformation learned from training data to the unknown)
# Read training data
inputX, _, _, _ = load_data(dir_input + 'X_data_batch_13995x6813_train' + '.txt')
# Read unknown protein samples
input, input_labels, sample_names, gene_names = load_data(dir_input + 'X_data_batch_13995x4256_unknowdata' + '.txt')

sample_names.remove('sample')
scaler = StandardScaler()
inputX_df = scaler.fit_transform(inputX)
input_df = pd.DataFrame(scaler.transform(input), index=sample_names, columns=gene_names)
input_df.T.to_csv(dir_input + 'X_data_batch_13995x4256_unknowdata_z.txt', sep='\t', header=True, index=True)

# Read z-transformed unknown data form file then predict protein abundance
input, input_labels, sample_names, gene_names = load_data(dir_input + 'X_data_batch_13995x4256_unknowdata_z' + '.txt')
x = input.reshape(input.shape[0], 13995, 1)

#prediction
pred = pd.DataFrame(model_saved.predict(x, verbose=0))
sample_names.remove("")
pred.index=sample_names
pred.columns=proNames()

pred.T.to_csv(dir_output + 'ProPredictfromtrainedmodel_TCGA4256_Xunnorm.txt', sep='\t', header=True, index=True)

