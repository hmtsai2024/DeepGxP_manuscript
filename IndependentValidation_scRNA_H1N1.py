
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from keras.models import load_model

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

dir_input='/home/CBBI/tsaih/Research_SingleCell/IndependentValidation/DatafromSciPenn/H1N1/'
#read gene data
adata_gene_test = sc.read(dir_input+"gene_data.mtx").T
adata_gene_test.var.index = pd.read_csv(dir_input+"gene_names.txt", index_col = 0).iloc[:, 0]
adata_gene_test.obs = pd.read_csv(dir_input+"meta_data.txt", sep = ',', index_col = 0)
#read protein data
adata_protein_test = sc.read(dir_input+"protein_data.mtx").T
adata_protein_test.var.index = [x[:len(x) - 5] for x in pd.read_csv(dir_input+"protein_names.txt", index_col = 0).iloc[:,0]]
adata_protein_test.obs = pd.read_csv(dir_input+"meta_data.txt", sep = ',', index_col = 0)

sc.write(dir_input+"gene_data", adata_gene_test, file_format='h5seurat')
adata_gene_test.write(dir_input+"gene_data"+".h5ad")
adata_protein_test.write(dir_input+"protein_data"+".h5ad")

#---done magic and protein normalization in R------

dir_input='/home/CBBI/tsaih/data/10Xdata/sciPENN_data/ProcessedData/'
dir_model='/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS/'
dir_data='/home/CBBI/tsaih/Research_SingleCell/IndependentValidation/DatafromSciPenn/H1N1/'
dir_output='/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS_IndependentValidation/'
from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

# Read training data
Xmethod='magic'
Ymethod='protein_norm_renorm'

Y = sc.read_h5ad(dir_input + 'pbmc_time0_' + Ymethod + '.h5ad')
df_Y = Y.to_df()
df_Y=df_Y.drop(columns=['Rag-IgG2c','Rat-IgG1-1', 'Rat-IgG1-2', 'Rat-IgG2b'])
del Y

X = sc.read_h5ad(dir_input + 'pbmc_time0_' + Xmethod + '.h5ad')
df_X = X.to_df()
del X

print(df_Y.shape)
print(df_X.shape)

# Load DeepGxP model
tmpName= 'CNN_X_magic_Y_protein_norm_renorm_NOgeneset_model_3.h5'
model_saved = load_model(dir_model + tmpName)

# Read independent dataset (H1N1) RNA data
dataInd=sc.read_h5ad(dir_data + 'gene_data_magic.h5ad')
df_dataInd = dataInd.to_df()
del dataInd

# Read Independent datset (H1N1) protein data
Y_true = sc.read_h5ad(dir_data + 'protein_data_normalized_margin2.h5ad')
Y_true = Y_true_margin2.to_df()

# Fill missing genes with zeros (RNA)
df_target=df_dataInd.reindex(columns=df_X.columns.tolist(), fill_value=0)

print('Predicting....')
test_pred = pd.DataFrame(model_saved.predict(df_target, verbose=0), index=df_target.index,
                                             columns=df_Y.columns)

test_pred.to_csv(dir_output + 'H1N1_imputation_magic_'+ 'DeepG2P_SC_predicted' + '.txt', sep='\t')

# Ensure row indices (Cell order) are the same
if not test_pred.index.equals(Y_true.index):
    raise ValueError("Row indices of A and B do not match")

# Find common columns
common_columns = test_pred.columns.intersection(Y_true.columns)

df_pred=test_pred
df_true=Y_true

results2=[]
# Calculate Pearson correlation and p-value for each common column
for col in common_columns:
    corr, pvalue = pearsonr(df_pred[col], df_true[col])
    rmse = np.sqrt(mean_squared_error(df_true[col], df_pred[col]))
    results2.append([col, corr, pvalue, rmse])

# Create a DataFrame from the results
results_df2 = pd.DataFrame(results2, columns=['ProteinName', 'Correlation', 'P-value', 'RMSE'])
results_df2=results_df2.sort_values(by='Correlation', ascending=False)
results_df2.to_csv(dir_output + 'Table_SummaryResult_H1N1_imputation_magic_'+ 'DeepG2P_SC_predicted' + '.txt', sep='\t')