from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from sciPENN.sciPENN_API import sciPENN_API
import scanpy as sc
from time import time
from sklearn.metrics import mean_squared_error

def Pearsonresult(true, pred):
    pearsonc = []
    pearsonp = []
    mse_test=[]
    for j in range(true.shape[1]):
        pearsonc.append(pearsonr(true.iloc[:, j], pred.iloc[:, j])[0])
        pearsonp.append(pearsonr(true.iloc[:, j], pred.iloc[:, j])[1])
        mse_test.append(mean_squared_error(true.iloc[:, j], pred.iloc[:, j]))

    CorMatrix = pd.DataFrame({
        'rppa': true.columns,
        'pearsonr_test': pearsonc,
        'pearsonp_test': pearsonp,
        'mse_test': mse_test

    })
    return CorMatrix

dir_input='/home/CBBI/tsaih/data/10Xdata/sciPENN_data/'
dir_input2='/home/CBBI/tsaih/data/10Xdata/sciPENN_data/ProcessedData/'
dir_output='/home/CBBI/tsaih/Research_SingleCell/CrossValidation/time0/'
from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

#index (n=53364)
index_train=pd.read_csv(dir_input2+'PBMCIndex_train_10xCV.txt', sep='\t')
index_val=pd.read_csv(dir_input2+'PBMCIndex_val_10xCV.txt', sep='\t')
index_test=pd.read_csv(dir_input2+'PBMCIndex_test_10xCV.txt', sep='\t')

index=pd.read_csv(dir_input2+'CellIndex_time0.txt', sep='\t')

adata_protein = sc.read_h5ad(dir_input+'pbmc/pbmc_protein.h5ad')
adata_gene = sc.read_h5ad(dir_input+'pbmc/pbmc_gene.h5ad')

# Original PBMC data
print(adata_gene.shape)
print(adata_protein.shape)

tableCNN = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])

# Run one round as an example
for i in range(1):
    print('Round' + str(i+1))

    cell_train=index_train.iloc[:, i].tolist() + index_val.iloc[:, i].tolist()
    cell_test = index_test.iloc[:, i].tolist()
    print(len(cell_train))
    print(len(cell_test))

    cellID_train=index.iloc[cell_train]['CellID'].tolist()
    cellID_test = index.iloc[cell_test]['CellID'].tolist()

    adata_gene_train = adata_gene[cellID_train]
    adata_gene_test = adata_gene[cellID_test]

    adata_protein_train = adata_protein[cellID_train]
    adata_protein_test = adata_protein[cellID_test]


    sciPENN = sciPENN_API([adata_gene_train], [adata_protein_train], adata_gene_test, train_batchkeys = ['donor'])

    start = time()
    sciPENN.train(n_epochs = 10000, ES_max = 12, decay_max = 6,
             decay_step = 0.1, lr = 10**(-3), weights_dir = "weights_dir/pbmc_time0_10xCV_"+str(i))
    imputed_test = sciPENN.predict()
    print(time() - start)

    adata_protein_test.X = adata_protein_test.X.toarray()
    adata_protein_test.layers["raw"] = adata_protein_test.X

    adata_protein_test = adata_protein_test[imputed_test.obs.index]

    sc.pp.normalize_total(adata_protein_test)
    sc.pp.log1p(adata_protein_test)
    sc.pp.filter_genes(adata_protein_test, min_counts = 1)


    #sciPENN predicted results

    adata_protein_test.layers['imputed'] = imputed_test.X
    adata_protein_test.layers.update(imputed_test.layers)

    adata_protein_test.obs['sample'] = [1] * adata_protein_test.shape[0]
    patients = np.unique(adata_protein_test.obs['sample'].values)

    for patient in patients:
        indices = [x == patient for x in adata_protein_test.obs['sample']]
        sub_adata = adata_protein_test[indices]

        sc.pp.scale(sub_adata)
        adata_protein_test[indices] = sub_adata.X


    df_pred=pd.DataFrame(adata_protein_test.layers["imputed"])
    df_true=adata_protein_test.to_df()

    Result=Pearsonresult(df_true, df_pred)
    tableCNN.loc[i, 'mse_test'] = Result['mse_test'].mean()
    tableCNN.loc[i, 'cor_test'] = Result['pearsonr_test'].mean()

    Result.to_csv(dir_output + 'Table_PearsonResult_10xCV_' + str(i) + '_sciPENN.txt', sep='\t')
tableCNN.to_csv(dir_output + 'Table_SummaryResult_10xCV_' + str(1) + '_sciPENN.txt', sep='\t')
