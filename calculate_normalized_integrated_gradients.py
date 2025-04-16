import pandas as pd
import numpy as np
from constant_variables import *
from scipy import stats
from glob import glob
import re
from tqdm import tqdm

dir_input="/home/CBBI/tsaih/Research/Model_Xnorm13995_new/IG/"
dir_output=dir_input+"GradientDistribution/"

from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

dir_oridata='/home/CBBI/tsaih/data/'

# Read output protein data
Ytrain=pd.read_csv(dir_oridata + "Y_data_187x6813_train.txt", sep='\t', index_col=[0]).T
Yval=pd.read_csv(dir_oridata + "Y_data_187x6813_val.txt", sep='\t', index_col=[0]).T
Ytest=pd.read_csv(dir_oridata + "Y_data_187x6813_test.txt", sep='\t', index_col=[0]).T
Y=pd.concat([Ytrain, Yval, Ytest])
del Ytrain, Yval, Ytest
Y.columns=Y.columns.str.replace('_p', '_P')

GeneNames=geneNames()

# Read self-gene vs protein data
Pairs=CorrRNAnPro()
Pairs.proName=Pairs.proName.str.upper()

# Read IG files
filenames = glob(dir_input + 'gradient_' + '*.txt')

# Find proteins that already had average IG calculated 
done_files= glob(dir_output + 'gradient_mean_HvsL_' + '*.txt')
done_pro=[]
for d in done_files:
    done_pro.append(re.search('gradient_mean_HvsL_' + '(.*).txt', d).group(1))

# Calculate normalized IG
proName=[]
geneName=[]

high_rank=[]
high_mean=[]
high_zscore=[]

low_rank=[]
low_mean=[]
low_zscore=[]

high_max=[]
high_min=[]
low_max=[]
low_min=[]

# Use ERALPHA as an example
for i in tqdm(range(len(filenames))):  
    #print(f)
    f=filenames[i]
    pro=re.search('gradient_' + '(.*).txt', f).group(1)
    print("{}: {}".format(list.index(filenames, f)+1, pro))
    
    if pro == 'ERALPHA':
        
        # if the protein had been done, extract the average IG results
        if pro in done_pro:
            print ('    done')
            df_result = pd.read_csv(dir_output+'gradient_mean_HvsL_' + pro+'.txt', sep="\t", index_col=[0])

        else:
            # Read IG matrix
            gradOri=pd.read_csv(f, sep="\t", header=None).T
            gradOri.index=sampleTrain()+sampleVal()+sampleTest()
            gradOri.columns=geneNames()
            Y_target=pd.DataFrame(Y[pro])
        
            # Average IG based on the top 10% highly/lowly expressed samples for each protein
            df_result=[]
            for t in [ 'high', 'low']:
                if t=='high':
                    targetSamples=list(Y_target[pro].nlargest(n=round(6813*0.1)).index)
                elif t=='low':
                    targetSamples = list(Y_target[pro].nsmallest(n=round(6813 * 0.1)).index)

                df=pd.DataFrame(gradOri.loc[targetSamples,:].mean(), columns=[t+'_ori'])
                df[t+'_zscore'] = stats.zscore(df[t+'_ori'])
                df[t+'_rank']=df[t+'_ori'].rank(ascending=False)
                df_result.append(df)
            df_result=pd.concat(df_result, axis=1)
            df_result=df_result.sort_values(by='high_rank', ascending=True)
            df_result.to_csv(dir_output + 'gradient_mean_HvsL_' + pro + '.txt', sep='\t', header=True, index=True)
    
        # Find pre-annotated self-gene 
        pairGene = Pairs[Pairs.proName == pro]['geneName'].tolist()[0]

        proName.append(pro)
        geneName.append(pairGene)
    
        high_rank.append(df_result.loc[pairGene]['high_rank'].tolist())
        high_mean.append(df_result.loc[pairGene]['high_ori'].tolist())
        high_zscore.append(df_result.loc[pairGene]['high_zscore'].tolist())

        low_rank.append(df_result.loc[pairGene]['low_rank'].tolist())
        low_mean.append(df_result.loc[pairGene]['low_ori'].tolist())
        low_zscore.append(df_result.loc[pairGene]['low_zscore'].tolist())

        high_max.append(df_result['high_ori'].max())
        high_min.append(df_result['high_ori'].min())
        low_max.append(df_result['low_ori'].max())
        low_min.append(df_result['high_ori'].min())

gradSummary=pd.DataFrame({
    'proName': proName,
    'geneName': geneName,
    'high_rank': high_rank,
    'high_mean': high_mean,
    'high_zscore':high_zscore,
    'low_rank': low_rank,
    'low_mean': low_mean,
    'low_zscore': low_zscore,
    'high_max': high_max,
    'high_min': high_min,
    'low_max': low_max,
    'low_min': low_min,
})

gradSummary.to_csv(dir_output + 'gradient_mean_summary'+ '.txt', sep='\t', header=True, index=True)
print('Finished')

df_result.head()

