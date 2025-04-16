import pandas as pd
import re
from glob import glob
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

dir_input='/home/CBBI/tsaih/Research_SingleCell/CrossValidation/time0/'
dir_input2='/home/CBBI/tsaih/Research_SingleCell/CrossValidation/time0_DeepG2Pstructure/'

filenames = glob(dir_input + 'Table_SummaryResult' + '*.txt')

# Performance of DeepGxP and sciPENN
data=[]

sciPENN_files = glob(dir_input + 'Table_SummaryResult_10xCV_1_sciPENN.txt')
for f in sciPENN_files:
    #print(f)
    df=pd.read_csv(f, sep='\t', index_col=[0])
    methods=f.split("_")[-1]
    methods=re.search('' + '(.*).txt', methods).group(1)

    if methods=='sciPENN':
        method='sciPENN'
        Xmethod='sciPENN'
        Ymethod='counts'
        GS = 'no'
    df['method'] = method
    df['X'] = Xmethod
    df['Y'] = Ymethod
    df['GS'] = GS
    df['round'] = list(range(1, 10+1))
    data.append(df)


deepgxp_files = glob(dir_input2 + 'Table_SummaryResult*.txt')
for f in deepgxp_files:
    #print(f)
    df=pd.read_csv(f, sep='\t', index_col=[0])
    methods=f.split("_")[-1]
    methods=re.search('' + '(.*).txt', methods).group(1)

    Xmethod = re.search('_X_' + '(.*)Y', f).group(1)
    gsYN = f.split("_")[-2]
    if gsYN == 'NOgenesetLayer':
        GS = 'no'
        Ymethod = re.search('Yprotein_' + '(.*)_NOgeneset', f).group(1)
    else:
        GS = re.search('genesetLayer_' + '(.*).ADT224', f).group(1)
        Ymethod = re.search('Yprotein_' + '(.*)_geneset', f).group(1)

    df['method'] = 'DeepGxP'
    df['X'] = Xmethod
    df['Y'] = Ymethod
    df['GS'] = GS
    df['round'] = list(range(1, 10+1))
    data.append(df)

# Combine all
data=pd.concat(data)

# Univariate linear regression model (use self-gene)
LR=pd.read_csv('/home/CBBI/tsaih/Research_SingleCell/CrossValidation/time0_LR/'+
               'Table_SummaryResult_10xCV_X_countsYprotein_norm_renorm_LR_ADT224.txt', sep='\t', index_col=[0])
LR['method'] = 'LR'
LR['X'] = 'LR'
LR['Y'] = 'norm_renorm'
LR['GS'] = 'no'
LR['round'] = list(range(1, 10+1))
data=data.append(LR)

# Fig 5B: Boxplot of model performance
metrics='cor_test'

if metrics=='cor_test':
    metric='Pearson_Correlation'
elif metrics=='mse_test':
    metric='MSE'

ax=sns.barplot(x='X', y=metrics, data=data[data['GS']=='no'],
               order=['LR','sciPENN', 'magic'], ci="sd")
ax=sns.stripplot(x='X', y=metrics, data=data[data['GS']=='no'],
                 order=['LR','sciPENN', 'magic'],
                 color='black', size=3, alpha=0.9)
plt.ylim(0.2, 0.6)
ax.set_xlabel('', fontsize=18)
ax.set_ylabel(metric+'\n(predicted vs. true)', fontsize=18)
ax.tick_params(labelsize=15)
plt.tight_layout()
# plt.savefig(dir_input + "Barplot_diffX_NOGS_" + metric + "wLR_higherY.png", format='png', dpi=600)
# plt.savefig(dir_input + "Barplot_diffX_NOGS_" + metric + "wLR_higherY.pdf", format='pdf', dpi=600)
# plt.close()

# Fig 5C, Distribution of Pearson correlation between true protein vs self-gene and DeepGxP-predicted protein abundance
# Read Pearson correlation between self-gene expression and protein abundance
selfGenes=pd.read_csv('/home/CBBI/tsaih/Research_SingleCell/DataExamination/'+'Table_pbmc_correlation_proteinvsselfRNA_time0.txt', sep='\t')
selfGenes=selfGenes[selfGenes['X']=='counts'].dropna(subset=['corr'])

# Read Pearson correlation between true and predicted protein abundance
DeepG2P=pd.read_csv('/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS/'+
                    'CNN_X_magic_Y_protein_norm_renorm_NOgeneset_corr_results_3.txt', sep='\t')
DeepG2P=DeepG2P.sort_values(by='rppa')

# For each protein, keep the protein-gene pair with the highest correlation
df_sG_highest_corr = selfGenes.loc[selfGenes.groupby('protName')['corr'].idxmax()]

# Reset index if needed (optional)
df_sG_highest_corr = df_sG_highest_corr.reset_index(drop=True)
df=pd.merge(df_sG_highest_corr, DeepG2P, left_on='protName', right_on='rppa')

# Find proportional improvement (median 0.31)
cutoff=0.31

# Step 1: Calculate proportions
# For 'A'
A_greater = (df['corr'] > cutoff).sum() / len(df)  # Proportion of A > 0.3
A_lesser = (df['corr'] <= cutoff).sum() / len(df)  # Proportion of A <= 0.3

# For 'B'
B_greater = (df['pearsonr_test'] > cutoff).sum() / len(df)  # Proportion of B > 0.3
B_lesser = (df['pearsonr_test'] <= cutoff).sum() / len(df)  # Proportion of B <= 0.3

# Step 2: Create a DataFrame to store the proportions for plotting
proportions_df = pd.DataFrame({
    'Category': ['<= 0.3', '> 0.3' ],
    'self-Gene': [A_lesser, A_greater],
    'DeepG2P-SC': [B_lesser, B_greater]
})

# Step 3: Plot a stacked bar plot
proportions_df.set_index('Category').T.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])

# Add labels and title
plt.title('Proportion of Values Greater and Less Than 0.31 in self-Gene and DeepG2P-SC')
plt.ylabel('Proportion')
plt.xlabel('Columns')
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.legend(title='Value Range')
plt.tight_layout()
# plt.savefig('/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS/' +
#             "StackedBar_SC_DeepG2PvsselfGene_proportion_0.31.png", format='png', dpi=600, transparent=True)
# plt.savefig('/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS/' +
#             "StackedBar_SC_DeepG2PvsselfGene_proportion_0.31.pdf", format='pdf', dpi=600, transparent=True)
# plt.close()

# proportions_df.to_csv('/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS/' +
#             "Table_proportion_SC_DeepG2PvsselfGene_0.31.txt", sep='\t')

# ---Densityplot---
corr_min, corr_max = df['corr'].min(), df['corr'].max()
pearsonr_min, pearsonr_max = df['pearsonr_test'].min(), df['pearsonr_test'].max()
ax = sns.kdeplot(df['corr'], color='steelblue', clip=(corr_min, corr_max))
ax = sns.kdeplot(df['pearsonr_test'], color='chocolate', clip=(pearsonr_min, pearsonr_max))

ax.set_xlabel("Pearson correlation", fontsize=15)
ax.set_ylabel('Density', fontsize=15)
ax.tick_params(labelsize=12)
plt.axvline(cutoff, c='gray', linestyle='--', linewidth=1)
plt.tight_layout()
# plt.savefig('/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS/' + "DensityPlot_SC_prorna_paired_truevspred_cutoff0.31_withindatarange.png", format='png', dpi=600, transparent=True)
# plt.savefig('/home/CBBI/tsaih/Research_SingleCell/savedModel/time0_noGS/' + "DensityPlot_SC_prorna_paired_truevspred_cutoff0.31_withindatarange.pdf", format='pdf', dpi=600, transparent=True)
# plt.close()

proportions_df