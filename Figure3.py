from constant_variables import *
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

#Fig 3A
dir_input="/home/CBBI/tsaih/Research/Model_Xnorm13995_new/IG/GradientDistribution_HighvsLow_proabd/PanCancer/"
#high vs low correlation
gradSummary=pd.read_csv(dir_input+'gradient_mean_summary' + '.txt', sep='\t',index_col=[0])
originalCor=CorrRNAnPro()
gradSummary=pd.merge(gradSummary, originalCor[['proName', 'PearsonCor']], on='proName')
gradSummary['modification']=np.where(gradSummary['proName'].str.contains('_P'), 'Phospho', 'Total')
gradSummary['modification']=np.where(gradSummary['proName'].str.contains('ACETYL'), 'Acetyl', gradSummary['modification'])
gradSummary['CorGroup']=np.where(gradSummary['PearsonCor']>gradSummary['PearsonCor'].median(), 'High', 'Low')
#grad in percentage
gradSummary['RankPercentage']=(gradSummary['high_rank']/13995)*100
gradSummary_total=gradSummary[gradSummary['modification']=='Total']

#plot
g=sns.violinplot(x='CorGroup', y='RankPercentage', data=gradSummary_total, order=['High', 'Low'], cut=0)
g=sns.stripplot(x='CorGroup', y='RankPercentage', data=gradSummary_total, order=['High', 'Low'],
                color='black', dodge=True)
plt.ylim(-10, 110)
g.invert_yaxis()
plt.tight_layout()
# plt.savefig(dir_output + 'Violinplot_' + 'Grdient_highrank_corGroup_percent_total' + ".png", format='png', dpi=600)
# plt.savefig(dir_output + 'Violinplot_' + 'Grdient_highrank_corGroup_percent_total' + ".pdf", format='pdf', dpi=600)
# plt.close()

#Fig 3B
dir_input='/home/CBBI/tsaih/Research/Model_Xnorm13995_new/IG/GradientDistribution_HighvsLow_proabd/PanCancer/'
dir_output= dir_input +'Plot_IGDistribution_mark196/'
from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

Pairs=CorrRNAnPro()
Pairs.proName=Pairs.proName.str.upper()

prot='CYCLINB1'
filename=dir_input+ 'gradient_mean_HvsL_' + prot +'.txt'
grad_mean = pd.read_csv(filename, sep="\t")
grad_mean.index=grad_mean['Unnamed: 0']
pairGene=Pairs[Pairs.proName==prot]['geneName'].tolist()[0]

ax = sns.distplot(grad_mean['high_zscore'], kde=False)
plt.vlines(x=grad_mean.loc[pairGene]['high_zscore'].tolist(), ymin=0, ymax=200, color='r')
plt.title("Protein:" + prot, fontsize=22)
ax.set_xlabel("Integrated gradient (zscore)", fontsize=18)
ax.set_ylabel("Number of genes", fontsize=18)
#additional ticks at -1.96 and 1.96
ax.set_xticks(list(ax.get_xticks()) + [-1.96, 1.96])
ax.set_yticklabels(ax.get_yticks().astype(int), size=14)
plt.text(grad_mean.loc[pairGene]['high_zscore'].tolist() + 0.0001, 200 + 50,
             "Rank:{}".format(grad_mean.loc[pairGene]['high_rank'].astype(int)), fontsize=17)
plt.tight_layout()
# plt.savefig(dir_output + "Densityplot_gradient_z_" + prot + ".png", format='png', dpi=600,
#                     transparent=True)
# plt.savefig(dir_output + "Densityplot_gradient_z_" + prot + ".pdf", format='pdf', dpi=600,
#                     transparent=True)
# plt.close()

prot='CYCLINE2'
filename=dir_input+ 'gradient_mean_HvsL_' + prot +'.txt'
grad_mean = pd.read_csv(filename, sep="\t")
grad_mean.index=grad_mean['Unnamed: 0']
pairGene=Pairs[Pairs.proName==prot]['geneName'].tolist()[0]

ax = sns.distplot(grad_mean['high_zscore'], kde=False)
plt.vlines(x=grad_mean.loc[pairGene]['high_zscore'].tolist(), ymin=0, ymax=200, color='r')
plt.title("Protein:" + prot, fontsize=22)
ax.set_xlabel("Integrated gradient (zscore)", fontsize=18)
ax.set_ylabel("Number of genes", fontsize=18)
#additional ticks at -1.96 and 1.96
ax.set_xticks(list(ax.get_xticks()) + [-1.96, 1.96])
ax.set_yticklabels(ax.get_yticks().astype(int), size=14)
plt.text(grad_mean.loc[pairGene]['high_zscore'].tolist() + 0.0001, 200 + 50,
             "Rank:{}".format(grad_mean.loc[pairGene]['high_rank'].astype(int)), fontsize=17)
plt.tight_layout()
# plt.savefig(dir_output + "Densityplot_gradient_z_" + prot + ".png", format='png', dpi=600,
#                     transparent=True)
# plt.savefig(dir_output + "Densityplot_gradient_z_" + prot + ".pdf", format='pdf', dpi=600,
#                     transparent=True)
# plt.close()

prot='CHK2'
filename=dir_input+ 'gradient_mean_HvsL_' + prot +'.txt'
grad_mean = pd.read_csv(filename, sep="\t")
grad_mean.index=grad_mean['Unnamed: 0']
pairGene=Pairs[Pairs.proName==prot]['geneName'].tolist()[0]

ax = sns.distplot(grad_mean['high_zscore'], kde=False)
plt.vlines(x=grad_mean.loc[pairGene]['high_zscore'].tolist(), ymin=0, ymax=200, color='r')
plt.title("Protein:" + prot, fontsize=22)
ax.set_xlabel("Integrated gradient (zscore)", fontsize=18)
ax.set_ylabel("Number of genes", fontsize=18)
#additional ticks at -1.96 and 1.96
ax.set_xticks(list(ax.get_xticks()) + [-1.96, 1.96])
ax.set_yticklabels(ax.get_yticks().astype(int), size=14)
plt.text(grad_mean.loc[pairGene]['high_zscore'].tolist() + 0.0001, 200 + 50,
             "Rank:{}".format(grad_mean.loc[pairGene]['high_rank'].astype(int)), fontsize=17)
plt.tight_layout()
# plt.savefig(dir_output + "Densityplot_gradient_z_" + prot + ".png", format='png', dpi=600,
#                     transparent=True)
# plt.savefig(dir_output + "Densityplot_gradient_z_" + prot + ".pdf", format='pdf', dpi=600,
#                     transparent=True)
# plt.close()

prot='CHK2_PT68'
filename=dir_input+ 'gradient_mean_HvsL_' + prot +'.txt'
grad_mean = pd.read_csv(filename, sep="\t")
grad_mean.index=grad_mean['Unnamed: 0']
pairGene=Pairs[Pairs.proName==prot]['geneName'].tolist()[0]

ax = sns.distplot(grad_mean['high_zscore'], kde=False)
plt.vlines(x=grad_mean.loc[pairGene]['high_zscore'].tolist(), ymin=0, ymax=200, color='r')
plt.title("Protein:" + prot, fontsize=22)
ax.set_xlabel("Integrated gradient (zscore)", fontsize=18)
ax.set_ylabel("Number of genes", fontsize=18)
#additional ticks at -1.96 and 1.96
ax.set_xticks(list(ax.get_xticks()) + [-1.96, 1.96])
ax.set_yticklabels(ax.get_yticks().astype(int), size=14)
plt.text(grad_mean.loc[pairGene]['high_zscore'].tolist() + 0.0001, 200 + 50,
             "Rank:{}".format(grad_mean.loc[pairGene]['high_rank'].astype(int)), fontsize=17)
plt.tight_layout()
# plt.savefig(dir_output + "Densityplot_gradient_z_" + prot + ".png", format='png', dpi=600,
#                     transparent=True)
# plt.savefig(dir_output + "Densityplot_gradient_z_" + prot + ".pdf", format='pdf', dpi=600,
#                     transparent=True)
# plt.close()

# Fig 3D
from scipy.stats import pearsonr
dir_input='/home/CBBI/tsaih/data/'
dir_output=dir_input + 'Correlation_TotalvsPhosph/'
from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

Ytrain=pd.read_csv(dir_input+"Y_data_187x6813_train.txt", sep='\t', index_col=[0]).T
Yval=pd.read_csv(dir_input+"Y_data_187x6813_val.txt", sep='\t', index_col=[0]).T
Ytest=pd.read_csv(dir_input+"Y_data_187x6813_test.txt", sep='\t', index_col=[0]).T
Y=pd.concat([Ytrain, Yval, Ytest])
Y.columns=Y.columns.str.upper()

t='CHK2'
p='CHK2_PT68'

sns.scatterplot(Y[t], Y[p], edgecolor=None, alpha=0.5, size=1, legend=False)
cor, pval = pearsonr(Y[t], Y[p])
cor = round(cor, 2)
pval = "{:.2e}".format(pval)
ax.text(0.05, 0.95, "cor:{}, p:{}".format(cor, pval), ha="left", va="top", transform=ax.transAxes,
               fontsize=10)
# plt.tight_layout()
# plt.savefig(dir_output + 'Scatterplot_'+ tumortype + '_' + t + '_vs_' + p + '.pdf',
#             format='pdf', dpi=600, transparent=True)
# plt.close()

