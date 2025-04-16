import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from constant_variables import *
from scipy.stats import mannwhitneyu, wilcoxon
from itertools import combinations

dir_data='/home/CBBI/tsaih/data/'
dir_output_fig='/home/CBBI/tsaih/Research/Model_Xnorm13995_new/' + 'PaperFigures/'

from pathlib import Path
Path(dir_output_fig).mkdir(parents=True, exist_ok=True)

performance=CNNResult()
originalCor=CorrRNAnPro()
df=pd.merge(originalCor[['proName', 'PearsonCor']], performance[['rppa', 'pearsonr_test']], left_on='proName', right_on='rppa')
df['modification']=np.where(df['proName'].str.contains('_P'), 'Phospho', 'Total')
df['modification']=np.where(df['proName'].str.contains('ACETYL'), 'Acetyl', df['modification'])
df=df.sort_values(by='pearsonr_test')
df['predictability']=np.where(df['PearsonCor']>df['PearsonCor'].median(), 'High self-RNA\ncorrelation', 'Low self-RNA\ncorrelation')
df['diff']=df['pearsonr_test']-df['PearsonCor']

#----add Protein groups annotation-----
ProtGrp=pd.read_csv(dir_data+'ProteinGroupsSummary.txt', sep='\t')
df=pd.merge(df, ProtGrp[['rppa', 'Panther_class']], left_on='proName', right_on='rppa')
df=df[~df.duplicated()]
df['Panther_class']=np.where(df['Panther_class']=='0', 'Others', df['Panther_class'])
df['Panther_class']=df['Panther_class'].str.capitalize()
threshold=df['PearsonCor'].median()
df['predictability']=np.where(df['PearsonCor']>threshold, 'High self-RNA\ncorrelation', 'Low self-RNA\ncorrelation')
df=df.reset_index(drop=True)

# Fig 2A
df_melt=pd.melt(df, id_vars=['proName', 'modification', 'predictability'], value_vars=['PearsonCor', 'pearsonr_test'])

groups=['High self-RNA\ncorrelation', 'Low self-RNA\ncorrelation']

i=1
fig, ax = plt.subplots(figsize=(5, 4))
for group in groups:
    ori = df[(df_melt['predictability'] == group)].drop_duplicates(subset=['proName'])['PearsonCor'].to_numpy()
    pred = df[(df_melt['predictability'] == group)].drop_duplicates(subset=['proName'])['pearsonr_test'].to_numpy()

    x1 = i - 0.2
    x2 = i + 0.3

    # plot lines
    for s_oi, s_pi in zip(ori, pred):
        ax.plot([x1 + 0.02, x2 - 0.02], [s_oi, s_pi], c='lightgray')

    ax.violinplot(ori, positions=[x1], showmedians=True, showmeans=False, showextrema=False, widths=0.2)
    ax.violinplot(pred, positions=[x2], showmedians=True, showmeans=False, showextrema=False, widths=0.2)

    i+=1

#fix the axes and labels
ax.grid(False)
ax.yaxis.grid(False)
ax.set_xticks(list(range(1, len(groups)+1)))
_ = ax.set_xticklabels(groups)

#legend
ori_patch = mpatches.Patch(color='darkkhaki', label='selfGene-measured prot')
pred_patch = mpatches.Patch(color='seagreen', label='DeepG2P-measured prot')
plt.legend(handles=[ori_patch, pred_patch], loc='lower left')

#plt.xlabel('(protein-RNA correlation>'+str(threshold)+')', fontsize=10)
plt.ylabel('Pearson correlation', fontsize=15)
plt.ylim(-0.35, 1.2)
plt.tight_layout()

# plt.savefig(dir_output_fig + 'ViolinScatterConnectPlot_Predictability_threshold_' + str(round(threshold, 3)) + '_median.png',
#             format='png', dpi=600, transparent=True)
# plt.savefig(dir_output_fig + 'ViolinScatterConnectPlot_Predictability_threshold_' + str(round(threshold, 3)) + '_median.pdf',
#             format='pdf', dpi=600, transparent=True)
# plt.close()


print(mannwhitneyu(
    df.loc[(df['predictability'] == 'High self-RNA\ncorrelation') , 'diff'],
    df.loc[(df['predictability'] == 'Low self-RNA\ncorrelation'), 'diff'],
    alternative = 'less')[1])

predGroup='High self-RNA\ncorrelation'
df_sub=df[df['predictability']==predGroup]
print(wilcoxon(df_sub['PearsonCor'], df_sub['pearsonr_test'], alternative="less")[1])

predGroup='Low self-RNA\ncorrelation'
df_sub=df[df['predictability']==predGroup]
print(wilcoxon(df_sub['PearsonCor'], df_sub['pearsonr_test'], alternative="less")[1])

# Fig 2D
order=['Total', 'Phospho']
x_col='modification'
y_col='diff'
hue_col='predictability'
hue_order=['High self-RNA\ncorrelation', 'Low self-RNA\ncorrelation']
width=0.8

df=df.loc[df['modification'] !='Acetyl']
ax=sns.violinplot(x=x_col, y=y_col, data=df, order=order, hue=hue_col, hue_order=hue_order, showfliers=False, cut=0, split=True)
ax=sns.stripplot(x=x_col, y=y_col, data=df, order=order, hue=hue_col, hue_order=hue_order,
                 dodge=True, jitter=True, size=2, color='black')

# get the offsets used by boxplot when hue-nesting is used
# https://github.com/mwaskom/seaborn/blob/c73055b2a9d9830c6fbbace07127c370389d04dd/seaborn/categorical.py#L367
n_levels = len(df[hue_col].unique())
each_width = width / n_levels
offsets = np.linspace(0, width - each_width, n_levels)
offsets -= offsets.mean()

pos = [x+o for x in np.arange(len(order)) for o in offsets]

counts = df.groupby([x_col,hue_col])[y_col].size()
counts = counts.reindex(pd.MultiIndex.from_product([order,hue_order]))
medians = df.groupby([x_col,hue_col])[y_col].median()
medians = medians.reindex(pd.MultiIndex.from_product([order,hue_order]))

for p,n,m in zip(pos,counts,medians):
    if not np.isnan(m):
        ax.annotate('{:.0f}'.format(n), xy=(p, -0.1), xycoords='data', ha='center', va='bottom')

plt.ylim(-0.35, 1.2)
plt.tight_layout()
ax.set_ylabel('Improvement')

# plt.savefig(dir_output_fig + 'ViolinBoxPlot_improvement_' + 'modification_predictaility' + '.png',
#             format='png', dpi=600, transparent=True)
# plt.savefig(dir_output_fig + 'ViolinBoxPlot_improvement_' + 'modification_predictability' + '.pdf',
#             format='pdf', dpi=600, transparent=True)
# plt.close()

print(mannwhitneyu(
    df.loc[(df['modification'] == 'Total') , 'diff'],
    df.loc[(df['modification'] == 'Phospho'), 'diff'],
    alternative = 'two-sided')[1])

df_count=pd.DataFrame(df.groupby(['Panther_class', 'modification']).count()['proName']).reset_index().sort_values(by='proName')

class2others=[]
for p in df_count['Panther_class']:
    #print(p)
    a=df_count[df_count['Panther_class']==p]
    #check if both Total and Phospho exist
    if all(x in a['modification'] for x in ['Total', 'Phospho']):
        if a[a['modification']=='Total']['proName'].values<2 and a[a['modification']=='Phospho']['proName'].values<2:
            class2others.append(p)
    else:
        mod=a['modification'].tolist()
        if ('Acetyl' not in mod) and (len(mod)<2):
            if a[a['modification']==mod]['proName'].values<2:
                class2others.append(p)

df['Panther_class']=np.where(df['Panther_class'].isin(class2others), 'Others', df['Panther_class'])
order=pd.DataFrame(df[df['modification']=='Total'].groupby('Panther_class').median()).sort_values(by='diff', ascending=False).index.tolist()

# Fig 2E
x_col='Panther_class'
y_col='diff'
hue_col='modification'
hue_order=['Total', 'Phospho']
width=0.8


ax=sns.boxplot(x='Panther_class', y='diff', data=df, order=order, hue=hue_col, hue_order=hue_order, showfliers=False)
ax=sns.stripplot(x='Panther_class', y='diff', data=df, order=order, hue=hue_col, hue_order=hue_order,
                 dodge=True, jitter=True, size=2, color='black')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

n_levels = len(df[hue_col].unique())
each_width = width / n_levels
offsets = np.linspace(0, width - each_width, n_levels)
offsets -= offsets.mean()

pos = [x+o for x in np.arange(len(order)) for o in offsets]

counts = df.groupby([x_col,hue_col])[y_col].size()
counts = counts.reindex(pd.MultiIndex.from_product([order,hue_order]))
medians = df.groupby([x_col,hue_col])[y_col].median()
medians = medians.reindex(pd.MultiIndex.from_product([order,hue_order]))


for p,n,m in zip(pos,counts,medians):
    if not np.isnan(m):
        ax.annotate('{:.0f}'.format(n), xy=(p, -0.15), xycoords='data', ha='center', va='bottom')

plt.ylim(-0.35, 1.2)
plt.tight_layout()
ax.set_ylabel('Improvement')
# plt.savefig(dir_output_fig + 'BoxPlot_improvement_' + 'Panther_TotalvsPhospho' + '.png',
#             format='png', dpi=600, transparent=True)
# plt.savefig(dir_output_fig + 'BoxPlot_improvement_' + 'Panther_TotalvsPhospho' + '.pdf',
#             format='pdf', dpi=600, transparent=True)
# plt.close()
df.groupby('predictability').median()

# Fig 2B,C
from scipy.stats import pearsonr
import math
dir_output_fig_provsrna=dir_output_fig + 'Scatterplot_ProteinvsGene_example/'
dir_output_fig_provspred=dir_output_fig + 'Scatterplot_ProteinvsPredicted_example/'
from pathlib import Path
Path(dir_output_fig_provspred).mkdir(parents=True, exist_ok=True)
Path(dir_output_fig_provsrna).mkdir(parents=True, exist_ok=True)

#original protein data
test_pro=pd.read_csv(dir_data + 'Y_data_187x6813_test.txt', sep='\t', index_col=[0])
test_pro=test_pro.T
#original rna data (ztransformed)
test_gene=pd.read_csv(dir_data + 'X_data_batch_13995x6813_test_z.txt', sep='\t', index_col=[0])
test_gene=test_gene.T
#prediction result
dir_model='/home/CBBI/tsaih/Research/Model_Xnorm13995_new/'
tmpName= '1Layer1D_1LayerDense_conv1024_kernelstride50_dense512_batch254_normX_biasFalse'
test_pred=pd.read_csv(dir_model + tmpName + '_predfromtest_' + str(1) + '.txt', sep='\t', index_col=[0])
test_pred.columns=test_pro.columns
test_pred.index=test_pro.index

proName='HSP70'
pairGene=originalCor[originalCor['proName']==proName.replace('_P', '_p')]['geneName'].tolist()[0]
df=pd.merge(pd.merge(test_pro[[proName]], test_pred[[proName]], left_index=True, right_index=True),
                test_gene[[pairGene]], left_index=True, right_index=True)

#Scatterplot: pro vs RNA
cor, pval = pearsonr(df[proName + '_x'], df[pairGene])
cor = round(cor, 2)
if pval < 0.0001:
    pval = '<1e-04'
else:
    pval = "{:.2e}".format(pval)

ax = sns.regplot(x=proName + '_x', y=pairGene, data=df, scatter_kws={'alpha':0.5, 'edgecolors':None})
plt.xlim(-2.5, 4.5) #HSP70
ax.set_box_aspect(1)
ax.set_xlabel('Protein (' + proName + ')', fontsize=18)
ax.set_ylabel('RNA (' + pairGene + ')', fontsize=18)
plt.text(0.1, 0.9, "cor:{}\np:{}".format(cor, pval),
            ha="left", va="top", transform=ax.transAxes, fontsize=17)
ax.set_xticklabels(ax.get_xticks().astype(int), size=14)
ax.set_yticklabels(ax.get_yticks().astype(int), size=14)
plt.tight_layout()
# plt.savefig(dir_output_fig_provsrna + "Scatter_provsrna_" + proName + "_" + pairGene + '_xlimmanual.png', format='png',
#             dpi=600, transparent=True)
# plt.savefig(dir_output_fig_provsrna + "Scatter_provsrna_" + proName + "_" + pairGene + '_xlimmanual_zscoreY.pdf', format='pdf',
#             dpi=600, transparent=True)
# plt.close()

#Predicted vs protein
cor, pval = pearsonr(df[proName+'_x'], df[proName+'_y'])
cor = round(cor, 2)
if pval<0.0001:
    pval='<1e-04'
else:
    pval="{:.2e}".format(pval)

value=max(max(abs(df[proName+'_x'])), max(abs(df[proName+'_y'])))
ax=sns.scatterplot(x=proName+'_x', y=proName+'_y', data=df, alpha=0.5, edgecolor=None, legend=False)
lim = math.ceil(value)
plt.plot([-lim, lim], [-lim, lim], '-r')
plt.xlim(-2.5, 4.5)
plt.ylim(-2.5, 4.5)
ax.set_box_aspect(1)
ax.set_xlabel('Protein (' + proName + ')', fontsize=18)
ax.set_ylabel('Prediction (' + proName + ')', fontsize=18)
plt.text(0.1, 0.9, "cor:{}\np:{}".format(cor, pval),
            ha="left", va="top", transform=ax.transAxes, fontsize=17)
ax.set_xticklabels(ax.get_xticks().astype(int), size=14)
ax.set_yticklabels(ax.get_yticks().astype(int), size=14)
plt.tight_layout()
#plt.savefig(dir_output_fig_provspred + "Scatter_provspred_limRoundUp_" + proName + 'manual.png', format='png', dpi=600, transparent=True)
# plt.savefig(dir_output_fig_provspred + "Scatter_provspred_limRoundUp_" + proName + 'manual_zscoreY.pdf', format='pdf', dpi=600, transparent=True)
# plt.close()

proName='ERALPHA'
pairGene=originalCor[originalCor['proName']==proName.replace('_P', '_p')]['geneName'].tolist()[0]
df=pd.merge(pd.merge(test_pro[[proName]], test_pred[[proName]], left_index=True, right_index=True),
                test_gene[[pairGene]], left_index=True, right_index=True)

#Scatterplot: pro vs RNA
cor, pval = pearsonr(df[proName + '_x'], df[pairGene])
cor = round(cor, 2)
if pval < 0.0001:
    pval = '<1e-04'
else:
    pval = "{:.2e}".format(pval)

ax = sns.regplot(x=proName + '_x', y=pairGene, data=df, scatter_kws={'alpha':0.5, 'edgecolors':None})

plt.xlim(-lim, lim)
ax.set_box_aspect(1)
ax.set_xlabel('Protein (' + proName + ')', fontsize=18)
ax.set_ylabel('RNA (' + pairGene + ')', fontsize=18)
plt.text(0.1, 0.9, "cor:{}\np:{}".format(cor, pval),
            ha="left", va="top", transform=ax.transAxes, fontsize=17)
ax.set_xticklabels(ax.get_xticks().astype(int), size=14)
ax.set_yticklabels(ax.get_yticks().astype(int), size=14)
plt.tight_layout()
# plt.savefig(dir_output_fig_provsrna + "Scatter_provsrna_" + proName + "_" + pairGene + '_xlimmanual.png', format='png',
#             dpi=600, transparent=True)
# plt.savefig(dir_output_fig_provsrna + "Scatter_provsrna_" + proName + "_" + pairGene + '_xlimmanual_zscoreY.pdf', format='pdf',
#             dpi=600, transparent=True)
# plt.close()

#Predicted vs protein
cor, pval = pearsonr(df[proName+'_x'], df[proName+'_y'])
cor = round(cor, 2)
if pval<0.0001:
    pval='<1e-04'
else:
    pval="{:.2e}".format(pval)

value=max(max(abs(df[proName+'_x'])), max(abs(df[proName+'_y'])))
    #
ax=sns.scatterplot(x=proName+'_x', y=proName+'_y', data=df, alpha=0.5, edgecolor=None, legend=False)
lim = math.ceil(value)
plt.plot([-lim, lim], [-lim, lim], '-r')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
ax.set_box_aspect(1)
ax.set_xlabel('Protein (' + proName + ')', fontsize=18)
ax.set_ylabel('Prediction (' + proName + ')', fontsize=18)
plt.text(0.1, 0.9, "cor:{}\np:{}".format(cor, pval),
            ha="left", va="top", transform=ax.transAxes, fontsize=17)
ax.set_xticklabels(ax.get_xticks().astype(int), size=14)
ax.set_yticklabels(ax.get_yticks().astype(int), size=14)
plt.tight_layout()
#plt.savefig(dir_output_fig_provspred + "Scatter_provspred_limRoundUp_" + proName + 'manual.png', format='png', dpi=600, transparent=True)
# plt.savefig(dir_output_fig_provspred + "Scatter_provspred_limRoundUp_" + proName + 'manual_zscoreY.pdf', format='pdf', dpi=600, transparent=True)
# plt.close()
