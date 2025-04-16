import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from constant_variables import *
from scipy.stats import mannwhitneyu
import scipy.stats as stats
import statsmodels.api as sm

dir_input = "/home/CBBI/tsaih/data/"
dir_output_fig='/home/CBBI/tsaih/Research/Model_Xnorm13995_new/' + 'PaperFigures/'

from pathlib import Path
Path(dir_output_fig).mkdir(parents=True, exist_ok=True)

#Fig 1B
performance=CNNResult()
originalCor=CorrRNAnPro()
df=pd.merge(originalCor[['proName', 'PearsonCor']], performance[['rppa', 'pearsonr_test']], left_on='proName', right_on='rppa')
random_result=pd.read_csv(dir_input + 'cor_rnapro_random.txt', sep='\t')
random_result2=random_result.head(9350)
ax=sns.distplot(random_result2['corr'], hist=False, rug=False, color='lightgray')
ax=sns.distplot(df['PearsonCor'], hist=False, rug=False, color='steelblue')
ax=sns.distplot(df['pearsonr_test'], hist=False, rug=False, color='chocolate')
ax.set_xlabel("Pearson correlation", fontsize=15)
ax.set_ylabel('Density', fontsize=15)
ax.tick_params(labelsize=12)
plt.axvline(random_result2['corr'].median(), c='lightgray', linestyle='--', linewidth=1)
plt.axvline(df['PearsonCor'].median(), c='steelblue', linestyle='--', linewidth=1)
plt.axvline(df['pearsonr_test'].median(), c='chocolate', linestyle='--', linewidth=1)
plt.xlim(-0.7, 1.2)
plt.ylim(0, 4)
plt.tight_layout()
# plt.savefig(dir_output_fig + "Histogram_prorna_pairedvsrandomvsmodel_truevspred.png", format='png', dpi=600, transparent=True)
# plt.savefig(dir_output_fig + "Histogram_prorna_pairedvsrandomvsmodel_truevspred.pdf", format='pdf', dpi=600, transparent=True)
# plt.close()

print(random_result2['corr'].median()) #random: 00387
print(df['PearsonCor'].median()) #self-gene: 0.30751
print(df['pearsonr_test'].median()) #DeepGxP: 0.67949

# random vs self
print(mannwhitneyu(random_result2['corr'], df['PearsonCor'], alternative="less")[1])
# DeepGxP vs self
print(mannwhitneyu(df['pearsonr_test'], df['PearsonCor'], alternative="greater")[1])
# random vs DeepGxP
print(mannwhitneyu(random_result2['corr'], df['pearsonr_test'], alternative="less")[1])

# Fig 1C
from glob import glob
import re

#y-scrambling
filenames_yscramble=glob('/home/CBBI/tsaih/Research/diffMLmethods_comparison/WholeModelRandom/' + 'ResultTable_Yscrambling_shuffle' + '*.txt')
df_yscram=[]
for f in filenames_yscramble:
    data=pd.read_csv(f, sep='\t')
    data['method']='Y_scrambling'
    data['fold'] = re.search('shuffle(.*).txt', f).group(1)
    df_yscram.append(data)
df_yscram=pd.concat(df_yscram, ignore_index=True)
df_yscram_perfold=df_yscram.groupby(['method', 'fold']).mean().reset_index()
df_yscram_perfold.columns=['methods', 'fold', 'cor_test', 'cor_P', 'mse_test']

# 7 DL and ML methods
dir_input="/home/CBBI/tsaih/Research/diffMLmethods_comparison/CrossValidation_Xnorm_Z01/"
methods=['CNN2', 'XGB', 'VAE' , 'mLR', 'LR', 'CCA', 'Resnet']
df=[]
for method in methods:
    data=pd.read_csv(dir_input + "Table_result_10foldCrossValidation_1_"+ method + '.txt', sep='\t')
    data['methods']=method
    df.append(data)

df=pd.concat(df, ignore_index=True)
df['methods']=np.where(df['methods']=='CNN2', 'DeepGxP', df['methods'])
df_plot=pd.concat([df[['methods', 'mse_test', 'cor_test']],
                   df_yscram_perfold[['methods', 'mse_test', 'cor_test']]],
                  ignore_index=True)

#SVM
df_svm_mse=pd.read_csv("/home/CBBI/tsaih/MatlabCode_chris/MSE_SVM_fitrlinear_10iterations.txt", sep='\t', header=None)
df_svm_cor=pd.read_csv("/home/CBBI/tsaih/MatlabCode_chris/R_SVM_fitrlinear_10iterations.txt", sep='\t', header=None)
df_svm=pd.DataFrame({
    'methods': ["SVM"]*10,
    "mse_test": df_svm_mse.mean(),
    "cor_test": df_svm_cor.mean()
})

df_plot2=pd.concat([df_plot, df_svm])
model_type=pd.DataFrame({
    'methods':['DeepGxP', 'XGB', 'SVM', 'VAE', 'mLR', 'Resnet', 'LR', 'CCA', 'Y_scrambling'],
    'modelType': ['ourModel', 'advML', 'ML', 'DL', 'ML', 'DL', 'ML', 'ML', 'control']
})
df_plot2=pd.merge(df_plot2, model_type, on='methods')

ax = sns.barplot(x='methods', y='cor_test', data=df_plot2, ci='sd',
                 order=['DeepGxP', 'XGB', 'SVM', 'VAE', 'mLR', 'Resnet', 'LR', 'CCA', 'Y_scrambling'],
                 palette=['dodgerblue', 'sandybrown', 'peachpuff', 'lightskyblue', 'peachpuff', 'lightskyblue', 'peachpuff', 'peachpuff', 'white'],
                 edgecolor='black')
ax = sns.stripplot(x='methods', y='cor_test', data=df_plot2, order=['DeepG2P', 'XGB', 'SVM', 'VAE', 'mLR', 'Resnet', 'LR', 'CCA', 'Y_scrambling'],
                       color='gray', size=3, alpha=0.9)
plt.legend([], [], frameon=False)
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='center')
ax.set_ylabel("Pearson correlation", fontsize=18)
ax.tick_params(labelsize=15)
plt.tight_layout()
#plt.savefig(dir_output_fig + "Barplot_diffML_" + 'PearsonCorrelation' + "_" + 'CV1_0213_errorbar_sd' + ".png", format='png', dpi=600, transparent=True)
# plt.savefig(dir_output_fig + "Barplot_diffML_" + 'PearsonCorrelation' + "_" + 'CV1_0213_errorbar_sd' + ".pdf", format='pdf', dpi=600, transparent=True)
# plt.close()

ax = sns.barplot(x='methods', y='mse_test', data=df_plot2, order=['DeepG2P', 'XGB', 'SVM', 'VAE', 'mLR', 'Resnet', 'LR', 'CCA', 'Y_scrambling'],
                 palette=['dodgerblue', 'sandybrown', 'peachpuff', 'lightskyblue', 'peachpuff', 'lightskyblue', 'peachpuff', 'peachpuff', 'white'],
                 edgecolor='black')
ax = sns.stripplot(x='methods', y='mse_test', data=df_plot2, order=['DeepG2P', 'XGB', 'SVM', 'VAE', 'mLR', 'Resnet', 'LR', 'CCA', 'Y_scrambling'],
                       color='gray', size=3, alpha=0.9)
plt.legend([], [], frameon=False)
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='center')
ax.set_ylabel("MSE", fontsize=18)
ax.tick_params(labelsize=15)
plt.tight_layout()
# plt.savefig(dir_output_fig + "Barplot_diffML_" + 'MSE' + "_" + 'CV1_0213' + ".png", format='png', dpi=600, transparent=True)
# plt.savefig(dir_output_fig + "Barplot_diffML_" + 'MSE' + "_" + 'CV1_0213' + ".pdf", format='pdf', dpi=600, transparent=True)
# plt.close()

# 1E,F omit one tumor
tumortype='BRCA'
dir_input='/home/CBBI/tsaih/Research/Model_Xnorm13995_new/OmitOnetumortype_nsample/' + tumortype + '/'

#Pearson
dfCor = []
filenames=glob(dir_input + 'pearsonResult_omit_*_'+ tumortype +'.txt')
for f in filenames:
    data = pd.read_csv(f, sep='\t')
    nsamples=re.search('_omit_' + '(.*)_'+ tumortype +'.txt', f).group(1)
    data['nsamples']=nsamples
    dfCor.append(data)

dfCor=pd.concat(dfCor, ignore_index=True)
dfCor['nsamples']=dfCor['nsamples'].astype(int)
dfCor=dfCor[dfCor['nsamples'].isin([49, 50, 51])==False]
dfCor_mean=dfCor.groupby(['nsamples', 'round']).mean().reset_index()
dfCor_mean['nsamples']=dfCor_mean['nsamples'].astype(int)
dfCor_mean=dfCor_mean.sort_values(by='nsamples', ascending=True)
dfCor_mean=dfCor_mean[dfCor_mean['nsamples'].isin([49, 50, 51])==False]

# Fig 1E (BRCA tumor)
y_lowess = pd.DataFrame(sm.nonparametric.lowess(dfCor_mean['pearsonr_test'].tolist(),dfCor_mean['nsamples'].tolist(), frac = 0.30))  # 30 % lowess smoothing
y_lowess=y_lowess.drop_duplicates()
y_lowess.columns=['x', 'y_lowess']
y_lowess['growth']=y_lowess['y_lowess'].pct_change()
idx=y_lowess['growth'].idxmin()-10

ax=sns.regplot(data=dfCor_mean, x="nsamples", y="pearsonr_test", fit_reg=False, scatter=False)
sns.lineplot(data=dfCor_mean, x="nsamples", y="pearsonr_test", err_style="bars")
sns.lineplot(data=dfCor_mean, x="nsamples", y="pearsonr_test")
plt.plot(y_lowess['x'], y_lowess['y_lowess'], color='red')
plt.scatter(y_lowess['x'][idx], y_lowess['y_lowess'][idx], color='black')
ax.vlines(x=y_lowess['x'][idx], ymin=0.34, ymax=y_lowess['y_lowess'][idx], ls='dashed', color='gray')
ax.hlines(y=y_lowess['y_lowess'][idx], xmin=0, xmax=y_lowess['x'][idx], ls='dashed', color='gray')
plt.xlabel('Number of Samples', size=16)
plt.ylabel('Pearson correlation', size=16)
plt.title(tumortype, loc='left', size=20)
plt.tight_layout()
# plt.savefig(dir_input + 'Pointplot_' + 'Cor' + '_' + tumortype + '_skip50_lowess_growth.png', format='png', dpi=600)
# plt.savefig(dir_input + 'Pointplot_' + 'Cor' + '_' + tumortype + '_skip50_lowess_growth.pdf', format='pdf', dpi=600)
# plt.close()
print(y_lowess['x'][idx], y_lowess['y_lowess'][idx])

# Fig 1F (ERALPHA)
a=dfCor[dfCor['rppa'].isin(['ERALPHA'])]
y_lowess = pd.DataFrame(sm.nonparametric.lowess(a['pearsonr_test'].tolist(),a['nsamples'].tolist(), frac = 0.30))  # 30 % lowess smoothing
y_lowess=y_lowess.drop_duplicates()
y_lowess.columns=['x', 'y_lowess']
y_lowess['growth']=y_lowess['y_lowess'].pct_change()
idx=y_lowess['growth'].idxmin()-10

ax=sns.regplot(data=a, x="nsamples", y="pearsonr_test", fit_reg=False, scatter=False)
sns.lineplot(data=a, x="nsamples", y="pearsonr_test", err_style="bars")
sns.lineplot(data=a, x="nsamples", y="pearsonr_test")
plt.plot(y_lowess['x'], y_lowess['y_lowess'], color='red')
plt.scatter(y_lowess['x'][idx], y_lowess['y_lowess'][idx], color='black')
ax.vlines(x=y_lowess['x'][idx], ymin=0.74, ymax=y_lowess['y_lowess'][idx], ls='dashed', color='gray')
ax.hlines(y=y_lowess['y_lowess'][idx], xmin=0, xmax=y_lowess['x'][idx], ls='dashed', color='gray')
plt.xlabel('Number of Samples', size=16)
plt.ylabel('Pearson correlation', size=16)
plt.title('ERALPHA', loc='left', size=20)
plt.tight_layout()
# plt.savefig(dir_input + 'Pointplot_' + 'Cor' + '_' + tumortype + '_skip50_lowess_ERALPHA_growth.png', format='png', dpi=600)
# plt.savefig(dir_input + 'Pointplot_' + 'Cor' + '_' + tumortype + '_skip50_lowess_ERALPHA_growth.pdf', format='pdf', dpi=600)
# plt.close()
print(y_lowess['x'][idx], y_lowess['y_lowess'][idx])

