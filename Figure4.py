from constant_variables import *
from lifelines import KaplanMeierFitter, CoxPHFitter
from matplotlib.transforms import Affine2D
from lifelines.statistics import logrank_test
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def SurvivalTests(df, SurvData, SurvType):
    #log rank
    group = np.where(df >= df.median(), 1, 0)
    ix = (group == 1)  # high expression
    kmf = KaplanMeierFitter()
    surv_time = SurvData['time_mon_new']
    status = SurvData[SurvType]
    time_high = surv_time[ix]
    status_high = status[ix]
    time_low = surv_time[~ix]
    status_low = status[~ix]
    resultLogrank = logrank_test(time_high, time_low, event_observed_A=status_high, event_observed_B=status_low)

    # cox regression
    df.reset_index(drop=True, inplace=True)
    surv_time.reset_index(drop=True, inplace=True)
    status.reset_index(drop=True, inplace=True)
    df = pd.concat([df, surv_time, status], axis=1)
    df.isnull().sum().sum()
    df.isnull().sum()
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time_mon_new', event_col=SurvType, step_size=0.5)
    resultCoxreg_HR=cph.summary['exp(coef)'][0]
    resultCoxreg_pval = cph.summary['p'][0]
    resultCoxreg_low =cph.summary['exp(coef) lower 95%'][0]
    resultCoxreg_high = cph.summary['exp(coef) upper 95%'][0]
    resultCoxreg_concord=cph.concordance_index_

    return resultLogrank.p_value, resultCoxreg_HR, resultCoxreg_pval, resultCoxreg_low, resultCoxreg_high, resultCoxreg_concord


def runSurvTest_GnP(x, y, tumortype, OS_tumor):
    x= Zscore(x, x.index)
    y = Zscore(y, y.index)
    df_survResult = np.empty((y.shape[1], 13))
    ls_target_gene=[]
    ls_target_pro=[]
    for i in tqdm(range(y.shape[1])):
        target_pro=y.columns[i].replace('_Y', '')
        target_gene=Pairs[Pairs['proName']==target_pro]['geneName'].to_list()[0]

        #protein
        df_target_pro=y[target_pro+'_Y']
        survResult_pro=SurvivalTests(df_target_pro, OS_tumor, survival_type)

        #RNA
        df_target_gene = x[target_gene]
        survResult_gene = SurvivalTests(df_target_gene, OS_tumor, survival_type)

        df_survResult[i, 0] = survResult_pro[0]
        df_survResult[i, 1] = survResult_pro[1]
        df_survResult[i, 2] = survResult_pro[2]
        df_survResult[i, 3] = survResult_pro[3]
        df_survResult[i, 4] = survResult_pro[4]
        df_survResult[i, 5] = survResult_gene[0]
        df_survResult[i, 6] = survResult_gene[1]
        df_survResult[i, 7] = survResult_gene[2]
        df_survResult[i, 8] = survResult_gene[3]
        df_survResult[i, 9] = survResult_gene[4]
        df_survResult[i, 10] = survResult_pro[5]
        df_survResult[i, 11] = survResult_gene[5]
        df_survResult[i, 12] = x.shape[0]
        ls_target_pro.append(target_pro)
        ls_target_gene.append(target_gene)
        pass
    df_survResult=pd.DataFrame(df_survResult,
                                            columns=['logrankp_pro', 'coxHR_pro', 'coxp_pro', 'coxci_low_pro', 'coxci_high_pro',
                                                     'logrankp_gene', 'coxHR_gene', 'coxp_gene', 'coxci_low_gene', 'coxci_high_gene',
                                                      'c-index_pro','c-index_gene', 'sample_n'])
    df_survResult['censor']=OS_tumor['OS'].sum()
    df_survResult.insert(loc=0, column='proName', value=ls_target_pro)
    df_survResult.insert(loc=1, column='geneName', value=ls_target_gene)
    df_survResult.insert(loc=2, column='tumortype', value=tumortype)

    df_survResult['log10coxp_pro'] = np.log10(df_survResult['coxp_pro']) * -1
    df_survResult['log10coxp_gene'] = np.log10(df_survResult['coxp_gene']) * -1
    df_survResult['FDR(BH)_pro'] = multitest.multipletests(df_survResult['coxp_pro'].values, method='fdr_bh')[1]
    df_survResult['FDR(BH)_gene'] = multitest.multipletests(df_survResult['coxp_gene'].values, method='fdr_bh')[1]
    df_survResult['log10FDR(BH)_pro'] = np.log10(df_survResult['FDR(BH)_pro']) * -1
    df_survResult['log10FDR(BH)_gene'] = np.log10(df_survResult['FDR(BH)_gene']) * -1
    # df_survResult.insert(loc=5, column='log10(coxHR_pro)', value=np.log10(df_survResult['coxHR_pro']))
    # df_survResult.insert(loc=8, column='log10(coxHR_gene)', value=np.log10(df_survResult['coxHR_gene']))
    df_survResult = df_survResult.sort_values(by=['coxHR_pro', 'coxp_pro'], ascending=False)

    return df_survResult

def Zscore(x, samplelist):
    df=(x.loc[samplelist] - x.loc[samplelist].mean()) / x.loc[samplelist].std(ddof=0)
    return df

def plotCoxReg(df, tumortype, lset, llabel, lcolor, nbar, FDR, ratio):
    df=df.sort_values(by='coxHR_'+ lset[0])

    ypos = list(range((nbar-1)*-10, nbar*10, 20))
    ypos.reverse()
    ypos = [y / 100 for y in ypos]

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex='col', figsize=(6, 10), gridspec_kw={'width_ratios': ratio})
    fig.suptitle(tumortype + '(' + str(int(df_plot['sample_n'].unique()[0])) + ')',  fontsize=16)

    for i in range(nbar):
        #error bar
        xerr=[
        [value - lower for value, lower in zip(df['coxHR_'+lset[i]], df['coxci_low_'+lset[i]])],
        [upper - value for value, upper in zip(df['coxHR_'+lset[i]], df['coxci_high_'+lset[i]])]]
        #adjust y position
        trans = Affine2D().translate(0.0, ypos[i]) + ax[0].transData
        #plot scatter
        ax[0].scatter(df['coxHR_' + lset[i]], df['proName'], c=lcolor[i], marker='o', transform=trans, s=15)
        #plot error bar
        ax[0].errorbar(df['coxHR_' + lset[i]], df['proName'], xerr=xerr,
                       label=llabel[i],
                       fmt='none',  # don't connect data points
                       ecolor=lcolor[i],  # color of error lines
                       elinewidth=0.8,  # width of error lines
                       capsize=2,  # length of error caps
                       zorder=-1,  # put error bars behind scatter points
                       transform=trans)

    fig.subplots_adjust(wspace=0.1, hspace=0.05)  # adjust space between two subplots

    # barplot for pvalue
    if FDR:
        pvaltype='FDR'
        plist = ['log10' + pvaltype+ '(BH)_' + p for p in lset]
    else:
        pvaltype = 'pval'
        plist = ['log10coxp_' + p for p in lset]
    plist.reverse()
    lcolor.reverse()
    df[['proName'] + plist].plot(kind='barh', x='proName', legend=False, ax=ax[1], color=lcolor)

    # Add axis names
    ax[0].set_xlabel('HR(95% CI)')
    ax[1].set_xlabel('-log10(' + pvaltype + ')')

    # change xtick size
    ax[0].tick_params(axis='y', labelsize=8)

    # Add titles
    #fig.suptitle(tumortype, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    #remove grids
    ax[0].grid(False)
    ax[1].grid(False)

    # add lines
    ax[0].axvline(1, linestyle='--', color='lightgray', linewidth=1)
    ax[1].axvline(np.log10(0.2) * -1, linestyle='--', color='lightgray', linewidth=1)

    # make significant proteins bold
    if FDR:
        sigpros = list(df[(df['FDR(BH)_' + 'pro'] < 0.2) & (df['FDR(BH)_' + 'pro'] < df['FDR(BH)_' + 'gene'])]['proName'])
    else:
        sigpros = list(df[(df['coxp_' + 'pro'] < 0.05) & (df['coxp_' + 'pro'] < df['coxp_' + 'gene'])]['proName'])

    for pro in ax[0].get_yticklabels():
        if str(pro).split("'")[1] in sigpros:
            pro.set_fontweight("bold")

    return plt

def KMplot(df, SurvData, SurvType, tumortype, featureName, color_high, color_low, linestyle_high, linestyle_low):
    df = (df - df.mean()) / df.std(ddof=0)
    group = np.where(df >= df.median(), 1, 0)
    ix = (group == 1)  # high expression
    kmf = KaplanMeierFitter()
    surv_time = SurvData['time_mon_new']
    status = SurvData[SurvType]

    kmf.fit(surv_time[ix], status[ix], label='High (n=' + str(len(status[ix])) + ')')
    ax = kmf.plot(ci_show=False, color=color_high, figsize=(5, 5), show_censors=True, at_risk_counts=False, linestyle=linestyle_high)
    kmf.fit(surv_time[~ix], status[~ix], label='Low (n=' + str(len(status[~ix])) + ')')
    ax = kmf.plot(ax=ax, ci_show=False, color=color_low, show_censors=True, at_risk_counts=False, linestyle=linestyle_low)
    ax.set_ylabel('Survival probability', fontsize=18)
    ax.set_xlabel('Overall survival (Months)', fontsize=18)
    #plt.rc('legend', fontsize=12)
    plt.legend(loc='lower right', fontsize=12, frameon=True)
    ax.set_ylim(-0.1, 1.1)

    #log-rank test
    time_high = surv_time[ix]
    status_high = status[ix]
    time_low = surv_time[~ix]
    status_low = status[~ix]
    resultLogrank = logrank_test(time_high, time_low, event_observed_A=status_high, event_observed_B=status_low)

    # cox regression
    df.reset_index(drop=True, inplace=True)
    surv_time.reset_index(drop=True, inplace=True)
    status.reset_index(drop=True, inplace=True)
    df = pd.concat([df, surv_time, status], axis=1)
    df.isnull().sum().sum()
    df.isnull().sum()
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time_mon_new', event_col=SurvType, step_size=0.5)
    Coxreg_HR = f"{cph.summary['exp(coef)'][0]: .3f}"
    Coxreg_pval = f"{cph.summary['p'][0]: .3f}"
    logrank_pval = f"{resultLogrank.p_value: .3f}"
    ax.text(0.5, 0.01,
            #'CoxHR = ' + str(Coxreg_HR) + '\n' +
            'CoxP= ' + str(Coxreg_pval) + '\n' +
            'Log-rankP= ' + str(logrank_pval),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=14)
    plt.title(tumortype + featureName, fontsize=20)
    #print(cph.print_summary())

    return plt, cph.summary['exp(coef)'][0], cph.summary['p'][0], resultLogrank.p_value

target_tumor='BRCA'
n_year=5
dir_input='/home/CBBI/tsaih/data/'
dir_output='/home/CBBI/tsaih/Research/' + 'Survival_PronGene_new/' + target_tumor + '_' + str(n_year) +'year_HR/'

from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

# Read survival data, RNA expression and protein abundance data
clinicData=ClinicalData()
omicData=fullData()
Pairs=CorrRNAnPro()
procorpertumor=ProCorperTumor()
Pairs['proName']=Pairs['proName'].str.replace('_P', '_p')
# Run cox regression on training data
cutoff = n_year * 12
survival_type='OS_new'
dataset=['Train', 'Val', 'Test']
samples=clinicData[(clinicData['Disease_short']==target_tumor) & (clinicData['DataSets'].isin(dataset))]


markers=['ER']
statuses=['positive']

SurvResult=[]
for marker in markers:
    for status in statuses:
        targetSamples=samples[samples[marker + '_status']==status][['SampleID', 'OS', 'OS.time']]
        targetSamples=targetSamples.dropna()
        targetSamples['time_mon']=targetSamples['OS.time']/30
        targetSamples.index = targetSamples['SampleID']

        #five year survival
        targetSamples['time_mon_new'] = np.where(targetSamples['time_mon'] > cutoff, cutoff, targetSamples['time_mon'])
        targetSamples['OS_new'] = np.where((targetSamples['time_mon'] < cutoff) & (targetSamples['OS'] == 1), 1, 0)

        df_exp=omicData.loc[omicData.index.isin(targetSamples['SampleID']), geneNames()]
        df_pro = omicData.loc[omicData.index.isin(targetSamples['SampleID']), [p + '_Y' for p in proNames()]]
        targetSamples=targetSamples.loc[df_pro.index]

        if (df_pro.shape[0] > 0) & (targetSamples[survival_type].sum() > 1):
            df_survResult = runSurvTest_GnP(df_exp, df_pro, marker, targetSamples)
            df_survResult['status']=status

            SurvResult.append(df_survResult)

SurvResult = pd.concat(SurvResult)
SurvResult.to_csv(dir_output + 'Table_Survivalpertumorperprotein_with95CI_sets_OSyear_' + str(n_year) + '_' +
                  ''.join(dataset) + '_' + target_tumor + '_markers' + '.txt', sep='\t', index=False)
print ('THE END')

# Fig 4B, Waterfall plot of proteins with significant cox-regression pvalue<0.05
i='ER'
s='positive'

df_plot=SurvResult[(SurvResult['tumortype']==i) & (SurvResult['status']==s)]
df_plot = df_plot[df_plot['coxp_pro'] < 0.05]
print(df_plot.shape)
print(df_plot)


plotCoxReg(df_plot, i + '_' + s, ['pro', 'gene'], ['Protein', 'RNA'], ['c', 'gold'], 2, FDR=False, ratio=[3,1])
#         plt.savefig(dir_output + 'Subplots_scatterbar_Coxregression_HR_RNAvsPro_' + i + '_' + s + '_pval2' + '.png', format='png',
#                     dpi=600)
#         plt.savefig(dir_output + 'Subplots_scatterbar_Coxregression_HR_RNAvsPro_' + i + '_' + s + '_pval2'+ '.pdf', format='pdf',
#                     dpi=600)
#         plt.close()

# Fig 4C, KMplot testing (Unknown samples)

un_exp=pd.read_csv(dir_input + 'X_data_batch_13995x4256_unknowdata.txt', sep="\t", index_col=[0]).T
un_pro=pd.read_csv('/home/CBBI/tsaih/Research/Model_Xnorm13995_new/Prediction_TCGAunknownsamples/' +
                   'ProPredictfromtrainedmodel_TCGA4256_Xunnorm.txt', sep="\t", index_col=[0]).T

un_targetSamples=clinicData[clinicData['DataSets'].isin(['Unknown_wRPPA', 'Unknown_woRPPA'])]
un_targetSamples.index=un_targetSamples['SampleID']

dir_output_KM=dir_output + 'KMplot_markers/'
from pathlib import Path
Path(dir_output_KM).mkdir(parents=True, exist_ok=True)

target_tumor='BRCA'
marker='ER'
status='positive'
proName='P53'
geneName='TP53'

un_targetSamplesSpec=un_targetSamples[(un_targetSamples['SampleTypeName']=='Primary Tumor')&
                                          (un_targetSamples['Disease_short']==target_tumor) &
                                          (un_targetSamples[marker+'_status']==status)][['SampleID', 'OS', 'OS.time']]
un_targetSamplesSpec = un_targetSamplesSpec.dropna()
un_targetSamplesSpec['time_mon'] = un_targetSamplesSpec['OS.time'] / 30
un_targetSamplesSpec.index = un_targetSamplesSpec['SampleID']

# Five year survival
un_targetSamplesSpec['time_mon_new'] = np.where(un_targetSamplesSpec['time_mon'] > cutoff, cutoff,
                                                    un_targetSamplesSpec['time_mon'])
un_targetSamplesSpec['OS_new'] = np.where(
        (un_targetSamplesSpec['time_mon'] < cutoff) & (un_targetSamplesSpec['OS'] == 1), 1, 0)
un_expTarget=un_exp.loc[un_exp.index.isin(un_targetSamplesSpec['SampleID']), geneNames()]
un_proTarget = un_pro.loc[un_pro.index.isin(un_targetSamplesSpec['SampleID']),  proNames()]
un_targetSamplesSpec=un_targetSamplesSpec.loc[un_proTarget.index]

# Predicted protein abundance
df = pd.merge(un_targetSamplesSpec, un_proTarget, left_index=True, right_index=True)

a=KMplot(df[proName], df[['SampleID', 'OS_new', 'time_mon_new']], 'OS_new', target_tumor+'('+marker+'_'+status +')', '\n'+'Protein:'+proName,
           color_high='red', color_low='blue', linestyle_high='-', linestyle_low='-')
plt.ylim(0.6, 1.05)
plt.tight_layout()

# Gene expression
df = pd.merge(un_targetSamplesSpec, un_expTarget, left_index=True, right_index=True)
a=KMplot(df[geneName], df[['SampleID', 'OS_new', 'time_mon_new']], 'OS_new', target_tumor + '(' + marker + '_' + status + ')',
           '\n' + 'Gene:' + geneName, color_high='red', color_low='blue', linestyle_high='-', linestyle_low='-')
plt.ylim(0.6, 1.05)
plt.tight_layout()