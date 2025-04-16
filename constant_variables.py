import pandas as pd
import numpy as np

def samples2tumortypes():
    df=pd.read_csv('/home/CBBI/tsaih/data/' + 'sample_summary_groups.txt', sep='\t')
    return df

def CNNResult():
    dir = '/home/CBBI/tsaih/Research/Model_Xnorm13995_new/'
    tmpName = '1Layer1D_1LayerDense_conv1024_kernelstride50_dense512_batch254_normX_biasFalse'
    df= pd.read_csv(dir + tmpName + '_corr_results_' + str(1) + '.txt', sep='\t')
    return df

def PhosphoPairs():
    dir = '/home/CBBI/tsaih/data/'
    df = pd.read_csv(dir + 'PhosphoMatchedPairs.txt', sep='\t')
    df.columns=['proName_total', 'proName_phosph', 'geneName']
    return df

def CorrRNAnPro():
    dir = '/home/CBBI/tsaih/data/'
    df=pd.read_csv(dir+'CorResult_PairedProtGene_selOne2One.txt', sep='\t')
    df['proName']=df['proName'].str.upper()
    return df

def sampleTrain():
    dir = '/home/CBBI/tsaih/data/'
    ls = pd.read_csv(dir + 'Y_data_187x6813_train.txt', sep='\t', index_col=[0]).columns.tolist()
    return ls

def sampleVal():
    dir = '/home/CBBI/tsaih/data/'
    ls = pd.read_csv(dir + 'Y_data_187x6813_val.txt', sep='\t', index_col=[0]).columns.tolist()
    return ls

def sampleTest():
    dir = '/home/CBBI/tsaih/data/'
    ls = pd.read_csv(dir + 'Y_data_187x6813_test.txt', sep='\t', index_col=[0]).columns.tolist()
    return ls

def sampleUnknown(RPPA):
    dir_data = '/home/CBBI/tsaih/data/'
    dir_input = '/home/CBBI/tsaih/Research/' + 'Prediction_newsamples/'

    df = pd.read_csv(dir_input + 'ProPredictfromtrainedmodel_TCGA4256.txt', sep='\t', index_col=[0])
    true_pro = pd.read_csv(dir_data + 'TCGA-PANCAN32-L4.csv')
    true_pro.index = true_pro.Sample_ID.str.slice(stop=15)
    true_pro.index = true_pro.index.str.replace('-', '.')

    if RPPA:
        samples = list(set(df.columns) & set(true_pro.index))  # 427
    else:
        samples = list(set(df.columns) - set(true_pro.index))  # 3829
    return samples

def geneNames():
    dir = '/home/CBBI/tsaih/data/'
    ls = pd.read_csv(dir + 'X_data_batch_13995x6813_val.txt', sep='\t', index_col=[0]).index.tolist()
    return ls

def proNames():
    dir = '/home/CBBI/tsaih/data/'
    ls = pd.read_csv(dir + 'Y_data_187x6813_val.txt', sep='\t', index_col=[0]).index.tolist()
    return ls

def Tumortype2Short():
    dir = '/home/CBBI/tsaih/data/'
    df = pd.read_csv(dir + 'Tumortypes_Abbreviation.txt', sep='\t')
    return df

def ProCorperTumor():
    dir= '/home/CBBI/tsaih/data/'
    df=pd.read_csv(dir + 'Original_correlation_proteinvsrna_pertumornpan_true.txt', sep='\t')
    return df

def myData(data, dataset):
    dir = '/home/CBBI/tsaih/data/'
    dir_model='/home/CBBI/tsaih/Research/Model_Xnorm13995_new/'
    tmpName = '1Layer1D_1LayerDense_conv1024_kernelstride50_dense512_batch254_normX_biasFalse'
    if data=='X':
        df=pd.read_csv(dir + 'X_data_batch_13995x6813_' + dataset + '.txt', sep='\t', index_col=[0]).T
    elif data=='Y':
        df = pd.read_csv(dir + 'Y_data_187x6813_' + dataset + '.txt', sep='\t', index_col=[0]).T
    elif data=='pred':
        df = pd.read_csv(dir_model + tmpName + '_predfrom' + dataset + '_' + str(1) + '.txt', sep='\t', index_col=[0])
        df.columns = proNames()
        if dataset=='train':
            df.index=sampleTrain()
        elif dataset=='val':
            df.index = sampleVal()
        elif dataset=='test':
            df.index = sampleTest()
    return df

def fullData():
    dir_pred = '/home/CBBI/tsaih/Research/Model_Xnorm13995_new/'
    # dir='/home/CBBI/tsaih/data/'
    # tmpName = '1Layer1D_1LayerDense_conv1024_kernelstride50_dense512_batch254_normX_biasFalse'
    #
    # Xtrain = pd.read_csv(dir + 'X_data_batch_13995x6813_train.txt', sep='\t', index_col=[0])
    # Xval=pd.read_csv(dir + 'X_data_batch_13995x6813_val.txt', sep='\t', index_col=[0])
    # Xtest = pd.read_csv(dir + 'X_data_batch_13995x6813_test.txt', sep='\t', index_col=[0])
    # X=pd.merge(pd.merge(Xtrain, Xval, left_index=True, right_index=True), Xtest, left_index=True, right_index=True).T
    # del Xtrain, Xval, Xtest
    #
    # # norm_constant = np.loadtxt(dir + "NormalizationConstantfromTrainingSet.txt", delimiter='\t')
    # # X_norm = X / norm_constant
    #
    # Xtrain = pd.read_csv(dir + 'X_data_batch_13995x6813_train_z.txt', sep='\t', index_col=[0])
    # Xval=pd.read_csv(dir + 'X_data_batch_13995x6813_val_z.txt', sep='\t', index_col=[0])
    # Xtest = pd.read_csv(dir + 'X_data_batch_13995x6813_test_z.txt', sep='\t', index_col=[0])
    # X_norm=pd.merge(pd.merge(Xtrain, Xval, left_index=True, right_index=True), Xtest, left_index=True, right_index=True).T
    # del Xtrain, Xval, Xtest
    #
    # Ytrain = pd.read_csv(dir + 'Y_data_187x6813_train.txt', sep='\t', index_col=[0])
    # Yval=pd.read_csv(dir+ 'Y_data_187x6813_val.txt', sep='\t', index_col=[0])
    # Ytest = pd.read_csv(dir + 'Y_data_187x6813_test.txt', sep='\t', index_col=[0])
    # Y = pd.merge(pd.merge(Ytrain, Yval, left_index=True, right_index=True), Ytest, left_index=True, right_index=True).T
    # del Ytrain, Yval, Ytest
    #
    # # prediction data
    # train_pred = pd.read_csv(dir_pred + tmpName + '_predfromtrain_' + str(1) + '.txt', sep='\t', index_col=[0])
    # val_pred = pd.read_csv(dir_pred + tmpName + '_predfromval_' + str(1) + '.txt', sep='\t', index_col=[0])
    # test_pred = pd.read_csv(dir_pred + tmpName + '_predfromtest_' + str(1) + '.txt', sep='\t', index_col=[0])
    # pred=pd.concat([train_pred, val_pred, test_pred])
    # pred.index=Y.index
    # pred.columns=Y.columns
    #
    # X.to_csv(dir+'X_data_batch_13995x6813_all.txt', sep='\t', index=True)
    # X_norm.to_csv(dir + 'X_data_batch_13995x6813_all_Xnorm.txt', sep='\t', index=True)
    # Y.to_csv(dir + 'Y_data_187x6813_all.txt', sep='\t', index=True)
    # pred.to_csv(dir_pred + tmpName+ 'pred_data_187x6813_all', sep='\t', index=True)
    #
    # Y=Y.add_suffix('_Y')
    # pred = pred.add_suffix('_pred')
    # X_norm = X_norm.add_suffix('_norm')
    #
    # df=pd.merge(pd.merge(pd.merge(X, X_norm, left_index=True, right_index=True), Y, left_index=True, right_index=True), pred, left_index=True, right_index=True)
    # df.to_csv(dir_pred + 'XYprednew_fulldata_6813.txt', sep='\t', index=True)

    df=pd.read_csv(dir_pred + 'XYprednew_fulldata_6813' + '.txt', sep='\t', index_col=[0])
    return df

def ClinicalData():
    dir_data = '/home/CBBI/tsaih/data/'
    # clinical = pd.read_csv(dir_data + 'TCGA_phenotype_denseDataOnlyDownload.tsv', sep="\t")
    # clinical['sample']=clinical['sample'].str.replace('-', '.')
    # tumortypeshort = pd.read_csv(dir_data + 'Tumortypes_Abbreviation.txt', sep='\t')
    # clinical = pd.merge(clinical, tumortypeshort, on='_primary_disease')
    # clinical.columns=['SampleID', 'SampleTypeID', 'SampleTypeName', 'Disease_long', 'Disease_short']
    # clinical['DataSets']=np.where(clinical['SampleID'].isin(sampleTrain()), 'Train', 'Others')
    # clinical['DataSets'] = np.where(clinical['SampleID'].isin(sampleVal()), 'Val', clinical['DataSets'])
    # clinical['DataSets'] = np.where(clinical['SampleID'].isin(sampleTest()), 'Test', clinical['DataSets'])
    # clinical['DataSets'] = np.where(clinical['SampleID'].isin(sampleUnknown(RPPA=True)), 'Unknown_wRPPA', clinical['DataSets'])
    # clinical['DataSets'] = np.where(clinical['SampleID'].isin(sampleUnknown(RPPA=False)), 'Unknown_woRPPA', clinical['DataSets'])
    #
    # #Suvival data
    # Surv=pd.read_csv(dir_data + 'Survival_SupplementalTable_S1_20171025_xena_sp', sep="\t")
    # Surv['sample'] = Surv['sample'].str.replace('-', '.')
    #
    # clinical=pd.merge(clinical, Surv, left_on='SampleID', right_on='sample', how='outer')
    # clinical['Note']=np.where(clinical['SampleID']=='TCGA.23.1023.01', 'SampleID-01-butSampleTypeID-02', '')
    #
    # #subtype info
    # cancersubtypes = pd.read_csv(dir_data + '1-s2.0-S0092867418303593-mmc1.txt', sep='\t')
    # cancersubtypes['SAMPLE_BARCODE'] = cancersubtypes['SAMPLE_BARCODE'].str.replace('-', '.')
    # clinical = pd.merge(clinical, cancersubtypes, left_on='SampleID', right_on='SAMPLE_BARCODE', how='outer')
    #
    # clinical=clinical[['SampleID', '_PATIENT', 'SampleTypeID', 'SampleTypeName', 'Disease_short',
    #                     'Disease_long', 'DataSets', 'SUBTYPE', 'OS', 'OS.time',
    #                     'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI',
    #                     'PFI.time',  'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage',
    #                     'clinical_stage', 'histological_type', 'histological_grade', 'initial_pathologic_dx_year', 'menopause_status',
    #                     'birth_days_to', 'vital_status', 'tumor_status', 'last_contact_days_to', 'death_days_to',
    #                     'cause_of_death', 'new_tumor_event_type', 'new_tumor_event_site', 'new_tumor_event_site_other','new_tumor_event_dx_days_to',
    #                     'treatment_outcome_first_course', 'margin_status', 'residual_tumor', 'Redaction', 'Note']]
    # clinical=clinical.sort_values(by=['Disease_short', 'SUBTYPE', 'DataSets'])
    # clinical['_PATIENT']=clinical['_PATIENT'].str.replace('-', '.')
    # clinical.to_csv(dir_data+'Table_ClinicalDataALLsamples.txt', sep='\t', index=False)
    ###merge with BRCA subtypes
    # clinical = pd.read_csv(dir_data + 'Table_ClinicalDataALLsamples.txt', sep="\t")
    # usecols = ['patient.bcr_patient_barcode',
    #            'patient.lab_proc_her2_neu_immunohistochemistry_receptor_status',
    #            'patient.breast_carcinoma_estrogen_receptor_status',
    #            'patient.breast_carcinoma_progesterone_receptor_status'
    #            ]
    # clinicBRCA = pd.read_csv(dir_data + 'BRCA.merged_only_clinical_clin_format.txt', sep='\t', index_col=[0]).T[usecols]
    # clinicBRCA.columns = ['_PATIENT', 'HER2_status', 'ER_status', 'PR_status']
    # clinicBRCA['_PATIENT'] = clinicBRCA['_PATIENT'].str.upper()
    # clinicBRCA['_PATIENT'] = clinicBRCA['_PATIENT'].str.replace('-', '.')
    # clinicData = pd.merge(clinical, clinicBRCA, on='_PATIENT', how='outer')
    # clinicData.to_csv(dir_data + 'Table_ClinicalDataALLsamples.txt', sep='\t', index=False)

    df=pd.read_csv(dir_data+'Table_ClinicalDataALLsamples.txt', sep='\t')
    return df

def ProteinGroups():
    dir_data = '/home/CBBI/tsaih/data/'
    df = pd.read_csv(dir_data + 'ProteinGroupsSummary.txt', sep='\t')
    df = df[['rppa', 'ProteinGroups', 'DTO Family', 'DTO Family Ext.']]
    return df