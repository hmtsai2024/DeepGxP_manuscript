# DeepGxP_manuscript

## Summary
DeepGxP is a Python-based framework for predicting and interpreting protein abundance from gene expression profiles. The model was originally trained on bulk RNA-seq and RPPA protein data from TCGA tumors, focusing on tumor-associated proteins. With minor architectural adjustments, DeepGxP also performs well on single-cell data, such as CITE-seq, which simultaneously measures RNA and surface protein expression in the same cell.
This repository contains the code used in our manuscript, including model training, evaluation, and prediction. Preprocessing and visualization of single-cell RNA-seq were performed in R, and relevant R scripts are included where applicable.

## Data
* TCGA RNA-seq data: https://xenabrowser.net/datapages/<br />
* TCGA RPPA data: https://tcpaportal.org/tcpa/download.html<br />
* CCLE data: https://depmap.org/portal/<br />
* CITE-seq PBMC data: https://atlas.fredhutch.org/nygc/multimodal-pbmc/<br />
* CITE-seq CBMC data: https://satijalab.org/seurat/articles/multimodal_vignette.html (Seurat tutorial)<br />
* CITE-seq H1N1 data: https://doi.org/10.35092/yhjc.c.4753772<br />
For detailed information on input/output data preparation, please refer to the (Supp)Methods section of our paper.

## Code
Bulk cells:<br />
DeepGxP_model_bulk: Train and save the model for bulk RNA-seq and RPPA protein. <br />
CrossValidation_bulk_*: 10-fold cross-validation on DeeGxP and multiple deep learning and machine learning models. <br />
PredictUnknownSamples: Use DeepGxP to predict TCGA samples that do not have RPPA data measured. <br />

Single-cells: <br />
scRNA_data_preprocess_in_R: Preprocess CITE-seq data with default Seurat pipeline and perform MAGIC imputation. <br />
DeepGxP_model_scRNA: Train and save the model using CITE-seq RNA and protein (ADT) data. <br />
CrossValidation_scRNA_*: 10-fold cross-validation on DeeGxP and sciPENN model. <br />
IndependentValidation_scRNA_H1N1: Use DeepGxP on independent validation data. <br />

Shared: <br />
compute_integrated_gradients: Calculate integrated gradients (IG) values from the saved model for each protein on every input, i.e. gene x samples for each protein.<br />
calculate_normalized_integrated_gradients: Average IG values across target samples and do z-transformation.<br />
ORA_single_protein: Performs overrepresentation analysis (ORA) on predictor genes for individual proteins.<br />
ORA_phospho_specific: Performs ORA on phosphoproteins with matched total proteins, using only phospho-specific predictor genes (i.e., excluding genes shared with the corresponding total protein).<br />

