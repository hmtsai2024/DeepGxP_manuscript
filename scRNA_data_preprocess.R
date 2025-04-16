options(future.globals.maxSize = 4 * 1024^3)

#Read input data
library(Seurat)
library(SeuratDisk)
library(reticulate)
library(SeuratData)
library(dplyr)
library(SeuratWrappers)
library(rsvd)
library(Rmagic)

save_data <- function(data_object, data_filename, file_suffix = "") {
  # Save to H5Seurat format
  output_filename <- paste0(data_filename, file_suffix, ".h5seurat")
  SaveH5Seurat(data_object, output_filename)
  # Convert to H5AD format
  Convert(output_filename, dest = "h5ad")
}

input_directory <- '/Data/'
setwd(input_directory)

# ------------------------PBMC---------------------------
# Downloaded from PBMC data (from https://atlas.fredhutch.org/nygc/multimodal-pbmc/)
pbmc_data <- LoadH5Seurat('multi.h5seurat')
data_filename <- file.path(input_directory, "multi")
data <- LoadH5Seurat(paste0(data_filename, ".h5seurat"))

# Define the timepoint for analysis (e.g., '0', '3', '7')
timepoint <- '0'
# Subset data based on the specified timepoint
pbmc_subset <- subset(x = data, subset = time == timepoint)

# ---RNA Data Processing
rna_data <- CreateSeuratObject(counts = pbmc_subset@assays$SCT@counts)
rna_data <- SCTransform(rna_data, assay = 'RNA', new.assay.name = 'SCT')
magic_result <- magic(t(GetAssayData(rna_data, assay = "SCT", slot = "data")))

# Create Seurat object from MAGIC-imputed data
rna_magic <- CreateSeuratObject(counts = t(magic_result$result))
rna_magic <- DietSeurat(rna_magic, counts = FALSE)

# ---Protein Data Processing
protein_data <- CreateSeuratObject(counts = pbmc_subset@assays$ADT@counts)
protein_data <- NormalizeData(protein_data, normalization.method = 'CLR', assay = "RNA", margin = 2)
protein_data <- DietSeurat(protein_data, counts = FALSE)

# -----
# Save Count Data for sciPENN
# RNA count data
rna_count_data <- CreateSeuratObject(counts = pbmc_subset@assays$SCT@counts)
rna_count_data <- DietSeurat(rna_count_data, counts = FALSE)

# Protein count data
protein_count_data <- CreateSeuratObject(counts = pbmc_subset@assays$ADT@counts)
protein_count_data <- DietSeurat(protein_count_data, counts = FALSE)

# Save data
save_data(rna_magic, data_filename, file_suffix = paste0("_time", timepoint, "_magic"))
save_data(protein_data, data_filename, file_suffix = paste0("_time", timepoint, "_protein_data"))
save_data(rna_count_data, data_filename, file_suffix = paste0("_time", timepoint, "_counts"))
save_data(protein_count_data, data_filename, file_suffix = paste0("_time", timepoint, "_protein_data_counts"))


# ------------------------H1N1---------------------------
# Load RNA data (H1N1)
rna_filename <- file.path(input_directory, "H1N1/gene_data")
Convert(paste0(rna_filename, ".h5ad"), dest = "h5seurat", overwrite = TRUE)
rna_h1n1 <- LoadH5Seurat(paste0(rna_filename, ".h5seurat"))

# Load protein data (H1N1)
protein_filename <- file.path(input_directory, "H1N1/protein_data")
Convert(paste0(protein_filename, ".h5ad"), dest = "h5seurat", overwrite = TRUE)
protein_h1n1 <- LoadH5Seurat(paste0(protein_filename, ".h5seurat"))

# ---------------------------------------------------
# ---RNA Data Processing
rna_h1n1 <- SCTransform(rna_h1n1, assay = 'RNA', new.assay.name = 'SCT')
magic_result <- magic(t(GetAssayData(rna_h1n1, assay = "SCT", slot = "data")))
rna_magic <- CreateSeuratObject(counts = t(magic_result$result))
rna_magic <- DietSeurat(rna_magic, counts = FALSE)

# ---Protein Data Processing
protein_h1n1 <- NormalizeData(protein_h1n1, normalization.method = 'CLR', assay = "RNA", margin = 2)
protein_h1n1 <- DietSeurat(protein_h1n1, counts = FALSE)

# -----
# Count Data for sciPENN
# RNA count data
rna_count <- CreateSeuratObject(counts = rna_h1n1@assays$RNA@counts)
rna_count <- DietSeurat(rna_count, counts = FALSE)

# Protein count data
protein_count <- CreateSeuratObject(counts = protein_h1n1@assays$RNA@counts)
protein_count <- DietSeurat(protein_count, counts = FALSE)

# Save data
save_data(rna_magic, rna_filename, file_suffix = "_magic")
save_data(protein_h1n1, protein_filename, file_suffix = "_protein_data")
save_data(rna_count, rna_filename, file_suffix = "_counts")
save_data(protein_count, protein_filename, file_suffix = "_counts")


# ------------------------CBMC (GSE100866)---------------------------
# Downloaded from https://satijalab.org/seurat/articles/multimodal_vignette---
cbmc.rna <- as.sparse(read.csv(file = paste(input_directory, "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz", sep=''),
                               sep = ",", header = TRUE, row.names = 1))
cbmc.rna <- CollapseSpeciesExpressionMatrix(cbmc.rna)
cbmc.adt <- as.sparse(read.csv(file = paste(input_directory, "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz", sep=''),
                               sep = ",", header = TRUE, row.names = 1))
all.equal(colnames(cbmc.rna), colnames(cbmc.adt))

data_filename <- paste0(input_directory, "GSE100866_CBMC_8K_13AB_")

#-----RNA data---
#perform SC normalization
rna <- CreateSeuratObject(counts = cbmc.rna)
rna <- SCTransform(rna, assay='RNA', new.assay.name = 'SCT')
magic_result <- magic(t(GetAssayData(rna, assay = "SCT", slot = "data")))
rna_magic <- CreateSeuratObject(counts=t(magic_result$result))
rna_magic <- DietSeurat(rna_magic, counts = FALSE)

#-----protein data-----
protein <- CreateSeuratObject(counts=cbmc.adt)
protein <- NormalizeData(protein, normalization.method = 'CLR', assay = "RNA", margin=2)
protein <- DietSeurat(protein, counts = FALSE)

#--count data---
#-----RNA data---
rna_count <- CreateSeuratObject(counts = cbmc.rna)
rna_count <- DietSeurat(rna_count, counts = FALSE)

#-----protein data-----
protein_count <- CreateSeuratObject(counts=cbmc.adt)
protein_count <- DietSeurat(protein_count, counts = FALSE)

# Save data
save_data(rna_magic, data_filename, file_suffix = "_magic")
save_data(protein, data_filename, file_suffix = "_protein_data")
save_data(rna_count, data_filename, file_suffix = "_counts")
save_data(protein_count, data_filename, file_suffix = "_protein_data_counts")

