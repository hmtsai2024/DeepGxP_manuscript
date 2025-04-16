library(clusterProfiler)
library(ggplot2)
library(DOSE)
library(dplyr)
library(stringr)
library(msigdbr)
library(org.Hs.eg.db)

m_h<- msigdbr(species = "Homo sapiens", category = "H") %>%
  dplyr::select(gs_name, entrez_gene)
m_c1<- msigdbr(species = "Homo sapiens", category = "C1") %>%
  dplyr::select(gs_name, entrez_gene)
m_kegg <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "KEGG") %>%
  dplyr::select(gs_name, entrez_gene)
m_reactome <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "REACTOME") %>%
  dplyr::select(gs_name, entrez_gene)
m_cgp <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CGP") %>%
  dplyr::select(gs_name, entrez_gene)
m_c3 <- msigdbr(species = "Homo sapiens", category = "C3") %>%
  dplyr::select(gs_name, entrez_gene)
m_c4 <- msigdbr(species = "Homo sapiens", category = "C4") %>%
  dplyr::select(gs_name, entrez_gene)
m_gobp <- msigdbr(species = "Homo sapiens", category = "C5",subcategory = "BP") %>%
  dplyr::select(gs_name, entrez_gene)
m_gocc <- msigdbr(species = "Homo sapiens", category = "C5",subcategory = "CC") %>%
  dplyr::select(gs_name, entrez_gene)
m_gomf <- msigdbr(species = "Homo sapiens", category = "C5",subcategory = "MF") %>%
  dplyr::select(gs_name, entrez_gene)
m_hpo <- msigdbr(species = "Homo sapiens", category = "C5",subcategory = "HPO") %>%
  dplyr::select(gs_name, entrez_gene)
m_c6 <- msigdbr(species = "Homo sapiens", category = "C6") %>%
  dplyr::select(gs_name, entrez_gene)
m_c7 <- msigdbr(species = "Homo sapiens", category = "C7") %>%
  dplyr::select(gs_name, entrez_gene)
m_c8<- msigdbr(species = "Homo sapiens", category = "C8") %>%
  dplyr::select(gs_name, entrez_gene)

runORA <- function(pathway, gene, universe=NULL){
  if (pathway=='GOMF'){
    ck <- enrichGO(gene, OrgDb='org.Hs.eg.db', universe=universe, ont = "MF", readable=TRUE, pvalueCutoff=1, qvalueCutoff=1)
  } else if (pathway=='GOBP'){
    ck <- enrichGO(gene, OrgDb='org.Hs.eg.db', universe=universe, ont = "BP", readable=TRUE, pvalueCutoff=1, qvalueCutoff=1)
  } else if(pathway=='GOCC'){
    ck <- enrichGO(gene, OrgDb='org.Hs.eg.db', universe=universe, ont = "CC", readable=TRUE, pvalueCutoff=1, qvalueCutoff=1)
  } else if (pathway=='Hallmark'){
    ck <- enricher(gene, universe=universe, TERM2GENE=m_h, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='Reactome'){
    ck <- enricher(gene, universe=universe, TERM2GENE=m_reactome, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='KEGG'){
    ck <- enricher(gene, universe=universe, TERM2GENE=m_kegg, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='C5_hpo'){
    ck <- enricher(gene, universe=universe, TERM2GENE=m_hpo, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='C1'){
    ck <- enricher(gene,  universe=universe, TERM2GENE=m_c1, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='C6'){
    ck <- enricher(gene, universe=universe, TERM2GENE=m_c6, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='C7'){
    ck <- enricher(gene,  universe=universe, TERM2GENE=m_c7, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='C8'){
    ck <- enricher(gene,  universe=universe, TERM2GENE=m_c8, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='C3'){
    ck <- enricher(gene,  universe=universe, TERM2GENE=m_c3, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='C4'){
    ck <- enricher(gene,  universe=universe, TERM2GENE=m_c4, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  } else if (pathway=='C5_cgp'){
    ck <- enricher(gene,  universe=universe, TERM2GENE=m_cgp, pvalueCutoff=1, qvalueCutoff=1)
    if (is.null(ck)==FALSE){
      ck <- setReadable(ck, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
    }
  }
  return (ck)
}


get_significant_genes <- function(df, direction, target_col){
  if (direction == 'mix'){
    df_sel <- df[abs(df[target_col]) >= 1.96,] #absolute 1.96
  } else if (direction == 'pos'){
    df_sel <- df[df[target_col] >= 1.96,] #positive
  } else if (direction == 'neg'){
    df_sel <- df[df[target_col] <= -1.96,] #negative
  }
}

# Remove redundant terms by kappa pvalue
runKappa <- function(df, pvalue, kappa){
  df=df[(df$pvalue<pvalue),]
  
  if (nrow(df)>0){
    rownames(df)=NULL
    
    # Extract the number before the slash and create a new column B
    df_plot <- df %>%
      mutate(GSsize = str_extract(BgRatio, "^[^/]+")) %>%
      mutate(GSsize = as.numeric(GSsize))
    df_plot$Description_up=toupper(df_plot$Description)
    df_plot$Description_up=paste('GOBP_', df_plot$Description_up, sep='')
    df_plot$Description_up=gsub(" ", "_", df_plot$Description_up)
    
    kappa_sel=kappa[kappa$gs1 %in% df_plot$Description_up & kappa$gs2 %in% df_plot$Description_up,]
    kappa_nonSig=setdiff(df_plot$Description_up, unique(append(kappa_sel$gs1, kappa_sel$gs2)))
    
    #clustering (start)
    # Create an empty list to store the clusters
    clusters <- list()
    
    # Iterate over the filtered dataframe to identify clusters
    repGS <- character(0)
    if (nrow(kappa_sel)>0){
      for (index in 1:nrow(kappa_sel)) {
        #print(index)
        gene_set1 <- kappa_sel[index, 'gs1']
        gene_set2 <- kappa_sel[index, 'gs2']
        
        
        # Check if gene_set1 or gene_set2 is already present in a cluster
        gene_set1_cluster <- NULL
        gene_set2_cluster <- NULL
        for (cluster_id in seq_along(clusters)) {
          cluster <- clusters[[cluster_id]]
          #print(cluster)
          if (gene_set1 %in% cluster) {
            gene_set1_cluster <- cluster_id
          }
          if (gene_set2 %in% cluster) {
            gene_set2_cluster <- cluster_id
          }
        }
        
        # Merge clusters if both gene_set1 and gene_set2 belong to different clusters
        if (!is.null(gene_set1_cluster) && !is.null(gene_set2_cluster) && gene_set1_cluster != gene_set2_cluster) {
          clusters[[gene_set1_cluster]] <- union(clusters[[gene_set1_cluster]], clusters[[gene_set2_cluster]])
          clusters <- clusters[-gene_set2_cluster]
        }
        
        # Add gene_set1 and gene_set2 to an existing cluster
        else if (!is.null(gene_set1_cluster) && is.null(gene_set2_cluster)) {
          clusters[[gene_set1_cluster]] <- union(clusters[[gene_set1_cluster]], gene_set2)
        }
        
        # Add gene_set2 to an existing cluster
        else if (is.null(gene_set1_cluster) && !is.null(gene_set2_cluster)) {
          clusters[[gene_set2_cluster]] <- union(clusters[[gene_set2_cluster]], gene_set1)
        }
        
        # Create a new cluster with gene_set1 and gene_set2
        else if (is.null(gene_set1_cluster) && is.null(gene_set2_cluster)) {
          new_cluster_id <- length(clusters) + 1
          clusters[[new_cluster_id]] <- c(gene_set1, gene_set2)
        }
      }
      
      # Print the clusters
      for (cluster_id in seq_along(clusters)) {
        cluster <- clusters[[cluster_id]]
        #print(paste0("Cluster ", cluster_id, ": ", cluster))
      }
      
      # Create a DataFrame from the list of lists
      #clusters_df <- do.call(rbind, cluster_data)
      cluster_df=data.frame()
      for (cluster_id in seq_along(clusters)) {
        cluster <- clusters[[cluster_id]]
        df_cluster=data.frame(ClusterID=cluster_id,
                              GeneSet=paste(cluster, collapse = '/'))
        cluster_df=rbind(cluster_df, df_cluster)
      }
      
      repGS <- c()
      for (i in 1:nrow(cluster_df)) {
        clusters <- strsplit(cluster_df[i, 'GeneSet'], "/")[[1]]
        #clusters <- trimws(clusters)
        
        targetRows <- kappa_sel[(kappa_sel$gs1 %in% clusters) & kappa_sel$gs2 %in% clusters, ]
        
        # Create an empty list to store the mean k values for each gene set
        mean_k_values <- list()
        
        for (cluster in clusters) {
          #print(cluster)
          # Filter the DataFrame for rows where the gene set is present in either 'gs1' or 'gs2'
          targetRows_cluster <- targetRows[targetRows$gs1 == cluster | targetRows$gs2 == cluster, ]
          
          # Calculate the mean k value for the gene set
          mean_k <- mean(targetRows_cluster$kstat)
          
          # Store the mean k value in the list
          mean_k_values[[cluster]] <- mean_k
        }
        
        # Create a DataFrame from the list
        mean_k_df <- data.frame('GeneSet' = names(mean_k_values), 'Meank' = unlist(mean_k_values))
        
        #mean_k_df <- merge(mean_k_df, GS, by.x = 'GeneSet', by.y = 'gs_name')
        #mean_k_df=merge(mean_k_df, df_plot[, c(11,targetCol[1])], by.x='GeneSet', by.y='Description_up')
        
        mean_k_df=merge(mean_k_df, df_plot, by.x='GeneSet', by.y='Description_up')
        
        
        #select lowest p
        max_value=min(mean_k_df$pvalue)
        mean_k_df_sel=mean_k_df[mean_k_df$pvalue==max_value,]
        
        #if same number of protien, select largest kappa
        if (nrow(mean_k_df_sel)>1){
          #old
          max_value <- max(mean_k_df_sel$Meank)
          mean_k_df_sel=mean_k_df_sel[mean_k_df_sel$Meank==max_value,]
          
          #if same kappa, select geneset size
          if (nrow(mean_k_df_sel)>1){
            max_value <- max(mean_k_df_sel$GSsize)
            mean_k_df_sel=mean_k_df_sel[mean_k_df_sel$GSsize==max_value,]
            
            if (nrow(mean_k_df_sel)>1){
              mean_k_df_sel=mean_k_df_sel[1,]
              
            }
          }
        }
        repGS[i] <- mean_k_df_sel[, 'GeneSet']
      }
      
      cluster_df$RepresentativeGeneSet <- repGS
    }
    
    #clustering (end)
    df_reduce=df_plot[(df_plot$Description_up %in%repGS)|(df_plot$Description_up %in% kappa_nonSig),]
  }
  return(df_reduce)
}

PlotDots_single <- function(df, proName, pathway, direction, abundance){
  minP <- min(df$pvalue)
  maxP <- max(df$pvalue)
  p<-ggplot(df, aes(x = GeneRatio_num, y = Description)) + 
    geom_point(aes(color = pvalue, size=Count)) +
    theme_bw(base_size = 14) +
    scale_colour_gradient(limits=c(0, maxP), low="red", high="blue") +
    ylab(NULL) +
    ggtitle(paste(proName, "-", pathway, ":", abundance , " (", direction, ") " , sep=''))
}

kappa_data <- read.delim("../../kappa/Table_KappaSigPairs_pvalue.txt", sep = "\t", header = TRUE, stringsAsFactors = FALSE)
gene_mapping <- read.delim("../../data/EB++GeneAnnotationEntriezID_mapping2ourGenes2.txt", sep = '\t', header = TRUE, stringsAsFactors = FALSE)

# Define all possible tumor types, analysis directions, pathways, and abundance categories
tumor_types=c("BLCA", "BRCA", "CESC", "COAD", "ESCA", "GBM", "HNSC", "KICH", "KIRC", "KIRP",
              "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV" ,"PAAD", "PanCan", "PCPG", "PRAD",
              "READ", "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC")
analysis_directions <- c('pos', 'neg')
pathway_categories <- c('GOMF', "Reactome", "C5_hpo", 'KEGG', 'C5_cgp', 'C3', 'C4', 'C6', 'C7', 'C8', 'C1', 'GOBP', 'GOCC', 'Hallmark')
abundance_categories <- c('high')

# Run one example for each list
tumor_types <- c("PanCan")
analysis_directions <- c('pos')
pathway_categories <- c('Hallmark')
abundance_categories <- c('high')

# Generate gene set enrichment results
for (abundance_category in abundance_categories) {
  target_column <- paste(abundance_category, '_zscore', sep = '')
  for (analysis_direction in analysis_directions) {
    tumor_counter <- 0
    for (tumor_type in tumor_types) {
      tumor_counter <- tumor_counter + 1
      print(paste('Tumor :', tumor_counter, sep=''))
      
      # Define directory paths for input and output
      input_dir <- paste("../../Results/Model_Xnorm13995_new/GradientDistribution_HighvsLow_proabd/", tumor_type, "/", sep = "")
      output_dir <- paste(input_dir, "ORA_singleProtein_example", '/', sep = '')
      dir.create(file.path(output_dir), showWarnings = TRUE)
      
      output_dir <- paste(output_dir, abundance_category, '/', sep = '')
      dir.create(file.path(output_dir), showWarnings = TRUE)
      
      output_dir <- paste(output_dir, analysis_direction, '/', sep = '')
      dir.create(file.path(output_dir), showWarnings = TRUE)
      
      # Set working directory to the input directory and get the file list
      file_list <- list.files(input_dir, full.names = TRUE, recursive = FALSE)
      file_info <- file.info(file_list)
      # Keep only files (not directories)
      file_list <- file_list[!file_info$isdir]
      file_list <- file_list[!grepl('summary', basename(file_list))]
      print(length(file_list))
      
      for (i in 1:length(file_list)) {
        print(i)
        # Show only one protein as an example (Cyclin B1)
        if (i==53){
          # Extract protein name from filename
          filename <- basename(file_list[i])
          protein_name <- sub("gradient_mean_HvsL_(.*)\\.txt", "\\1", filename)
          print(protein_name)
          # Read average IG data
          protein_data <- paste(input_dir, 'gradient_mean_HvsL_', protein_name, '.txt', sep = '')
          protein_data <- read.delim(protein_data, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
          significant_genes <- get_significant_genes(protein_data, analysis_direction, target_column)
          significant_genes <- merge(significant_genes, gene_mapping, by = 'X')
          significant_genes <- as.character(significant_genes[,'EntrezID'])
          
          # Loop through each pathway category
          for (pathway_category in pathway_categories) {
            enriched_pathways <- runORA(pathway = pathway_category, gene = significant_genes)
            
            if (!is.null(enriched_pathways)) {
              enriched_data <- data.frame(tumor_type = tumor_type, protein_name = protein_name, pathway_category = pathway_category, enriched_pathways)
              rownames(enriched_data) <- NULL
              write.table(enriched_data, file = paste(output_dir, "Table_enrichedPaths_", pathway_category, "_", analysis_direction, "_", protein_name, "_", tumor_type, ".txt", sep = ""), 
                          quote = FALSE, row.names = FALSE, col.names = TRUE, sep = "\t")
              print(enriched_data[1:5,])
              
              # Run kappa clustering to remove redundant pathways
              reduced_data <- runKappa(df = enriched_data, pvalue = 0.05, kappa = kappa_data)
              write.table(reduced_data, file = paste(output_dir, "Table_enrichedPaths_", pathway_category, "_", analysis_direction, "_", protein_name, "_", tumor_type, "_removeRedundantKappa.txt", sep = ""),
                          quote = FALSE, row.names = FALSE, col.names = TRUE, sep = "\t")
              
              # Select top 5 enriched pathways by p-value for plotting
              top_enriched_data <- reduced_data %>% slice_min(order_by = pvalue, n = 5)
              top_enriched_data$GeneRatio_num <- parse_ratio(top_enriched_data$GeneRatio)
              top_enriched_data$Description <- factor(top_enriched_data$Description, levels = unique(top_enriched_data$Description[order(top_enriched_data$GeneRatio_num)]))
              plot <- PlotDots_single(df = top_enriched_data, proName = protein_name, pathway = pathway_category, direction = analysis_direction, abundance = abundance_category)
              pdf(paste(output_dir, "Dotplot_", pathway_category, '_', analysis_direction, "_", protein_name, "_", tumor_type, '.pdf', sep = ""), width = 12, height = 8)
              print(plot)
              dev.off()
              print(plot)
            }
          }
        }
      }
    }
  }
}