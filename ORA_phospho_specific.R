library(clusterProfiler)
library(ggplot2)
library(DOSE)
library(ggnewscale)
library(forcats)
library(GOSemSim)
library(VennDiagram)
library(gplots)
library(dplyr)
library(org.Hs.eg.db)
library(msigdbr)

source("functions.R")

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

gene_mapping <- read.delim("../../data/EB++GeneAnnotationEntriezID_mapping2ourGenes2.txt", sep = '\t', header = TRUE, stringsAsFactors = FALSE)

tumortype='LUAD'
dir_input=paste("../../Results/Model_Xnorm13995_new/GradientDistribution_HighvsLow_proabd/", tumortype, "/", sep="")

if (tumortype=='PanCan'){
  dir_output=paste("../../Results/Model_Xnorm13995_new/GradientDistribution_HighvsLow_proabd/", "ORA_GenesoninPhospho_example/", sep='')
} else {
  dir_output=paste(dir_input, "ORA_GenesoninPhospho_example/", sep='')
  dir.create(file.path(dir_output), showWarnings = T)
}

filelist <- list.files(path = dir_input, pattern = 'gradient_mean_.*\\.txt', full.names = TRUE)
filelist=filelist[-grep('gradient_mean_summary.txt', filelist)]
print(length(filelist))

if (tumortype=='PanCan'){
  CorPairpt=read.delim("../Results/Model_Xnorm13995_new/Correlation_TotalvsPhosph/Table_Correlation_TotalvsPhospho.txt", sep = "\t",header = T, stringsAsFactors = F )
  cutoff=median(CorPairpt$cor)
} else {
  #pertumor
  CorPairpt=read.delim("../../Results/Model_Xnorm13995_new/Correlation_TotalvsPhosph/Table_Correlation_TotalvsPhospho_perTumor.txt", sep = "\t",header = T, stringsAsFactors = F )
  CorPairpt=CorPairpt[CorPairpt$tumortype==tumortype,]
  CorPairpt$proName_phosph=gsub('_P', '_p', CorPairpt$proName_phosph)
  CorPairpt=CorPairpt[order(CorPairpt$cor, decreasing=TRUE),]
  rownames(CorPairpt)=NULL
  order=CorPairpt$proName_phosph
  cutoff=median(CorPairpt$cor)
}

H_total=CorPairpt[CorPairpt$cor>cutoff, 'proName_total']
H_phospho=CorPairpt[CorPairpt$cor>cutoff, 'proName_phosph']
L_total=CorPairpt[CorPairpt$cor<=cutoff, 'proName_total']
L_phospho=CorPairpt[CorPairpt$cor<=cutoff, 'proName_phosph']

phosphoMatched=read.delim("../../data/PhosphoMatchedPairs.txt", sep = "\t",header = T, stringsAsFactors = F )

directions=c('pos') #'neg'
pathways=c('Hallmark') #'GOMF', "Reactome", "C5_hpo", 'KEGG', 'C5_cgp', 'C3', 'C4', 'C6', 'C7', 'C8', 'C1', 'GOBP', 'GOCC'
abundances=c('high')
proteinTypes=c('Total', 'Phospho', 'PminusT')

for (abundance in abundances){
  dir_suboutput=paste(dir_output, abundance, '/', sep='')
  dir.create(file.path(dir_suboutput), showWarnings = T)
  targetCol=paste(abundance, '_zscore', sep='')
  
  df_count=data.frame()
  df_count_enrich=data.frame()
  for (direction in directions){
    dir_subsuboutput=paste(dir_suboutput, direction, '/', sep='')
    dir.create(file.path(dir_subsuboutput), showWarnings = T)
    done_proteins=list.files(dir_subsuboutput)
    
    n=0
    for (i in 1:nrow(phosphoMatched)){
      
      # Example results for EGFR vs EGFR p1068  
      n=n+1
      print(paste(direction, ':', n, sep=''))
      protName_t=phosphoMatched$total_proName[i]
      protName_p=phosphoMatched$rppa[i]
      
      if (protName_t %in% c('HER2')){
        #using clusterprofiler
        Phosph=paste(dir_input, 'gradient_mean_HvsL_', protName_p,'.txt', sep='')
        Phosph=read.delim(Phosph, sep = "\t",header = T, stringsAsFactors = F )
        Phosph=get_significant_genes(Phosph, direction, targetCol)
        Phosph=merge(Phosph, gene_mapping, by='X')
        Phosph=as.character(Phosph[,'EntrezID'])
        
        Total=paste(dir_input, 'gradient_mean_HvsL_', protName_t,'.txt', sep='')
        Total=read.delim(Total, sep = "\t",header = T, stringsAsFactors = F )
        Total=get_significant_genes(Total, direction, targetCol)
        Total=merge(Total, gene_mapping, by='X')
        Total=as.character(Total[,'EntrezID'])
        
        gene=setdiff(Phosph, Total)
        
        df_num=data.frame(
          direction=direction,
          protName_p=protName_p,
          protName_t=protName_t,
          numCommon=length(intersect(Phosph, Total)),
          numP=length(Phosph),
          numT=length(Total)
        )
        if (protName_p %in% H_phospho){
          df_num$corGroup='H'
        } else if (protName_p %in% L_phospho){
          df_num$corGroup='L'
        }
        
        df_count=rbind(df_count, df_num)
        
        #--------runORA------------
        todofilename=paste("Table_overall_", direction, "_" , protName_p, "_enrichedPaths_noUniverse.txt", sep='')
        if (!todofilename %in% done_proteins){
          #print (paste('-----', pathway, sep=''))
          df_overall=data.frame()
          
          for (proteinType in proteinTypes){
            if (proteinType=='Total'){
              geneList=Total
              targetName=paste(protName_t,'_full',  sep='')
            } else if (proteinType=='Phospho'){
              geneList=Phosph
              targetName=paste(protName_p,'_full',  sep='')
            } else if (proteinType=='PminusT'){
              geneList=gene
              targetName=paste(protName_p,'_PminusT',  sep='')
            }
            
            for (pathway in pathways){
              ck=runORA(pathway=pathway, gene=geneList)
              if (is.null(ck)==FALSE){
                ck2=data.frame(proName=targetName, matchedTotal=protName_t, Cluster=pathway, ck)
                rownames(ck2)=NULL
                ck2 %>%
                  group_by(proName) %>%
                  slice_min(order_by = -pvalue, n = 5) %>%
                  ungroup() %>%
                  print()
                df_overall=rbind(df_overall, ck2)
              }
            }
          }
          write.table(df_overall,file=paste(dir_subsuboutput, "Table_", "overall", "_",  direction, "_", protName_p, "_enrichedPaths_noUniverse" ,".txt", sep=""), quote=F,row.names=F,col.names=T,sep="\t") 
        }
      }
    }
  }
  write.table(df_count,file=paste(dir_suboutput, "Table_", "numGene_commonTotalPhosph", "_",  abundance,"_noUniverse.txt", sep=""), quote=F,row.names=F,col.names=T,sep="\t")
}
