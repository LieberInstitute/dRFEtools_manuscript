## This script plots eQTLs based on expression imputation analysis.
##
## It is a port of a python notebook converted to R.
##
##################################################################

suppressPackageStartupMessages({
    library(tidyverse)
    library(ggpubr)
})

get_residualized_df <- function(){
    expr_file = paste0("../../../../../neuropsychiatric_analysis/_m/",
                       "residualized_expression.tsv")
    return(data.table::fread(expr_file) %>% column_to_rownames("V1"))
}
memRES <- memoise::memoise(get_residualized_df)

get_biomart_df <- function(){
    biomart = data.table::fread("../_h/biomart.csv")
}
memMART <- memoise::memoise(get_biomart_df)

get_pheno_df <- function(){
    phenotype_file = '../../../../../_m/phenotypes_bsp1.tsv'
    return(data.table::fread(phenotype_file))
}
memPHENO <- memoise::memoise(get_pheno_df)

get_genotypes <- function(){
    traw_file = paste0("/ceph/projects/brainseq/genotype/download/topmed/",
                       "imputation_filter/convert2plink/filter_maf_01/",
                       "a_transpose/_m/LIBD_Brain_TopMed.traw")
    traw = data.table::fread(traw_file) %>% rename_with(~ gsub('\\_.*', '', .x))
    return(traw)
}
memSNPs <- memoise::memoise(get_genotypes)

get_geno_annot <- function(){
    return(memSNPs() %>% select(CHR, SNP, POS, COUNTED, ALT))
}
memANNOT <- memoise::memoise(get_geno_annot)

get_snps_df <- function(){
    return(memSNPs() %>% select("SNP", starts_with("Br")))
}
memGENO <- memoise::memoise(get_snps_df)

get_ml <- function(){
    fn1 <- "../../metrics_summary/_m/dRFE_minimal_subset_summary.txt"
    return(data.table::fread(fn1) %>% filter(Test=="test_r2", Median > 0))
}
memML <- memoise::memoise(get_ml)

get_eqtls <- function(){
    fn2 <- "../../metrics_summary/_m/dRFE_predictive_features.txt.gz"
    return(data.table::fread(fn2) %>% filter(Feature %in% memML()$Feature)%>%
           select(-Median) %>% mutate(gene_id=gsub("_", ".", Feature)))
}
memEQTL <- memoise::memoise(get_eqtls)

save_ggplots <- function(fn, p, w, h){
    for(ext in c('.pdf', '.png', '.svg')){
        ggsave(paste0(fn, ext), plot=p, width=w, height=h)
    }
}

letter_snp <- function(number, a0, a1){
    if(is.na(number)){ return(NA) }
    if( length(a0) == 1 & length(a1) == 1){
        seps = ""; collapse=""
    } else {
        seps = " "; collapse=NULL
    }
    return(paste(paste0(rep(a0, number), collapse = collapse),
                 paste0(rep(a1, (2-number)), collapse = collapse), sep=seps))
}

get_snp_df <- function(variant_id, gene_id){
    zz = memANNOT() %>% filter(SNP == variant_id)
    xx = memGENO() %>% filter(SNP == variant_id) %>%
        column_to_rownames("SNP") %>% t %>% as.data.frame %>%
        rownames_to_column("BrNum")%>% mutate(COUNTED=zz$COUNTED,ALT=zz$ALT) %>%
        rename("SNP"=all_of(variant_id))
    yy = memRES()[gene_id, ] %>% t %>% as.data.frame %>%
        rownames_to_column("RNum") %>% inner_join(memPHENO(), by="RNum")
    ## Annotated SNPs
    letters = c()
    for(ii in seq_along(xx$COUNTED)){
        a0 = xx$COUNTED[ii]; a1 = xx$ALT[ii]; number = xx$SNP[ii]
        letters <- append(letters, letter_snp(number, a0, a1))
    }
    xx = xx %>% mutate(LETTER=letters, ID=paste(SNP, LETTER, sep="\n"))
    df = inner_join(xx, yy, by="BrNum") %>% mutate_if(is.character, as.factor)
    return(df)
}
memDF <- memoise::memoise(get_snp_df)

get_gene_symbol <- function(gene_id){
    ensemblID = gsub("\\..*", "", gene_id)
    geneid = memMART() %>% filter(ensembl_gene_id == gsub("\\..*", "", gene_id))
    if(dim(geneid)[1] == 0){
        return("")
    } else {
        return(geneid$external_gene_name)
    }
}

plot_simple_eqtl <- function(fn, gene_id, variant_id, eqtl_annot){
    y0 = quantile(memDF(variant_id,gene_id)[[gene_id]],probs=c(0.01))[[1]] - 0.2
    y1 = quantile(memDF(variant_id,gene_id)[[gene_id]],probs=c(0.99))[[1]] + 0.2
    bxp = memDF(variant_id, gene_id) %>%
        ggboxplot(x="ID", y=gene_id, fill="red", color="red", add="jitter",
                  xlab=variant_id,ylab="Residualized Expression",outlier.shape=NA,
                  add.params=list(alpha=0.5), alpha=0.4, legend="bottom",
                  palette="npg", ylim=c(y0,y1),
                  ggtheme=theme_pubr(base_size=20, border=TRUE)) +
        font("xy.title", face="bold") +
        ggtitle(paste(get_gene_symbol(gene_id),gene_id,eqtl_annot,sep='\n')) +
        theme(plot.title = element_text(hjust = 0.5, face="bold"))
    print(bxp)
    save_ggplots(fn, bxp, 7, 7)
}

###### MAIN
eGenes <- memEQTL() %>% arrange(Rank) %>% group_by(gene_id) %>% slice(1)
for(num in seq_along(eGenes$gene_id)){
    variant_id = eGenes$SNP[num]
    gene_id = eGenes$gene_id[num]
    r2_score = filter(memML(), Feature == eGenes$Feature[num])$Median
    eqtl_annot = paste("Test R2: ", signif(r2_score, 2))
    fn = paste(eGenes$Feature[num],"eqtl", sep="_")
    plot_simple_eqtl(fn, gene_id, variant_id, eqtl_annot)
}

#### Reproducibility information ####
print("Reproducibility information:")
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
