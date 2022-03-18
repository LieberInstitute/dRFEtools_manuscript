## Calculate the partial R2 using RaFFE SNPs and individually

suppressPackageStartupMessages({
    library("tidyverse")
})

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

get_pheno <- function(){
    load("../../../../../neuropsychiatric_analysis/_m/voomSVA.RData")
    pheno <- v$design %>% as.data.frame %>%
        mutate(Dx=ifelse(MDD==0 & Schizo==0,"CTL",ifelse(MDD==1,"MDD","SZ")),
               Sex=ifelse(Male==1, "Male", "Female")) %>%
        select(-c(Intercept, "MDD", "Schizo", "Male"))
    return(pheno)
}
memPHENO <- memoise::memoise(get_pheno)

select_snps <- function(gname){
    snp_dir = "../../../_m/"
    fn = paste(snp_dir, gname, "snps.csv", sep="/")
    dt <- memEQTL() %>% filter(Feature == gname)
    gene_id = dt$gene_id %>% unique
    return(data.table::fread(fn) %>% select(all_of(c("V1", dt$SNP, gene_id)))%>%
           rename_with(~gsub(":", "_", .), starts_with('chr')))
}
memSNPs <- memoise::memoise(select_snps)

calculate_idv_partial <- function(gname){
                                        # Get data
    dt <- memEQTL() %>% filter(Feature == gname)
    gene_id = dt$gene_id %>% unique
    df = memPHENO() %>% rownames_to_column("V1") %>%
        inner_join(memSNPs(gname), by="V1") %>% column_to_rownames("V1")
                                        # Model 1
    model1 = paste(paste0(gene_id, "~ Dx"), "Sex + Age + mitoRate + rRNA_rate",
                   "overallMapRate + Adapter70.71_R2 + percentGC_R1 + percentGC_R2",
                   "snpqPC1 + snpqPC2 + snpqPC3",
                   paste(colnames(df)[grep("^qPC",colnames(df))],collapse=" + "),
                   sep=" + ")
    reduced = anova(lm(model1, data=df))
                                        # Loop through SNPs for model 2
    snp_lt <- c(); partial1 <-  c(); partial2 <-  c(); partial_r2 <-  c()
    for(col in colnames(df)[(grep("^chr", colnames(df)))]){
        snp_lt = c(snp_lt, gsub("_", ":", col))
        model2 = paste(model1, col, sep=" + ")
        full = anova(lm(model2, data=df))
        p1 = reduced["Residuals", "Sum Sq"]
        p2 = full["Residuals", "Sum Sq"]
        partial1 = c(partial1, p1); partial2 = c(partial2, p2)
        partial_r2 = c(partial_r2, (p1 - p2) / p1)
    }
    dft = data.frame("Geneid"=gene_id, "SNP"=snp_lt, "Partial_R2"=partial_r2,
                     "Full_R2"=partial2, "Reduced_R2"=partial1)
    return(dt %>% inner_join(dft, by=c("gene_id"="Geneid", "SNP")))
}

####    MAIN    ####
bigdata <- list()
genes <- memEQTL()$Feature %>% unique
for(ii in seq_along(genes)){
    gname <- genes[ii]
    bigdata[[ii]] <- calculate_idv_partial(gname)
}
                                        # Save data
bind_rows(bigdata) %>%
    data.table::fwrite("partial_r2_genetic_variances.tsv", sep='\t')
                                        # Correlation between Rank and Partial R2
lf <- logr::log_open("summary.log", logdir=FALSE, autolog=FALSE)
logr::log_print("Summary of correlation between Rank and Partial r2:")

est_fit <- bind_rows(bigdata) %>% group_by(gene_id) %>%
    do(fit=broom::tidy(lm(Partial_R2 ~ Rank, data=.))) %>% unnest(fit) %>%
    filter(term != "(Intercept)") %>%
    mutate(p.bonf = p.adjust(p.value, "bonf"), p.bonf.sig = p.bonf < 0.05,
           p.bonf.cat = cut(p.bonf, breaks = c(1,0.05, 0.01, 0.005, 0),
                            labels = c("<= 0.005","<= 0.01","<= 0.05","> 0.05"),
                            include.lowest = TRUE),
           p.fdr = p.adjust(p.value, "fdr"),
           log.p.bonf = -log10(p.bonf))
logr::log_print(est_fit %>% count(p.bonf.cat))
logr::log_print(est_fit %>% as.data.frame)
logr::log_close()

#### Reproducibility information ####
print("Reproducibility information:")
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
