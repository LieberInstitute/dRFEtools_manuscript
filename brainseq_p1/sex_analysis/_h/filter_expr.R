## Select and filter data for MDD analysis
suppressPackageStartupMessages({
    library(dplyr)
    library(SummarizedExperiment)
})

                                        # Function from jaffelab github
merge_rse_metrics <- function(rse) {
    stopifnot(is(rse, 'RangedSummarizedExperiment'))
    rse$numMapped = sapply(rse$numMapped, sum)
    rse$numReads = sapply(rse$numReads, sum)
    rse$numUnmapped = sapply(rse$numUnmapped, sum)
    rse$mitoMapped = sapply(rse$mitoMapped, sum)
    rse$totalMapped = sapply(rse$totalMapped, sum)
    rse$overallMapRate = mapply(function(r, n) {
        sum(r*n)/sum(n)
    }, rse$overallMapRate, rse$numReads)
    rse$mitoRate = mapply(function(r, n) {
        sum(r*n)/sum(n)
    }, rse$mitoRate, rse$numMapped)
    rse$rRNA_rate = mapply(function(r, n) {
        sum(r*n)/sum(n)
    }, rse$rRNA_rate, rse$numMapped)
    rse$totalAssignedGene = mapply(function(r, n) {
        sum(r*n)/sum(n)
    }, rse$totalAssignedGene, rse$numMapped)
    return(rse)
}

get_mds <- function(){
    mds_file = paste0("/ceph/projects/brainseq/genotype/download/topmed/",
                      "imputation_filter/convert2plink/filter_maf_01/mds/",
                      "_m/LIBD_Brain_TopMed.mds")
    mds = data.table::fread(mds_file) %>%
        rename_at(.vars = vars(starts_with("C")),
                  function(x){sub("C", "snpPC", x)}) %>%
        mutate_if(is.character, as.factor)
    return(mds)
}
memMDS <- memoise::memoise(get_mds)

check_dup <- function(df){
    sample <- df %>% select_if(is.numeric)
    variables <- names(sample)
    return(cytominer::correlation_threshold(variables, sample, cutoff=0.95))
}

prep_data <- function(diagnosis){
    fn = paste0("/ceph/projects/brainseq/rnaseq/phase1_DLPFC_PolyA/jaffe_counts/",
                "_m/dlpfc_polyA_brainseq_phase1_hg38_rseGene_merged_n732.rda")
    load(fn)
    rse_df = rse_gene
    keepIndex = which((rse_df$Dx==diagnosis) & rse_df$Age > 17 &
                      rse_df$Race %in% c("AA", "CAUC"))
    rse_df = rse_df[, keepIndex]
    rse_df$Sex <- factor(rse_df$Sex)
    colData(rse_df)$RIN = sapply(colData(rse_df)$RIN,"[",1)
    rownames(colData(rse_df)) <- sapply(strsplit(rownames(colData(rse_df)), "_"),
                                        "[", 1)
    #rse_df <- merge_rse_metrics(rse_df)
    pheno = colData(rse_df) %>% as.data.frame %>%
        select("RNum", "BrNum", "Dx", "Sex", "Age", "RIN", "mitoRate",
               "totalAssignedGene", "overallMapRate", "concordMapRate",
               "rRNA_rate", starts_with(c("Adapter", "phredGT", "percent")),
               -AdapterContent) %>%
        mutate_if(is.list, ~sapply(., sum)) %>%
        mutate_if(is.numeric, scales::rescale) %>%
        inner_join(memMDS() %>% select("FID", "snpPC1", "snpPC2", "snpPC3"),
                   by=c("BrNum"="FID")) %>% distinct(RNum, .keep_all = TRUE)
                                        # Remove highly correlated variables
    ## Tested this by hand
    drop_vars <- c("totalAssignedGene", "concordMapRate", "phredGT30_R1",
                   "phredGT30_R2", "phredGT35_R1", "phredGT35_R2",
                   "Adapter50.51_R1", "Adapter50.51_R2", "Adapter88_R1",
                   "Adapter88_R2", "Adapter70.71_R1")
    pheno <- pheno %>% select(-all_of(drop_vars))
    if(length(check_dup(pheno)) != 0){
        pheno <- pheno %>% select(-check_dup(pheno))
    }
                                        # Make DGEList variable
    x <- edgeR::DGEList(counts=assays(rse_df)$counts[, pheno$RNum],
                        genes=rowData(rse_df), samples=pheno)
    design0 <- model.matrix(~Sex, data=x$samples)
    keep.x <- edgeR::filterByExpr(x, design=design0)
    x <- x[keep.x, , keep.lib.sizes=FALSE]
    print(paste('There are:', sum(keep.x), 'features left!'))
    x <- edgeR::calcNormFactors(x, method="TMM")
    return(x)
}
memPREP <- memoise::memoise(prep_data)

cal_qSV <- function(){
    qsv_file = paste0("/ceph/projects/brainseq/rnaseq/degradation/_m/",
                      "degradationMat_DLPFC_polyA_Phase1.csv.gz")
    dm <- data.table::fread(qsv_file) %>% tibble::column_to_rownames("V1")
    qSV <- sva::qsva(dm) %>% as.data.frame
    if("TRUE" %in% grepl("_", rownames(qSV))){
                                        # Remove underscore if needed
        rownames(qSV) <- sapply(strsplit(rownames(qSV), "_"), "[", 1)
    }
    return(qSV)
}
memQSV <- memoise::memoise(cal_qSV)

qSV_model <- function(diagnosis){
    x <- memPREP(diagnosis)
                                        # Design matrix
    mod = model.matrix(~Sex + Age + RIN + mitoRate + rRNA_rate +
                           overallMapRate + Adapter70.71_R2 + percentGC_R1 +
                           percentGC_R2 + snpPC1 + snpPC2 + snpPC3,
                       data=x$samples)
    colnames(mod) <- gsub("SexM", "Male", colnames(mod))
    colnames(mod) <- gsub("\\(Intercept\\)", "Intercept", colnames(mod))
                                        # Load qSV
    qsv = memQSV() %>% tibble::rownames_to_column()
    modQsva <- mod %>% as.data.frame %>%
        tibble::rownames_to_column() %>%
        inner_join(qsv, by="rowname") %>%
        rename_all(list(~stringr::str_replace_all(., 'PC', 'qPC'))) %>%
        tibble::column_to_rownames("rowname") %>% as.matrix
    return(modQsva)
}
memMODEL <- memoise::memoise(qSV_model)

get_voom <- function(diagnosis){
                                        # Preform voom
    x <- memPREP(diagnosis)
    # Get model
    modQsva <- memMODEL(diagnosis)
    v <- limma::voom(x[, rownames(modQsva)], modQsva)
    return(v)
}
memVOOM <- memoise::memoise(get_voom)

cal_res <- function(diagnosis){
                                        # Calculate residuals
    v          <- memVOOM(diagnosis)
    null_model <- v$design %>% as.data.frame %>%
        select(-c("Male")) %>% as.matrix
    fit_res    <- limma::lmFit(v, design=null_model)
    res        <- v$E - ( fit_res$coefficients %*% t(null_model) )
    res_sd     <- apply(res, 1, sd)
    res_mean   <- apply(res, 1, mean)
    res_norm   <- (res - res_mean) / res_sd
    write.table(res_norm, sep="\t", quote=FALSE,
                file=paste0(tolower(diagnosis), '/residualized_expression.tsv'))
}
memRES <- memoise::memoise(cal_res)

#### MAIN ####
main <- function(){
    for(diagnosis in c("Control", "MDD")){
        dir.create(tolower(diagnosis))
                                        # Run voom normalization
        v <- memVOOM(diagnosis)
        save(v, file=paste0(tolower(diagnosis),'/voomSVA.RData'))
                                        # Residualize data
        memRES(diagnosis)
    }
}

main()

## Reproducibility
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
