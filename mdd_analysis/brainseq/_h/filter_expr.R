## Select and filter data for MDD analysis
suppressPackageStartupMessages({
    library(dplyr)
    library(SummarizedExperiment)
})

                                        # Function from jaffelab github
merge_rse_metrics <- function(rse) {
    stopifnot(is(rse, 'RangedSummarizedExperiment'))

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

    rse$numMapped = sapply(rse$numMapped, sum)
    rse$numReads = sapply(rse$numReads, sum)
    rse$numUnmapped = sapply(rse$numUnmapped, sum)
    rse$mitoMapped = sapply(rse$mitoMapped, sum)
    rse$totalMapped = sapply(rse$totalMapped, sum)
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

prep_data <- function(){
    fn = paste0("/ceph/projects/brainseq/rnaseq/phase1_DLPFC_PolyA/jaffe_counts/",
                "_m/dlpfc_polyA_brainseq_phase1_hg38_rseGene_merged_n732.rda")
    load(fn)
    rse_df = rse_gene
    keepIndex = which((rse_df$Dx %in% c("Control", "MDD")) &
                      rse_df$Age > 17 & rse_df$Race %in% c("AA", "CAUC"))
    rse_df = rse_df[, keepIndex]
    rse_df$Dx = factor(rse_df$Dx, levels = c("Control", "MDD"))
    rse_df$Sex <- factor(rse_df$Sex)
    rse_df <- merge_rse_metrics(rse_df)
    colData(rse_df)$RIN = sapply(colData(rse_df)$RIN,"[",1)
    rownames(colData(rse_df)) <- sapply(strsplit(rownames(colData(rse_df)), "_"),
                                        "[", 1)
    pheno = colData(rse_df) %>% as.data.frame %>%
        inner_join(memMDS(), by=c("BrNum"="FID")) %>%
        distinct(RNum, .keep_all = TRUE)
    x <- edgeR::DGEList(counts=assays(rse_df)$counts[, pheno$RNum],
                        genes=rowData(rse_df), samples=pheno)
    design0 <- model.matrix(~Dx, data=x$samples)
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

qSV_model <- function(){
    x <- memPREP()
                                        # Design matrix
    mod = model.matrix(~Dx + Race + Sex + Age + mitoRate + rRNA_rate +
                           RIN + totalAssignedGene + overallMapRate +
                           snpPC1 + snpPC2 + snpPC3,
                       data=x$samples)
    colnames(mod) <- gsub("Dx", "", colnames(mod))
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

get_voom <- function(){
                                        # Preform voom
    x <- memPREP()
    # Get model
    modQsva <- memMODEL()
    v <- limma::voom(x[, rownames(modQsva)], modQsva)
    save(v, file='voomSVA.RData')
    return(v)
}

## Run voom normalization
get_voom()

## Reproducibility
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
