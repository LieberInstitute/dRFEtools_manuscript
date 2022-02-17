## This script is used to extract phenotypes from the RSE object.

suppressPackageStartupMessages({
    library(magrittr)
    library(SummarizedExperiment)
})

main <- function(){
    fn = paste0("/ceph/projects/brainseq/rnaseq/phase1_DLPFC_PolyA/jaffe_counts/",
                "_m/dlpfc_polyA_brainseq_phase1_hg38_rseGene_merged_n732.rda")
    load(fn)
    rse_df = rse_gene
    rse_df$Sex <- factor(rse_df$Sex)
    colData(rse_df)$RIN = sapply(colData(rse_df)$RIN,"[",1)
    rownames(colData(rse_df)) <- sapply(strsplit(rownames(colData(rse_df)), "_"),
                                        "[", 1)

    colData(rse_df) %>% as.data.frame %>%
        dplyr::select("RNum", "BrNum", "Dx", "Race", "Sex", "Age", "RIN", "mitoRate",
                      "totalAssignedGene", "overallMapRate", "rRNA_rate",
                      dplyr::starts_with(c("Adapter", "phredGT", "percent")),
                      -AdapterContent,) %>%
        dplyr::mutate_if(is.list, ~sapply(., sum)) %>%
        data.table::fwrite("phenotypes_bsp1.tsv", sep='\t')
}

main()

## Reproducibility
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
