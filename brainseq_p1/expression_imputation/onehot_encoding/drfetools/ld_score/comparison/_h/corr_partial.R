## This script examines correlation between partial R2 and LD for
## genetic variants
suppressPackageStartupMessages({
    library(dplyr)
})

get_partial <- function(){
    fn = "../../../partial_r2/_m/partial_r2_genetic_variances.tsv"
    return(data.table::fread(fn))
}

get_ld <- function(){
    bigdata <- list()
    files <- list.files("../../_m", "*ld", full.names=TRUE)
    for(ii in seq_along(files)){
        gname = gsub(".ld", "", basename(files[ii]))
        bigdata[[ii]] <- data.table::fread(files[[ii]]) %>%
            mutate(Feature = gname)
    }
    return(bind_rows(bigdata))
}

lf <- logr::log_open("summary.log", logdir=FALSE, autolog=FALSE)
logr::log_print("Summary of partial r2")
df1 <- get_partial() %>% group_by(Feature) %>%
    summarize(Max_Variance=max(Partial_R2), N_SNPS=n())
logr::log_print(df1)
logr::log_print("Summary of features with LD r2 > 0.8")
df2 <- get_ld() %>% filter(R2 > 0.8) %>% group_by(Feature) %>%
    summarize(LD_SNPS=n())
logr::log_print(df2)
logr::log_print("Table 1: Expression Imputation Results:")
logr::log_print(inner_join(df1, df2, by="Feature") %>%
                mutate(Percent_LD=LD_SNPS/N_SNPS))
logr::log_print("Summary of high partial r2, LD:")
high_corr <- get_partial() %>% filter(Partial_R2 > 0.5) %>%
    select(Feature, SNP, Partial_R2)
logr::log_print(high_corr)
logr::log_print(get_ld() %>%
                filter(Feature %in% unique(high_corr$Feature),
                       SNP_A %in% unique(high_corr$SNP)))
logr::log_close()
