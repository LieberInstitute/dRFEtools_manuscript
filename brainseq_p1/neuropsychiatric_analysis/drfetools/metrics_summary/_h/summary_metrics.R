## This script calculates model statistics across N folds.

library(ggpubr)
library(tidyverse)

save_plot <- function(p, fn, w, h){
    for(ext in c(".pdf", ".png", ".svg")){
        ggsave(filename=paste0(fn,ext), plot=p, width=w, height=h)
    }
}


extract_rank <- function(method){
    fn = paste0("../../_m/", tolower(method), "/rank_features.txt")
    ml_df = data.table::fread(fn) %>%
        rename('Geneid'='V1', 'Fold'='V2', 'Rank'='V3') %>%
        pivot_wider(names_from = Fold, values_from = Rank) %>%
        mutate(median_all = apply(., 1, median)) %>%
        arrange(median_all) %>% mutate(Rank = rank(median_all)) %>%
        select(Geneid, Rank)
    return(ml_df)
}


summary_max_data <- function(method){
    fn = paste0("../../_m/", tolower(method), "/dRFEtools_10folds.txt")
    ml_df = data.table::fread(fn) %>% mutate_at("fold", as.character) %>%
        select(fold, n_features, train_acc, train_nmi, train_roc, test_acc,
               test_nmi, test_roc) %>%
        pivot_longer(-fold, names_to='Test', values_to='Score') %>%
        group_by(Test) %>%
        summarise(Mean = mean(Score), Median = median(Score),
                  Std = sd(Score), .groups = "keep") %>%
        mutate(Method=method)
    return(ml_df)
}


summary_redundant_data <- function(method){
    fn = paste0("../../_m/", tolower(method), "/dRFEtools_10folds.txt")
    ml_df = data.table::fread(fn) %>% mutate_at("fold", as.character) %>%
        select(fold, ends_with("_redundant")) %>%
        pivot_longer(-fold, names_to='Test', values_to='Score') %>%
        group_by(Test) %>%
        summarise(Mean = mean(Score), Median = median(Score),
                  Std = sd(Score), .groups = "keep") %>%
        mutate(Method=method)
    return(ml_df)
}


extract_ml_data <- function(MAX=TRUE){
    ml_file = "../../_m/rf/dRFEtools_10folds.txt"
    if(MAX){
        ml_df2 = data.table::fread(ml_file) %>% mutate_at("fold", as.character) %>%
            select(fold, test_roc)
        method = "Minimal"
    } else {
        ml_df2 = data.table::fread(ml_file) %>% mutate_at("fold", as.character) %>%
            select(fold, test_roc_redundant)
        method = "Redundant"
    }
    ml_df1 = data.table::fread(ml_file) %>% mutate_at("fold", as.character) %>%
        select(fold, train_roc) %>% mutate(Method=method, Type="Train") %>%
        pivot_longer(-c(fold,Method,Type), names_to='Metric', values_to='Score')
    ml_df2 = ml_df2 %>% mutate(Method=method, Type="Test") %>%
        pivot_longer(-c(fold,Method,Type), names_to='Metric', values_to='Score')
    return(bind_rows(ml_df1, ml_df2))
}


extract_n_save <- function(){
    nf = filter(summary_max_data("RF"), Test == "n_features")$Median
    nfr = filter(summary_redundant_data("RF"),Test == "n_redundant")$Median
    extract_rank("RF") %>% filter(Rank <= nf) %>%
        data.table::fwrite("dRFE_predictive_features.txt.gz", sep='\t')
    extract_rank("RF") %>% filter(Rank <= nfr) %>%
        data.table::fwrite("dRFE_redundant_features.txt.gz", sep='\t')
    summary_max_data("RF") %>%
        data.table::fwrite("dRFE_minimal_subset_summary.txt", sep='\t')
    summary_redundant_data("RF") %>%
        data.table::fwrite("dRFE_redundant_subset_summary.txt", sep='\t')
}


plot_n_save <- function(){
    datalist = list()
    max_lt = c(TRUE, FALSE)
    for(MAX in seq_along(max_lt)){
        datalist[[MAX]] <- extract_ml_data(max_lt[MAX])
    }
    dt = bind_rows(datalist) %>% mutate_if(is.character, as.factor) %>%
        mutate(Type=factor(Type, levels=c("Train", "Test")))
    bxp = ggboxplot(dt, x="Method", y="Score", color="Type", palette="npg",
                    ylab="Model ROC", xlab="", add="jitter",
                    outlier.shape=NA, panel.labs.font=list(face='bold'),
                    legend="bottom", add.params=list(alpha=0.5), ylim=c(0.5, 1),
                    ggtheme=theme_pubr(base_size=20, border=TRUE)) +
        geom_hline(yintercept=0.85, linetype=2) + rotate_x_text(45)
    fn = "model_roc_plot"
    save_plot(bxp, fn, 3, 5)
}

##### MAIN #######
main <- function(){
    extract_n_save()
    plot_n_save()
}

main()

## Reproducibility
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
