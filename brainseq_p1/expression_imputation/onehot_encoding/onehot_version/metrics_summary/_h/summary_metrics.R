## This script calculates model statistics across N folds.

library(ggpubr)
library(tidyverse)

save_plot <- function(p, fn, w, h){
    for(ext in c(".pdf", ".png", ".svg")){
        ggsave(filename=paste0(fn,ext), plot=p, width=w, height=h)
    }
}


extract_rank <- function(method){
    fn = "../../database/_m/rank_features.txt.gz"
    ml_df = data.table::fread(fn) %>% group_by(Feature, SNP) %>%
        summarize(median_all = median(Rank)) %>% arrange(Feature,median_all) %>%
        mutate(Rank = rank(median_all)) %>% select(-median_all)
    return(ml_df)
}
memRANK <- memoise::memoise(extract_rank)


summary_max_data <- function(){
    fn = "../../database/_m/dRFEtools_10Folds.txt.gz"
    ml_df = data.table::fread(fn) %>% mutate_at("fold", as.character) %>%
        select(feature, fold, n_features, train_r2, train_mse, train_evar,
               test_r2, test_mse, test_evar) %>%
        pivot_longer(-c(fold, feature), names_to='Test', values_to='Score') %>%
        group_by(feature, Test) %>% rename("Feature"="feature") %>%
        summarise(Mean = mean(Score), Median = median(Score),
                  Std = sd(Score), .groups = "keep")
    return(ml_df)
}


summary_redundant_data <- function(){
    fn = "../../database/_m/dRFEtools_10Folds.txt.gz"
    ml_df = data.table::fread(fn) %>% mutate_at("fold", as.character) %>%
        select(feature, fold, ends_with("_redundant")) %>%
        pivot_longer(-c(fold, feature), names_to='Test', values_to='Score') %>%
        group_by(feature, Test) %>% rename("Feature"="feature") %>%
        summarise(Mean = mean(Score), Median = median(Score),
                  Std = sd(Score), .groups = "keep")
    return(ml_df)
}


extract_n_save <- function(){
    nf_df <- summary_max_data() %>% as.data.frame %>%
        filter(Test == "n_features") %>% select(Feature, Median)
    nfr_df <- summary_redundant_data() %>% as.data.frame %>%
        filter(Test == "n_redundant") %>% select(Feature, Median)
    memRANK() %>% inner_join(nf_df, by="Feature") %>% filter(Rank <= Median) %>%
        data.table::fwrite("dRFE_predictive_features.txt.gz", sep='\t')
    memRANK() %>% inner_join(nfr_df,by="Feature") %>% filter(Rank <= Median) %>%
        data.table::fwrite("dRFE_redundant_features.txt.gz", sep='\t')
    summary_max_data() %>% as.data.frame %>%
        data.table::fwrite("dRFE_minimal_subset_summary.txt", sep='\t')
    summary_redundant_data() %>% as.data.frame %>%
        data.table::fwrite("dRFE_redundant_subset_summary.txt", sep='\t')
}


extract_ml_data <- function(MAX=TRUE){
    ml_file = "../../database/_m/dRFEtools_10Folds.txt.gz"
    if(MAX){
        ml_df2 = data.table::fread(ml_file) %>% mutate_at("fold", as.character) %>%
            select(feature, fold, test_r2)
        method = "Minimal"
    } else {
        ml_df2 = data.table::fread(ml_file) %>% mutate_at("fold", as.character) %>%
            select(feature, fold, test_r2_redundant)
        method = "Redundant"
    }
    ml_df1 = data.table::fread(ml_file) %>% mutate_at("fold", as.character) %>%
        select(feature, fold, train_r2) %>% mutate(Method=method, Type="Train") %>%
        pivot_longer(-c(feature, fold, Method, Type), names_to='Metric',
                     values_to='Score') %>% rename("Feature"="feature")
    ml_df2 = ml_df2 %>% mutate(Method=method, Type="Test") %>%
        pivot_longer(-c(feature, fold, Method, Type), names_to='Metric',
                     values_to='Score') %>% rename("Feature"="feature")
    return(bind_rows(ml_df1, ml_df2))
}


plot_n_save <- function(){
    datalist = list()
    max_lt = c(TRUE, FALSE)
    for(MAX in seq_along(max_lt)){
        datalist[[MAX]] <- extract_ml_data(max_lt[MAX])
    }
    dt = bind_rows(datalist) %>% mutate_if(is.character, as.factor) %>%
        mutate(Type=factor(Type, levels=c("Train", "Test")))
    dist = ggdensity(dt, x="Score", add="mean", rug=TRUE, color="Type",
                     fill="Type", palette="npg", facet.by="Method",
                     panel.labs.font=list(face='bold'), legend="bottom",
                     add.params=list(alpha=0.5), ncol=1, xlab="Test R2",
                     ylab="", ggtheme=theme_pubr(base_size=20, border=TRUE))
    save_plot(dist, "density_plot_r2", 6, 6)
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
