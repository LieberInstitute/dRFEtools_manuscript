## This script calculates model statistics across N folds.

library(ggpubr)
library(tidyverse)

save_plot <- function(p, fn, w, h){
    for(ext in c(".pdf", ".png", ".svg")){
        ggsave(filename=paste0(fn,ext), plot=p, width=w, height=h)
    }
}


extract_rank <- function(diagnosis){
    fn = paste0("../../_m/", tolower(diagnosis), "/rank_features.txt")
    ml_df = data.table::fread(fn) %>%
        rename('Geneid'='V1', 'Fold'='V2', 'Rank'='V3') %>%
        pivot_wider(names_from = Fold, values_from = Rank) %>%
        mutate(median_all = apply(., 1, median)) %>%
        arrange(median_all) %>% mutate(Rank = rank(median_all)) %>%
        select(Geneid, Rank)
    return(ml_df)
}


summary_max_data <- function(diagnosis){
    fn = paste0("../../_m/", tolower(diagnosis), "/dRFEtools_10folds.txt")
    ml_df = data.table::fread(fn) %>% mutate_at("fold", as.character) %>%
        select(fold, n_features, train_acc, train_nmi, train_roc, test_acc,
               test_nmi, test_roc) %>%
        pivot_longer(-fold, names_to='Test', values_to='Score') %>%
        group_by(Test) %>%
        summarise(Mean = mean(Score), Median = median(Score),
                  Std = sd(Score), .groups = "keep") %>%
        mutate(Diagnosis=diagnosis)
    return(ml_df)
}


summary_redundant_data <- function(diagnosis){
    fn = paste0("../../_m/", tolower(diagnosis), "/dRFEtools_10folds.txt")
    ml_df = data.table::fread(fn) %>% mutate_at("fold", as.character) %>%
        select(fold, ends_with("_redundant")) %>%
        pivot_longer(-fold, names_to='Test', values_to='Score') %>%
        group_by(Test) %>%
        summarise(Mean = mean(Score), Median = median(Score),
                  Std = sd(Score), .groups = "keep") %>%
        mutate(Diagnosis=diagnosis)
    return(ml_df)
}

test_signif <- function(diagnosis){
    ctl = data.table::fread("../../_m/control/dRFEtools_10folds.txt")
    if(diagnosis != "Control"){
        fn = paste0("../../_m/", tolower(diagnosis), "/dRFEtools_10folds.txt")
        df = data.table::fread(fn)
        print(paste0("Control < ",diagnosis, ":"))
        print(t.test(ctl$test_acc, df$test_acc, alternative="less"))
    }
}

extract_ml_data <- function(diagnosis){
    ml_file = paste0("../../_m/", tolower(diagnosis), "/dRFEtools_10folds.txt")
    ml_df1 = data.table::fread(ml_file) %>% mutate_at("fold", as.character) %>%
        select(fold, train_acc) %>% mutate(Diagnosis=diagnosis, Type="Train") %>%
        pivot_longer(-c(fold,Diagnosis,Type), names_to='Metric', values_to='Score')
    ml_df2 = data.table::fread(ml_file) %>% mutate_at("fold", as.character) %>%
        select(fold, test_acc) %>% mutate(Diagnosis=diagnosis, Type="Test") %>%
        pivot_longer(-c(fold,Diagnosis,Type), names_to='Metric', values_to='Score')
    return(bind_rows(ml_df1, ml_df2))
}


extract_n_save <- function(){
    maxlist = list(); pred_featlist = list(); redund_list = list()
    for(diagnosis in c("Control", "MDD", "Schizo", "Bipolar")){
        test_signif(diagnosis)
        nf = filter(summary_max_data(diagnosis), Test == "n_features")$Median
        nfr = filter(summary_redundant_data(diagnosis),
                     Test == "n_redundant")$Median
        dat0 = extract_rank(diagnosis) %>% filter(Rank < nf) %>%
            mutate(Diagnosis=diagnosis)
        dat1 <- summary_max_data(diagnosis)
        dat2 = extract_rank(diagnosis) %>% filter(Rank < nfr) %>%
            mutate(Diagnosis=diagnosis)
        pred_featlist[[diagnosis]] <- dat0
        maxlist[[diagnosis]] <- dat1
        redund_list <- dat2
    }
    bind_rows(pred_featlist) %>%
        data.table::fwrite("dRFE_predictive_features.txt.gz", sep='\t')
    bind_rows(maxlist) %>%
        data.table::fwrite("dRFE_minimal_subset_summary.txt", sep='\t')
    bind_rows(redund_list) %>%
        data.table::fwrite("dRFE_redundant_features.txt.gz", sep='\t')
}


plot_n_save <- function(){
    datalist = list()
    for(diagnosis in c("Control", "MDD", "Schizo", "Bipolar")){
        datalist[[diagnosis]] <- extract_ml_data(diagnosis)
    }
    dt = bind_rows(datalist) %>% mutate_if(is.character, as.factor) %>%
        mutate(Type=factor(Type, levels=c("Train", "Test")))
    bxp = ggboxplot(dt, x="Diagnosis", y="Score", color="Type", palette="npg",
                    ylab="Model Accuracy", xlab="", add="jitter",
                    outlier.shape=NA, panel.labs.font=list(face='bold'),
                    legend="bottom", add.params=list(alpha=0.5), ylim=c(0, 1),
                    ggtheme=theme_pubr(base_size=20, border=TRUE)) +
        geom_hline(yintercept=0.85, linetype=2) + rotate_x_text(45)
    save_plot(bxp, "model_accuracy_plot", 6, 6)
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
