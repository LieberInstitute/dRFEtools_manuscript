## This R script examines and plots the accuracy of RFE versus dRFE.

library(dplyr)
library(ggpubr)

save_plot <- function(p, fn, w, h){
    for(ext in c(".pdf", ".png", ".svg")){
        ggsave(filename=paste0(fn,ext), plot=p, width=w, height=h)
    }
}

cal_metrics <- function(){
    fn = "../../feature_accuracy/_m/simulated_data_accuracy_metrics.tsv"
    return(data.table::fread(fn) %>%
           mutate(ID=paste0(RFE_Method, " (",Elimination,")"),
                  Acc=(TP+TN)/(TP+FN+TN+FP), FDR=(1-(TP/(TP+FP))),
                  F1_score=(2*TP)/(2*TP + FP + FN), TNR=(TN/(FP+TN))))
}

get_roc_metrics <- function(){
    fn = "../../feature_accuracy/_m/simulated_data_roc_metrics.tsv.gz"
    return(data.table::fread(fn))
}

print_summary <- function(){
    print("Mean:")
    df.mean <- cal_metrics() %>%
        group_by(RFE_Method, Elimination, Algorithm) %>%
        summarize_at(vars(c("Acc", "FDR", "F1_score", "TNR", "CPU")), mean)
    print(df.mean)
    print("Median:")
    df.median <- cal_metrics() %>%
        group_by(RFE_Method, Elimination, Algorithm) %>%
        summarize_at(vars(c("Acc", "FDR", "F1_score", "TNR", "CPU")), median)
    print(df.median)
}

plot_roc <- function(){
    outfile = "roc_curve_feature_selection_simulation"
    df = get_roc_metrics() %>%
        mutate(ID=paste0(RFE_Method," (",Elimination,")"),
               FPR=ifelse(is.na(FPR), TPR, FPR),
               Algorithm=toupper(Algorithm)) %>%
        mutate_if(is.character, as.factor) %>%
        group_by(Feature, ID, Algorithm) %>%
        summarize_at(vars(c("TPR", "FPR")), median)
    df$ID <- factor(df$ID,levels=c("RFE (0.1)","RFE (100)","dRFE (0.1)","dRFE (0.2)"))
    sca = ggscatter(df, x="FPR", y="TPR", color="Algorithm", palette="npg",
                    facet.by="ID", legend="bottom", add.params=list(alpha=0.5),
                    ncol=4, panel.labs.font=list(face="bold"),
                    xlab="Specificity (FPR)", ylab="Sensitivity (TPR)",
                    ggtheme=theme_pubr(base_size=15, border=TRUE)) +
        font("xylab", face="bold")
    save_plot(sca, outfile, 12, 4)
}

plot_metrics <- function(metric, ylab){
    outfile = paste0("boxplot_classification_simulation_", metric)
    bxp = cal_metrics() %>%
        ggboxplot(x="ID", y=metric, color="Algorithm", add="jitter",
                  palette="npg", xlab="", ylab=ylab, add.params=list(alpha=0.5),
                  outlier.shape=NA, legend="bottom",
                  ggtheme=theme_pubr(base_size=20, border=TRUE)) +
        rotate_x_text(45) + font("ylab", face="bold")
    save_plot(bxp, outfile, 5, 6)
}

## Main section
print_summary()
plot_metrics("Acc", "Accuracy")
plot_metrics("FDR", "False Discovery Rate")
plot_metrics("CPU", "Computational Time")
plot_roc()

## Reproducibility information
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
