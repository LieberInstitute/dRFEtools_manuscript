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

## Reproducibility information
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
