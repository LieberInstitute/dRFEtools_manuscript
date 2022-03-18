## This R script examines and plots the accuracy of RFE versus dRFE.

suppressPackageStartupMessages({
    library(logr)
    library(dplyr)
    library(ggpubr)
})

save_plot <- function(p, fn, w, h){
    for(ext in c(".pdf", ".png", ".svg")){
        ggsave(filename=paste0(fn,ext), plot=p, width=w, height=h)
    }
}

cal_metrics <- function(){
    fn = "binary_test_accuracy_metrics.tsv"
    return(data.table::fread(fn) %>%
           mutate(Acc=(TP+TN)/(TP+FN+TN+FP), FDR=(1-(TP/(TP+FP))),
                  F1_score=(2*TP)/(2*TP + FP + FN), TNR=(TN/(FP+TN))))
}

print_summary <- function(){
    print("Mean:")
    df.mean <- cal_metrics() %>%
        group_by(Diagnosis) %>%
        summarize_at(vars(c("Acc", "FDR", "F1_score", "TNR")), mean)
    print(df.mean)
    print("Median:")
    df.median <- cal_metrics() %>%
        group_by(Diagnosis) %>%
        summarize_at(vars(c("Acc", "FDR", "F1_score", "TNR")), median)
    print(df.median)
}

plot_metrics <- function(metric, ylab){
    outfile = paste0("boxplot_test_brainseq_", tolower(metric))
    bxp = cal_metrics() %>%
        ggboxplot(x="Diagnosis", y=metric, color="Diagnosis", add="jitter",
                  palette="npg", xlab="", ylab=ylab, add.params=list(alpha=0.5),
                  outlier.shape=NA, legend="bottom",
                  ggtheme=theme_pubr(base_size=20, border=TRUE)) +
        rotate_x_text(45) + font("ylab", face="bold")
    save_plot(bxp, outfile, 3, 6)
}

##### Main section
                                        # Start log
lf <- log_open("summary.log", logdir=FALSE, autolog=FALSE)
log_print("Summary of accuracy by diagnosis:")
print_summary() %>% log_print
                                        # End logging
log_close()
                                        # Plotting
plot_metrics("Acc", "Accuracy")
plot_metrics("FDR", "False Discovery Rate")

## Reproducibility information
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
