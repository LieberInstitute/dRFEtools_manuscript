## Summarize results and plot correlation with q-value

suppressPackageStartupMessages({
    library(dplyr)
    library(gtsummary)
})

save_table <- function(pp, fn){
    for(ext in c(".pdf", ".tex")){
        gt::gtsave(as_gt(pp), filename=paste0(fn,ext))
    }
}

load_phenotypes <- function(){
    return(data.table::fread("../../_m/phenotypes_bsp1.tsv") %>%
           mutate_if(is.character, as.factor) %>%
           filter(Race %in% c("AA", "CAUC"), Age > 17, Dx != "Bipolar"))
}
memPHENO <- memoise::memoise(load_phenotypes)

#### MAIN
                                         # Generate pretty tables
fn = "sample_breakdown.table"
pp = memPHENO() %>% select(Dx, Race, Sex, Age, RIN) %>%
    mutate(Sex=factor(Sex, labels=c("Female", "Male")),
           Race=factor(forcats::fct_drop(Race), labels=c("AA", "EA")),
           Dx=factor(forcats::fct_drop(Dx), labels=c("CTL", "MDD", "SZ"))) %>%
    tbl_summary(by="Dx", missing="no",
                statistic=all_continuous() ~ c("{mean} ({sd})")) %>%
    modify_header(all_stat_cols()~"**{level}**<br>N = {n}") %>%
    modify_spanning_header(all_stat_cols()~"**Diagnosis**") %>%
    bold_labels() %>% italicize_levels()
save_table(pp, fn)

#### Reproducibility information
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
