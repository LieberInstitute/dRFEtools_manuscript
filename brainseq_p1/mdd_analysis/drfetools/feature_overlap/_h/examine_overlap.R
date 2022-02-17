## This script examines shared genes between DE and predictive analysis
library(ggvenn)
library(magrittr)

save_plot <- function(p, fn, w=7, h=7){
    for(EXT in c(".png", ".pdf")){
        ggplot2::ggsave(paste0(fn, EXT), plot=p, width=w, height=h)
    }
}

get_deg <- function(){
    fn <- "../../../differential_analysis/_m/diffExpr_CTLvsMDD_full.txt"
    df <- data.table::fread(fn) %>% dplyr::filter(adj.P.Val < 0.05)
    return(df$gencodeID)
}
memDEG <- memoise::memoise(get_deg)

get_predictive <- function(){
    fn <- "../../metrics_summary/_m/dRFE_predictive_features.txt.gz"
    df <- data.table::fread(fn) %>% dplyr::filter(Method == "RF")
    return(df$Geneid)
}
memRF <- memoise::memoise(get_predictive)

get_redundant <- function(){
    fn <- "../../metrics_summary/_m/dRFE_redundant_features.txt.gz"
    df <- data.table::fread(fn) %>% dplyr::filter(Method == "RF")
    return(df$Geneid)
}
memML <- memoise::memoise(get_redundant)

plot_venn <- function(){
    list_venn1 <- list(DEG=memDEG(), RF=memRF())
    list_venn2 <- list(DEG=memDEG(), RF=memML())
    v1 <- ggvenn(list_venn1, c("DEG", "RF"))
    v2 <- ggvenn(list_venn2, c("DEG", "RF"))
    save_plot(v1, "predictive_VS_degs")
    save_plot(v2, "redundant_VS_degs")
}

extract_overlap <- function(){
    data.frame("Geneid"=intersect(memDEG(), memRF())) %>%
        data.table::fwrite("shared_genes.tsv", sep='\t')
}

##### MAIN ########
main <- function(){
    extract_overlap()
    plot_venn()
}

main()

## Reproducibility
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
