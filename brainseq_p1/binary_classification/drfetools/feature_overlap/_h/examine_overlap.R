## This script examines shared genes between DE and predictive analysis
library(ggvenn)
library(magrittr)

save_plot <- function(p, fn, w=7, h=7){
    for(EXT in c(".png", ".pdf")){
        ggplot2::ggsave(paste0(fn, EXT), plot=p, width=w, height=h)
    }
}

get_deg <- function(diagnosis){
    dx_lt <- list("MDD"="../../../differential_analysis/_m/mdd/diffExpr_CTLvsMDD_full.txt",
                  "Schizo"="../../../differential_analysis/_m/schizo/diffExpr_CTLvsSZ_full.txt")
    df <- data.table::fread(dx_lt[[diagnosis]]) %>%
        dplyr::filter(adj.P.Val < 0.05)
    return(df$gencodeID)
}
memDEG <- memoise::memoise(get_deg)

get_predictive <- function(diagnosis){
    fn <- "../../metrics_summary/_m/dRFE_predictive_features.txt.gz"
    df <- data.table::fread(fn) %>% dplyr::filter(Diagnosis == diagnosis)
    return(df$Geneid)
}
memRF <- memoise::memoise(get_predictive)

get_redundant <- function(diagnosis){
    fn <- "../../metrics_summary/_m/dRFE_redundant_features.txt.gz"
    df <- data.table::fread(fn) %>% dplyr::filter(Diagnosis == diagnosis)
    return(df$Geneid)
}
memML <- memoise::memoise(get_redundant)

plot_venn <- function(diagnosis){
    list_venn1 <- list(DEG=memDEG(diagnosis), RF=memRF(diagnosis))
    list_venn2 <- list(DEG=memDEG(diagnosis), RF=memML(diagnosis))
    v1 <- ggvenn(list_venn1, c("DEG", "RF"))
    v2 <- ggvenn(list_venn2, c("DEG", "RF"))
    save_plot(v1, paste("predictive_VS_degs", tolower(diagnosis), sep="_"))
    save_plot(v2, paste("redundant_VS_degs", tolower(diagnosis), sep="_"))
}

extract_overlap <- function(diagnosis){
    data.frame("Geneid"=intersect(memDEG(diagnosis), memRF(diagnosis))) %>%
        data.table::fwrite(paste0("shared_genes_",tolower(diagnosis),".tsv"),
                           sep='\t')
}

##### MAIN ########
main <- function(){
    for(dx in c("MDD", "Schizo")){
        extract_overlap(dx)
        plot_venn(dx)
    }
}

main()

## Reproducibility
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
