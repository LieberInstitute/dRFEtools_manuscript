## This script performs differential expression analysis for MDD.suppressMessages({
    library(dplyr)
})

save_volcanoPlot <- function(top, label, feature){
    pdf(file=paste0(feature, "/volcanoPlot_", label, ".pdf"), 8, 6)
    with(top, plot(logFC, -log10(P.Value), pch=20, cex=0.6))
    with(subset(top, adj.P.Val<=0.05), points(logFC, -log10(P.Value),
                                              pch=20, col='red', cex=0.6))
    with(subset(top, abs(logFC)>0.50), points(logFC, -log10(P.Value),
                                              pch=20, col='orange', cex=0.6))
    with(subset(top, adj.P.Val<=0.05 & abs(logFC)>0.50),
         points(logFC, -log10(P.Value), pch=20, col='green', cex=0.6))
    dev.off()
}

save_MAplot <- function(top, label, feature){
    pdf(file=paste0(feature, "/MAplot_", label, ".pdf"), 8, 6)
    with(top, plot(AveExpr, logFC, pch=20, cex=0.5))
    with(subset(top, adj.P.Val<0.05),
         points(AveExpr, logFC, col="red", pch=20, cex=0.5))
    dev.off()
}

extract_de <- function(contrast, label, efit, feature){
    top <- limma::topTable(efit, coef=contrast, number=Inf, sort.by="P")
    top$SE <- sqrt(efit$s2.post) * efit$stdev.unscaled
    top.fdr <- top %>% filter(adj.P.Val<=0.05)
    print(paste("Comparison for:", label))
    print(paste('There are:', dim(top.fdr)[1], 'DE features!'))
    data.table::fwrite(top,
                       file=paste0(feature, "/diffExpr_", label, "_full.txt"),
                       sep='\t', row.names=TRUE)
    data.table::fwrite(top.fdr,
                       file=paste0(feature, "/diffExpr_", label, "_FDR05.txt"),
                       sep='\t', row.names=TRUE)
    save_volcanoPlot(top, label, feature)
    save_MAplot(top, label, feature)
}

get_voom <- function(){
    load('../../_m/voomSVA.RData')
    return(v)
}

fit_voom <- function(){
    v       <- get_voom()
    modQsva <- v$design
    fit0 <- limma::lmFit(v, modQsva)
    contr.matrix <- limma::makeContrasts(CTLvsMDD=MDD,
                                         levels=colnames(modQsva))
    fit <- limma::contrasts.fit(fit0, contrasts=contr.matrix)
    esv <- limma::eBayes(fit)
    return(esv)
}
memFIT <- memoise::memoise(fit_voom)

##### MAIN ######
main <- function(){
    outdir = "."
                                        # Fit with eBayes
    efit <- memFIT()
                                        # Extract DE results
    extract_de(1, "CTLvsMDD", efit, outdir)
}

main()

## Reproducibility
Sys.time()
proc.time()
options(width = 120)
sessioninfo::session_info()
