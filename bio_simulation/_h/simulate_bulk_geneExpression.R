suppressPackageStartupMessages({
    library(SummarizedExperiment)
    library(recount3)
    library(SPsimSeq)
    library(dplyr)
})
cat("SPsimSeq package version",
    as.character(packageVersion("SPsimSeq")), "\n")
set.seed(20210929)
                                        # Load and filter data
human_projects <- available_projects()
project_info = human_projects %>%
    filter(file_source == "tcga", project == "LUAD") %>%
    select("project", "organism", "project_home")
rse_gene = create_rse(project_info[1, ])
keepIndex = which((rse_gene$tcga.cgc_sample_sample_type != "Recurrent Tumor"))
rse_gene = rse_gene[, keepIndex]
x <- edgeR::DGEList(counts=assays(rse_gene)$raw_counts,
                    samples=colData(rse_gene),
                    genes=rowData(rse_gene))
design0 <- model.matrix(~tcga.cgc_sample_sample_type, data=x$samples)
keep.x <- edgeR::filterByExpr(x, design=design0)
x <- x[keep.x, , keep.lib.sizes=FALSE]
                                        # Simulate data
sim.data.bulk <- SPsimSeq(n.sim=10, s.data=x$counts, n.genes=20000,
                          group=x$samples$tcga.cgc_sample_sample_type,
                          batch.config=1, group.config=c(0.6, 0.4),
                          tot.samples=400, pDE=0.005, lfc.thrld=0.10,
                          result.format="list")
                                        # Save data
outdir = "bulk_data"
dir.create(outdir)
for(ii in 1:10){
    sim.data <- sim.data.bulk[[ii]]
    sim.data$counts %>% as.data.frame %>%
        data.table::fwrite(paste0(outdir, "/simulated_counts_",ii,".tsv.gz"),
                           sep="\t", row.names=TRUE)
    sim.data$colData %>% as.data.frame %>%
        data.table::fwrite(paste0(outdir, "/simulated_sampleInfo_",ii,".tsv"),
                           sep="\t", row.names=TRUE)
    sim.data$rowData %>% as.data.frame %>%
        data.table::fwrite(paste0(outdir, "/simulated_geneInfo_",ii,".tsv.gz"),
                           sep="\t", row.names=TRUE)

}
## rse_gene_heart = create_rse(project_info[1,])
## keepIndex = which((rse_gene_heart$gtex.smtsd == "Heart - Left Ventricle"))
## rse_gene_heart = rse_gene_heart[, keepIndex]
## rse_gene_brain = create_rse(project_info[2,])
## keepIndex = which((rse_gene_brain$gtex.smtsd == "Brain - Hippocampus"))
## rse_gene_brain = rse_gene_brain[, keepIndex]
                                        # Filter low expressing features
## counts <- cbind(assays(rse_gene_brain)$raw_counts,
##                 assays(rse_gene_heart)$raw_counts)
## pheno <- rbind(colData(rse_gene_brain), colData(rse_gene_heart))
