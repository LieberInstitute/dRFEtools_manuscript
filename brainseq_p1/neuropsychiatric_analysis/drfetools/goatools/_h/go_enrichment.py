"""
This script calculates gene ontology enrichment in python with GOATOOLS.
"""
import functools
import numpy as np
import pandas as pd
from os import environ
from pybiomart import Dataset
from collections import Counter
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# GO analysis
from goatools.obo_parser import GODag
from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

@functools.lru_cache()
def get_database():
    dataset = Dataset(name="hsapiens_gene_ensembl",
                      host="http://www.ensembl.org",
                      use_cache=True)
    db = dataset.query(attributes=["ensembl_gene_id",
                                   "external_gene_name",
                                   "entrezgene_id"],
                       use_attr_names=True).dropna(subset=['entrezgene_id'])
    return db


@functools.lru_cache()
def get_res():
    return pd.read_csv("../../../_m/residualized_expression.tsv",
                       sep='\t', index_col=0)


@functools.lru_cache()
def get_background():
    df = get_res()
    df["ensembl_gene_id"] = df.index.str.replace("\\..*", "", regex=True)
    df["Geneid"] = df.index
    return pd.merge(get_database(), df.loc[:, ["Geneid", "ensembl_gene_id"]],
                    on="ensembl_gene_id")


@functools.lru_cache()
def get_redundant():
    return pd.read_csv("../../metrics_summary/_m/dRFE_redundant_features.txt.gz",
                       sep='\t')


@functools.lru_cache()
def convert2entrez():
    background = get_background().dropna(subset=['entrezgene_id'])
    features = get_background().merge(get_redundant(), on="Geneid")\
                               .dropna(subset=["entrezgene_id"])
    return background, features


def obo_annotation(alpha=0.05):
    # background
    bg, _ = convert2entrez()
    # database annotation
    fn_obo = download_go_basic_obo()
    fn_gene2go = download_ncbi_associations() # must be gunzip to work
    obodag = GODag(fn_obo) # downloads most up-to-date
    anno_hs = Gene2GoReader(fn_gene2go, taxids=[9606])
    # get associations
    ns2assoc = anno_hs.get_ns2assc()
    for nspc, id2gos in ns2assoc.items():
        print("{NS} {N:,} annotated human genes".format(NS=nspc, N=len(id2gos)))
    goeaobj = GOEnrichmentStudyNS(
        bg['entrezgene_id'], # List of human genes with entrez IDs
        ns2assoc, # geneid/GO associations
        obodag, # Ontologies
        propagate_counts = False,
        alpha = alpha, # default significance cut-off
        methods = ['fdr_bh'])
    return goeaobj


def run_goea():
    _, df = convert2entrez()
    geneids_study = {z[0]:z[1] for z in zip(df['entrezgene_id'],
                                            df['external_gene_name'])}
    goeaobj = obo_annotation()
    goea_results_all = goeaobj.run_study(geneids_study)
    goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
    ctr = Counter([r.NS for r in goea_results_sig])
    print('Significant results[{TOTAL}] = {BP} BP + {MF} MF + {CC} CC'.format(
        TOTAL=len(goea_results_sig),
        BP=ctr['BP'],  # biological_process
        MF=ctr['MF'],  # molecular_function
        CC=ctr['CC'])) # cellular_component
    goeaobj.wr_xlsx("GO_analysis.xlsx", goea_results_sig)
    goeaobj.wr_txt("GO_analysis.txt", goea_results_sig)


def main():
    environ['NUMEXPR_MAX_THREADS'] = '32'
    run_goea()


if __name__ == '__main__':
    main()
