"""
This script calculates gene ontology enrichment in python with GOATOOLS.
"""
import functools
import numpy as np
import pandas as pd
from os import environ
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
def get_degs():
    return pd.read_csv("../../../../mdd_analysis/differential_analysis/_m/diffExpr_CTLvsMDD_full.txt",
                       sep='\t', index_col=0)


@functools.lru_cache()
def get_predictive():
    df = pd.read_csv("../../metrics_summary/_m/dRFE_predictive_features.txt.gz",
                     sep='\t')
    return df[(df["Method"] == "RF")].copy()


@functools.lru_cache()
def get_redundant():
    df = pd.read_csv("../../metrics_summary/_m/dRFE_redundant_features.txt.gz",
                     sep='\t')
    return df[(df["Method"] == "RF")].copy()


def convert2entrez():
    background = get_degs().dropna(subset=['EntrezID'])
    features = get_degs().merge(get_predictive(), left_index=True,
                                right_on="Geneid")\
                         .dropna(subset=["EntrezID"])
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
        bg['EntrezID'], # List of human genes with entrez IDs
        ns2assoc, # geneid/GO associations
        obodag, # Ontologies
        propagate_counts = False,
        alpha = alpha, # default significance cut-off
        methods = ['fdr_bh'])
    return goeaobj


def run_goea():
    _, df = convert2entrez()
    geneids_study = {z[0]:z[1] for z in zip(df['EntrezID'], df['Symbol'])}
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
