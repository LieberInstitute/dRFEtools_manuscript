"""
This script clusters samples using predictive features and TSNE.
"""
import pandas as pd
from plotnine import *
from functools import lru_cache
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@lru_cache()
def get_predictive():
    '''
    Load RF predictive features from dRFEtools
    '''
    fn = '../../metrics_summary/_m/dRFE_predictive_features.txt.gz'
    return pd.read_csv(fn, sep='\t', index_col=0)


@lru_cache()
def get_residualized():
    '''
    Load residualization file.
    '''
    fn = '../../../_m/residualized_expression.tsv'
    return pd.read_csv(fn, sep='\t', index_col=0).transpose()


@lru_cache()
def get_pheno_data():
    """
    Load phenotype data.
    """
    fn = '../../../../_m/phenotypes_bsp1.tsv'
    return pd.read_csv(fn, index_col=0, sep='\t')


@lru_cache()
def get_res_df():
    """
    Merge predictive and residualized data.
    """
    return get_residualized()[list(get_predictive().index)].copy()


def cal_pca(df):
    """
    Calculates PCA and results a data frame with results
    """
    x = StandardScaler().fit_transform(df)
    pca = PCA().fit_transform(x)
    cols = ["PC%d" % x for x in range(1, pca.shape[1]+1)]
    return pd.DataFrame(data=pca, columns=cols, index=df.index)


def cal_tsne(df):
    """
    This calculates TSNE with dataframe input
    """
    return TSNE(learning_rate="auto", init="random", perplexity=20,
                random_state=13, n_jobs=-1).fit_transform(df)


def get_tsne_df():
    '''
    new_pheno: This gets the correct size of samples using the the first two
               columns of residualized expression
      - the residualized expression data frame, has the correct samples
      - output new_pheno shape row numbers should be the same as res_df row numbers
    '''
    expr_res = get_res_df()
    # Generate pheno data frame with correct samples
    new_pheno = get_pheno_data().merge(get_res_df().iloc[:, 0:1],
                                       right_index=True, left_index=True)\
                                .drop(get_res_df().iloc[:, 0:1].columns, axis=1)
    tsne_df = pd.DataFrame(data=cal_tsne(cal_pca(get_res_df())),
                           columns=["D1", "D2"], index=get_res_df().index)
    return pd.concat([tsne_df, new_pheno], axis=1)


def plot_tsne():
    tsne_df = get_tsne_df()
    tsne_df['Dx'] = tsne_df.Dx.astype('category').cat\
                           .rename_categories({'Control': 'CTL', "Schizo": "SZ",
                                               "Bipolar": "BD"})
    pp = ggplot(tsne_df, aes(x='D1', y='D2', fill='Dx'))\
    + geom_point(alpha=0.75, size=4) + theme_matplotlib()\
    + theme(axis_text=element_text(size=18), plot_title=element_text(size=16),
            axis_title=element_text(size=21, face="bold"),
            legend_text=element_text(size=16), legend_title=element_blank(),
            legend_position="bottom")
    return pp


def save_plot(p, fn, width=7, height=7):
    '''Save plot as svg, png, and pdf with specific label and dimension.'''
    for ext in ['.png', '.pdf']:
        p.save(fn+ext, width=width, height=height)


def main():
    pp = plot_tsne()
    save_plot(pp, "predictive_clustering")


if __name__ == "__main__":
    main()
