"""
Explore variation between control and MDD.
"""
import numpy as np
import pandas as pd
from plotnine import *
from functools import lru_cache
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@lru_cache()
def get_predictive(dx):
    '''
    Load RF predictive features from dRFEtools
    '''
    fn = '../../metrics_summary/_m/dRFE_predictive_features.txt.gz'
    df = pd.read_csv(fn, sep='\t', index_col=0)
    return df[(df['Diagnosis'] == dx)].copy()


@lru_cache()
def get_overlap(dx):
    """
    Load genes that are shared between DE and predictive.
    """
    fn = "../../feature_overlap/_m/shared_genes_%s.tsv" % dx.lower()
    return pd.read_csv(fn, sep='\t', index_col=0)


@lru_cache()
def get_residualized(dx):
    '''
    Load residualization file.
    '''
    fn = '../../../_m/%s/residualized_expression.tsv' % dx.lower()
    return pd.read_csv(fn, sep='\t', index_col=0).transpose()


@lru_cache()
def get_pheno_data():
    """
    Load phenotype data.
    """
    fn = '../../../../_m/phenotypes_bsp1.tsv'
    return pd.read_csv(fn, index_col=0, sep='\t')


@lru_cache()
def get_res_df(fnc, dx):
    """
    Merge predictive and residualized data.
    """
    return get_residualized(dx)[list(fnc(dx).index)].copy()


def get_explained_variance(df):
    x = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2).fit(x)
    pc1 = pca.explained_variance_ratio_[0]
    pc2 = pca.explained_variance_ratio_[1]
    print("Explained Variance\nPC1:\t%0.5f\nPC2:\t%0.5f" % (pc1, pc2))


def cal_pca(df):
    x = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2).fit_transform(x)
    return pd.DataFrame(data=pca, columns=['PC1', 'PC2'], index=df.index)


def get_pca_df(fnc, dx):
    '''
    new_pheno: This gets the correct size of samples using the the first two
               columns of residualized expression
      - the residualized expression data frame, has the correct samples
      - output new_pheno shape row numbers should be the same as res_df row numbers
    '''
    expr_res = get_res_df(fnc, dx)
    # Generate pheno data frame with correct samples
    new_pheno = get_pheno_data().merge(expr_res.iloc[:, 0:1], right_index=True,
                                       left_index=True)\
                                .drop(expr_res.iloc[:, 0:1].columns, axis=1)
    principalDf = cal_pca(expr_res)
    get_explained_variance(expr_res)
    return pd.concat([principalDf, new_pheno], axis = 1)


def calculate_corr(xx, yy):
    '''This calculates R^2 correlation via linear regression:
         - used to calculate relationship between 2 arrays
         - the arrays are principal components 1 or 2 (PC1, PC2) AND ancestry
         - calculated on a scale of 0 to 1 (with 0 being no correlation)
        Inputs:
          x: array of variable of interest (continous or binary)
          y: array of PC
        Outputs:
          1. r2
          2. p-value, two-sided test
            - whose null hypothesis is that two sets of data are uncorrelated
          3. slope (beta): directory of correlations
    '''
    slope, intercept, r_value, p_value, std_err = linregress(xx, yy)
    return slope, r_value, p_value


def get_corr(dft):
    xx = dft.Dx.astype("category").cat.codes
    yy = dft.PC1
    slope1, r_value1, p_value1 = calculate_corr(xx, yy)
    return r_value1**2, p_value1


def corr_annotation(dft):
    rho, pval = get_corr(dft)
    label = 'PC1 R2: %.2f\nP-value: %.2e' % (rho, pval)
    return label


def plot_corr(fnc, dx):
    pca_df = get_pca_df(fnc, dx)
    pca_df['Dx'] = pca_df.Dx.astype('category').cat\
                           .rename_categories({'Control': 'CTL', "Schizo": "SZ",
                                               "Bipolar": "BD"})
    title = '\n'.join([corr_annotation(pca_df)])
    pp = ggplot(pca_df, aes(x='PC1', y='PC2', fill='Dx'))\
    + geom_point(alpha=0.75, size=4) + theme_matplotlib()\
    + theme(axis_text=element_text(size=18), plot_title=element_text(size=16),
            axis_title=element_text(size=21, face="bold"),
            legend_text=element_text(size=16), legend_title=element_blank(),
            legend_position="bottom")
    pp += ggtitle(title)
    return pp


def save_plot(p, fn, width=7, height=7):
    '''Save plot as svg, png, and pdf with specific label and dimension.'''
    for ext in ['.png', '.pdf']:
        p.save(fn+ext, width=width, height=height)


def main():
    for dx in ["MDD", "Schizo", "Bipolar"]:
        pp = plot_corr(get_predictive, dx)
        qq = plot_corr(get_overlap, dx)
        save_plot(pp, "predictive_variation_%s" % dx.lower())
        save_plot(qq, "shared_variation_%s" % dx.lower())


if __name__ == '__main__':
    main()
