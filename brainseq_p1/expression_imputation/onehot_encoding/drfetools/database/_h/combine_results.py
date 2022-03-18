"""
This script combines the results into easily accessible files for
downstream analysis.
"""
import re
import pandas as pd
from glob import iglob

def combine_results(fn):
    df = pd.DataFrame()
    for filename in iglob("../../_m/*/%s" % (fn)):
        m = re.search(r'../../_m/(.+?)/', filename)
        if m:
            feature_name = m.group(1)
            tmp_df = pd.read_csv(filename, sep='\t')
            tmp_df['feature'] = feature_name
            df = pd.concat([df, tmp_df], axis=0)
    return df


def get_ranks():
    rank = pd.DataFrame()
    for filename in iglob("../../_m/*/rank_features.txt"):
        m = re.search(r'../../_m/(.+?)/', filename)
        if m:
            feature_name = m.group(1)#.replace('_', '.', regex=True)
            tmp_df = pd.read_csv(filename,sep='\t',names=['SNP','Fold','Rank'])
            tmp_df['Feature'] = feature_name
            rank = pd.concat([rank, tmp_df], axis=0)
    return rank


def main():
    # Summary stats
    combine_results("dRFEtools_10folds.txt")\
        .to_csv("dRFEtools_10Folds.txt.gz", sep='\t', index=False)
    # Test predictions
    combine_results("test_predictions.txt")\
        .rename(columns={'Unnamed: 0': 'RNum'})\
        .set_index('RNum')\
        .to_csv("test_predictions.txt.gz", sep='\t', index=True)
    # Elimination results
    combine_results("feature_elimination_allFolds_metrics.txt")\
        .to_csv("drfe_metrics.txt.gz", sep='\t', index=False)
    # Rank features
    get_ranks().to_csv("rank_features.txt.gz", sep='\t', index=False)


if __name__ == '__main__':
    main()
