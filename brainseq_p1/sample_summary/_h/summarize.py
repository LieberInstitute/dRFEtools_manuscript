"""
This script aggregates data by diagnosis and prints
summary tables to the log file.
"""
import functools
import pandas as pd

@functools.lru_cache()
def get_pheno():
    return pd.read_csv("../../_m/phenotypes_bsp1.tsv", sep='\t', index_col=0)


def main():
    df = get_pheno()[(get_pheno()["Age"] > 17) &
                     (get_pheno()["Race"].isin(["AA", "CAUC"]))]
    with open("summarize.log", "w") as f:
        dx = df.loc[:, ["Dx", "Race", "Sex", "Age", "RIN"]]
        print(dx.groupby(["Dx"]).size(), file=f)
        print(dx.groupby(["Dx", "Sex"]).size(), file=f)
        print(dx.groupby(["Dx", "Race"]).size(), file=f)
        print("Mean:", file=f)
        print(dx.groupby(["Dx"]).mean(), file=f)
        print("Standard deviation", file=f)
        print(dx.groupby(["Dx"]).std(), file=f)


if __name__ == "__main__":
    main()
