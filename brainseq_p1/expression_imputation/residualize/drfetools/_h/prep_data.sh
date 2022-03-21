## Author: Kynon J Benjamin
##
## This script converts plink SNPs into onehot encoded values. This scripts
## takes positional input to to run in parallel.
##
## Input: Features from positional input, phenotype information,
## residualized expression, and plink file location.
##
## Output: Onehot encoded text file of SNPs for each features.

PLINK_FILES="../../../_m"
PHENO_FILE="../../../../_m/phenotypes_bsp1.tsv"
EXPR_FILE="../../_m/residualized_expression.tsv"

GNAME=`echo $1 | awk '{print $1}' `
ii=`echo $GNAME | sed 's/\./_/' `

if [ -f "$PLINK_FILES/$GNAME.fam" ];then
    mkdir $ii
    python ../_h/genotype_data_extractor.py \
           --plink_file_prefix $PLINK_FILES/$GNAME --dirname $ii \
           --pheno_file $PHENO_FILE --expr_file $EXPR_FILE
else
    echo "$GNAME has no SNPs within the region!"
fi
