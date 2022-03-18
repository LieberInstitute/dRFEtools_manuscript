## Author: Kynon J Benjamin
##
## This script runs random forest with dynamic recusive feature elimination.
##
## Input: Positional input of SNPs with gene expression and phenotype.
##
## Output: dRFEtools output for random forest regression.

PHENO_FILE="../../../../_m/phenotypes_bsp1.tsv"

export geneid=`basename $1`
python ../_h/dRFE_rf.py --fn $1 --pheno_file $PHENO_FILE 2> \
       log_files/$geneid.log 1> log_files/$geneid.log
