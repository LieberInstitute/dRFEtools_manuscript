## Author: Kynon J Benjamin
##
## This script extracts SNPs with a 500kbp window around the start and end
## position of a gene.
##
## Input: Positional input for feature annotation, plink format of SNPs (BFILE),
## chromosome information, and plink2 PATH
##
## Output: BFILE (BED/BIM/FAM) for each feature being analyzed.

PLINK_PATH="/ceph/opt/plink-ng/2.0/bin"
CHR_INFO="/ceph/genome/human/gencode26/GRCh38.p10.ALL/chromInfo/_m/chromInfo.txt"
GENO="/ceph/projects/brainseq/genotype/download/topmed/imputation_filter/convert2plink/filter_maf_01/_m/LIBD_Brain_TopMed"

# Get the gene positions +/- 500 kb
GNAME=`echo $1 | awk '{print $1}'`
CHR=`echo $1 | awk '{print $2}' | sed 's/chr//' -`
P0=`echo $1 | awk '{print $3 - 5e5}' | awk '{if($1<0)print 0;else print}'`
P1=`echo $1 | awk '{print $4 + 5e5}'`
P1=`awk -vn=$CHR -ve=$P1 '$1=="chr"n {if(e > $2)print $2;else print e}' $CHR_INFO`
echo $GNAME $CHR $P0 $P1
$PLINK_PATH/plink2 --bfile $GENO --from-bp $P0 --to-bp $P1 --chr $CHR \
                   --make-bed --out $GNAME
