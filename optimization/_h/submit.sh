#!/bin/sh
#$ -cwd
#$ -R y
#$ -l mem_free=10.0G,h_vmem=10G,h_fsize=50G
#$ -N 'drfetools_optimization'
#$ -o ./summary.out
#$ -e ./summary.out
#$ -m e -M jade.benjamin@libd.org

set -o errexit -o pipefail

echo "**** Job starts ****"
date -Is

echo "**** JHPCE info ****"
echo "User: ${USER}"
echo "Job id: ${JOB_ID}"
echo "Job name: ${JOB_NAME}"
echo "Hostname: ${HOSTNAME}"

module load pandoc
module load R/4.0.3
module list

echo "**** Run jupyter notebook ****"

for NOTEBOOK in optimization; do
    cp ../_h/${NOTEBOOK}.ipynb tmp_${NOTEBOOK}.ipynb
    jupyter-nbconvert --execute --ExecutePreprocessor.timeout=-1 --to notebook \
                      --stdout tmp_${NOTEBOOK}.ipynb > ${NOTEBOOK}.ipynb
    jupyter nbconvert --to html ${NOTEBOOK}.ipynb
    jupyter nbconvert --to pdf ${NOTEBOOK}.ipynb
    rm -f tmp_${NOTEBOOK}.ipynb
done

echo "**** Job ends ****"
date -Is
