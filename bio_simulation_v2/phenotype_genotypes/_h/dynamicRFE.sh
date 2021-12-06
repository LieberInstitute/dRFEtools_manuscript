#!/bin/sh
set -o errexit -o pipefail

for NOTEBOOK in dRFEtools; do
    cp ../_h/${NOTEBOOK}.ipynb tmp_${NOTEBOOK}.ipynb
    jupyter-nbconvert --execute --ExecutePreprocessor.timeout=-1 --to notebook \
                      --stdout tmp_${NOTEBOOK}.ipynb > ${NOTEBOOK}.ipynb
    jupyter nbconvert --to html ${NOTEBOOK}.ipynb
    jupyter nbconvert --to pdf ${NOTEBOOK}.ipynb
    rm -f tmp_${NOTEBOOK}.ipynb
done
