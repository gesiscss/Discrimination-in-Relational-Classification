#!/bin/bash
CSVFN=$1
NFOLDS=$2
NJOBS=$3
MAXFEATURES=$4
ROOT=$5
FIGEXT=$6
if [[ $# -ne 6 ]] ; then
    echo 'CVSFN | NFOLDS | NJOBS | MAXFEATURES | ROOT | FIGEXT : arguments are missing.'
    exit 1
fi
echo $CSVFN $NFOLDS $MAXFEATURES $ROOT;

DIRECTORY="$ROOT/BAHm4-S3D/$NFOLDS-FOLDS/$MAXFEATURES-MAXFEAT"

if [ ! -d "$DIRECTORY" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    python3.5 feature_selection.py -opt s3d-folds -root $ROOT -fn $CSVFN -num_folds $NFOLDS -max_features $MAXFEATURES -njobs $NJOBS;
    python3.5 feature_selection.py -opt s3d-crossval -root $ROOT -fn $CSVFN -num_folds $NFOLDS -njobs $NJOBS -max_features $MAXFEATURES;
    python3.5 feature_selection.py -opt s3d-evaluation -root $ROOT -fn $CSVFN -num_folds $NFOLDS -njobs $NJOBS -max_features $MAXFEATURES -figext $FIGEXT;
    python3.5 feature_selection.py -opt s3d-viz-model -root $ROOT -fn $CSVFN -num_folds $NFOLDS -njobs $NJOBS -max_features $MAXFEATURES -figext $FIGEXT;
    python3.5 feature_selection.py -opt s3d-viz-binning -root $ROOT -fn $CSVFN -num_folds $NFOLDS  -njobs $NJOBS -max_features $MAXFEATURES -figext $FIGEXT;
    python3.5 feature_selection.py -opt s3d-features -root $ROOT -fn $CSVFN -num_folds $NFOLDS -njobs $NJOBS -max_features $MAXFEATURES -figext $FIGEXT;
    python3.5 feature_selection.py -opt s3d-performance -root $ROOT -fn $CSVFN -num_folds $NFOLDS -njobs $NJOBS -max_features $MAXFEATURES -figext $FIGEXT;
else
    echo "already done. nothing to do"
    exit 0
fi