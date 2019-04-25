import matplotlib
matplotlib.use('Agg')


from libs.utils.arguments import ArgumentsHandler
from libs.utils.loggerinit import *

import libs.s3d.utils as utils
from libs.s3d.split_data import main as split_data
import pandas as pd
import numpy as np
import six
import operator
import sys

from libs.s3d.pys3d import PYS3D
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import confusion_matrix

###################################################################################################
# CONSTANTS
###################################################################################################

MUST = ['opt', 'root', 'fn']
W = 10
H = 10
METRIC = 'mae'
max_features = 15
pink_color = '#FBB4AE'

###################################################################################################
# SIGALARM (BINNING PLOTS)
###################################################################################################

import signal

class TimeoutException(Exception):  # Custom exception class
    pass

def timeout_handler(signum, frame):  # Custom signal handler
    logging.info('Timeout')
    # raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)
TIMEOUT = 60

###################################################################################################
# FUNCTIONS
###################################################################################################

def s3d_cross_validation(dataset, output, cf, max_features, num_jobs):
    s3d = PYS3D(dataset, output, cf)
    lambda_list = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.00001, 0.00002, 0.00003]
    s3d.cross_val_multicore(lambda_list, max_features, calc_threshold=True, num_cores=num_jobs)

def s3d_evaluation(dataset, output, cf, metric, num_jobs, ext):
    s3d = PYS3D(dataset, output, cf)
    metric = s3d.validate_metric(metric)
    df_eval = s3d.evaluate(cv_metric=metric, num_jobs=num_jobs)
    print(df_eval)
    s3d_plot_evaluation(s3d, df_eval, ext)

def s3d_viz_best_performance(path, sampling, metric):
    s3d = PYS3D(path, sampling, flush=False)
    cv_param_df = utils.find_best_param(s3d.cv_performance_file, validation_metric=metric)
    print(cv_param_df)


###################################################################################################
# VIZ FUNCTIONS
###################################################################################################

def s3d_plot_evaluation(s3d, df_eval, ext):
    df_eval.dropna(axis=1, inplace=True)

    columns = list(df_eval.columns)

    _features = set([])
    for index, row in df_eval.iterrows():
        features = s3d.get_features(int(row['split_version']))
        _features |= set(features)
        for f in features:
            df_eval.loc[index, f] = True

    _features = sorted(_features)
    columns.extend(_features)
    df_eval = df_eval[columns]

    for c in df_eval.columns:
        if c not in _features:
            try:
                if 'lambda' not in c:
                    df_eval[c] = df_eval[c].astype(np.float64).round(2)
            except:
                pass

    df_eval = df_eval.append(df_eval.sum(),ignore_index=True)
    df_eval.fillna('', inplace=True)

    last_row = len(df_eval)
    for c in df_eval.columns:
        if c not in _features:
            df_eval.loc[last_row-1,c] = ''

    renames = {c: ''.join([x[0].upper() for x in c.split('_') if len(x) > 0]) for c in df_eval.columns if len(c.split('_')) > 1}
    df_eval.rename(columns=renames, inplace=True)

    ax = render_mpl_table(df_eval, header_columns=0, col_width=0.9)
    renames = sorted(renames.items(), key=operator.itemgetter(0))
    ax.text(-0.05, 0, '\n'.join(['{}: {}'.format(short,real) for real,short in renames]), rotation=0, wrap=True, fontsize=9)

    ax.set_title(sampling.upper())
    fn = os.path.join(s3d.viz_path, 'viz-summary.{}'.format(ext))
    plt.savefig(fn, bbox_inches='tight')
    logging.info('{} saved!'.format(fn))
    plt.close()

    fn = os.path.join(s3d.viz_path, 'viz-summary.csv')
    df_eval.to_csv(fn)
    logging.info('{} saved!'.format(fn))

def render_mpl_table(data, col_width=0.6, row_height=0.5, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0.12, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([6, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, rowLabels=data.index, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)

        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def s3d_viz_model(dataset, output, cf, num_jobs, ext):
    s3d = PYS3D(dataset, output, cf)
    Parallel(n_jobs=num_jobs)(delayed(_s3d_viz_model)(s3d, num_fold, ext) for num_fold in range(s3d.num_folds))

def _s3d_viz_model(s3d, num_fold, ext):
    fig, ax = utils.visualize_s3d_steps(os.path.join(s3d.model_path,str(num_fold)), figsize=(W, H))
    fn = os.path.join(s3d.viz_path, 'viz-{}-model.{}'.format(num_fold,ext))
    fig.savefig(fn, bbox_inches='tight')
    logging.info('{} saved!'.format(fn))
    plt.close()

def s3d_viz_binning(dataset, output, cf, num_jobs, ext):
    s3d = PYS3D(dataset, output, cf)
    Parallel(n_jobs=num_jobs)(delayed(_s3d_viz_binning)(s3d, num_fold, ext) for num_fold in range(s3d.num_folds))

def _s3d_viz_binning(s3d, num_fold, ext):

    features = s3d.get_features(num_fold)
    dim = len(features)

    thres = None
    if s3d.classification_flag:

        thres = None
        while thres is None:
            try:
                thres = s3d.calculate_disc_threshold(os.path.join(s3d.model_path, str(num_fold)), dim)
            except Exception as ex:
                print('ERROR calculating threshold: num_fold:{}, dim:{}'.format(num_fold,dim))
                dim -= 1

            if dim <= 0:
                print('ERROR dim<=0')
                return

    dim = min(4, dim)
    splits_at_dim, N_dim, intensity_dim, pred_dim, chosen_f_dim = utils.visualize_s3d_model_reader(os.path.join(s3d.model_path,str(num_fold)), dim, 0 if thres is None else thres)

    # FREQUENCY
    my_cmap = mc.LinearSegmentedColormap.from_list('custom_pink', ['#ffffff', pink_color], N=256)
    _s3d_viz_binning_per_type(s3d, 'freq', dim, splits_at_dim, my_cmap, N_dim, 'Freq', chosen_f_dim, num_fold, mc.LogNorm, None, ext)

    if s3d.classification_flag:
        # PREDICTION
        my_cmap = mc.ListedColormap(['white']*3 + [pink_color])
        _s3d_viz_binning_per_type(s3d, 'pred', dim, splits_at_dim, my_cmap, pred_dim, 'Pred.', chosen_f_dim, num_fold, None, thres, ext)


    else:
        # EXPECTED VALUE
        my_cmap = mc.LinearSegmentedColormap.from_list('custom_pink', ['#ffffff', pink_color], N=256)
        _s3d_viz_binning_per_type(s3d, 'expe', dim, splits_at_dim, my_cmap, intensity_dim, '$E[y]$.', chosen_f_dim, num_fold, None, None, ext)

def _s3d_viz_binning_per_type(s3d, ttype, dim, splits_at_dim, cmap, values, values_label, chosen_f_dim, num_fold, norm_func, thres, ext):

    signal.alarm(TIMEOUT) # seconds
    fn = os.path.join(s3d.viz_path, 'viz-{}-binning-{}.{}'.format(num_fold, ttype, ext))

    cb_kw = {'aspect': 30}
    if thres is not None:
        cb_kw['ticks'] = [0, thres, 1] #assuming classfication is binary: 0 or 1

    try:

        if dim > 1:
            fig, ax_arr = utils.visualize_s3d_model(dim, splits_at_dim, cmap,
                                                values, values_label, chosen_f_dim,
                                                xscale='linear', yscale='linear',
                                                fontsize=9,
                                                norm_func=norm_func,
                                                cb_kwargs=cb_kw)
            ax_arr[0, 0].minorticks_off()

        else:
            fig, ax = utils.visualize_s3d_model_1d(splits_at_dim, values,
                                                   xlab=chosen_f_dim[0], ylab=values_label,
                                                   xscale='linear',
                                                   hlines_kwargs={'color':pink_color, 'linewidth': 3})

        fig.savefig(fn, bbox_inches='tight')
        logging.info('{} saved!'.format(fn))
        plt.close()

    except TimeoutException:
        return

    except Exception as ex:
        plt.close()
        logging.info('{} NOT saved. {}'.format(fn,ex))
        signal.alarm(0)

def s3d_viz_network_features(dataset, output, cf, num_jobs, ext):
    s3d = PYS3D(dataset, output, cf)
    Parallel(n_jobs=num_jobs)(delayed(_s3d_viz_network_features)(s3d, num_fold, ext) for num_fold in range(s3d.num_folds))

def _s3d_viz_network_features(s3d, num_fold, ext):
    net, (fig, ax) = utils.visualize_feature_network(model_folder=os.path.join(s3d.model_path, str(num_fold)),
                                                     w_scale=5,
                                                     label_kwargs={'font_size':12},
                                                     edge_color='grey',
                                                     figsize=(30,30),
                                                     node_size=500)
    fn = os.path.join(s3d.viz_path, 'viz-{}-features.{}'.format(num_fold,ext))
    fig.savefig(fn, bbox_inches='tight')
    logging.info('{} saved!'.format(fn))
    plt.close()

def s3d_viz_performance(dataset, output, cf, metric, num_jobs, ext, classes):
    s3d = PYS3D(dataset, output, cf)
    metric = s3d.validate_metric(metric)
    Parallel(n_jobs=num_jobs)(delayed(_s3d_viz_performance)(s3d, num_fold, metric, ext, classes) for num_fold in range(s3d.num_folds))

def _s3d_viz_performance(s3d, num_fold, metric, ext, classes):

    # performance features
    fp, best_n_f, best_val, best_lambda_, split_version = utils.visualize_cv(os.path.join(s3d.cv_path, 'performance.csv'),
                                                                             validation_metric=metric,
                                                                             split_version=num_fold)

    print('for split version {}...\nbest number of features: {}\nbest lambda: {}'.format(split_version, best_n_f, best_lambda_))
    fn = os.path.join(s3d.viz_path, 'viz-{}-performance-{}.{}'.format(num_fold,metric,ext))
    fp.savefig(fn, bbox_inches='tight')
    logging.info('{} saved!'.format(fn))
    plt.close()
    _s3d_viz_performance_prediction(s3d, num_fold, ext, classes)

def _s3d_viz_performance_prediction(s3d, num_fold, ext, classes):
    # performance prediction
    pink_color = '#FBB4AE'

    features = s3d.get_features(num_fold)
    num_features = len(features)

    y_score = pd.np.loadtxt(os.path.join(s3d.prediction_path, str(num_fold), 'predicted_expectations_MF_{}.csv'.format(int(num_features))))
    y_true = pd.read_csv( os.path.join(s3d.data_path, str(num_fold), 'test.csv')  )['target'].values

    if s3d.classification_flag:
        # thres = 0.5 #s3d.calculate_disc_threshold(s3d.model_paths[num_fold], 4 if num_features > 4 else num_features)
        thres = s3d.calculate_disc_threshold(os.path.join(s3d.model_path,str(num_fold)), num_features)
        y_pred = (y_score >= thres).astype(int)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        utils.plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix, without normalization')
        fn = os.path.join(s3d.viz_path, 'viz-{}-performance-prediction.{}'.format(num_fold, ext))
        plt.savefig(fn, bbox_inches='tight')
        logging.info('{} saved!'.format(fn))
        plt.close()

        # Plot normalized confusion matrix
        plt.figure()
        utils.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')
        fn = os.path.join(s3d.viz_path, 'viz-{}-performance-prediction-norm.{}'.format(num_fold,ext))
        plt.savefig(fn, bbox_inches='tight')
        logging.info('{} saved!'.format(fn))
        plt.close()

    else:
        y_pred = y_score
        plt.scatter(y_true, y_pred, color=pink_color)
        plt.xlabel('True Value')
        plt.ylabel('Predicted')

        fn = os.path.join(s3d.viz_path, 'viz-{}-performance-prediction.{}'.format(num_fold,ext))
        plt.savefig(fn, bbox_inches='tight')
        logging.info('{} saved!'.format(fn))
        plt.close()


###################################################################################################
# MAIN
###################################################################################################
if __name__ == '__main__':
    arghandler = ArgumentsHandler(MUST)

    arghandler.parse_aguments()
    if(arghandler.are_valid_arguments()):

        root = arghandler.get('root') # where results of relational_classification are
        num_jobs = int(arghandler.get('njobs'))  # number of jobs
        csv_fn = arghandler.get('fn')  # csv file:
        # BAHm4_color_prior_bayes_relaxation_n_Snodes_B0.5_global_all_class_0.8.csv
        # BAHm4_color_prior_bayes_relaxation_n_Snodes_B0.5_global_all_reg.csv

        if not os.path.exists(csv_fn):
            print('{} does not exist.'.format(csv_fn))
            sys.exit(0)


        cf = int(csv_fn.split('_')[-2] == 'class')  # classification 1, regression 0

        thres = None
        if cf == 1:
            try:
                thres = float(csv_fn.split('_')[-1].replace('.csv',''))  # threshold classification
            except:
                pass

        dataset = csv_fn.split('_')[0].split('/')[-1] # BAHm4, BAHm20, CalTech
        folder = '{}-S3D'.format(dataset)

        if thres is not None:
            folder = '{}-{}'.format(folder,int(thres*100))

        output = os.path.join(root,folder)
        utils.create_paths(output)

        sampling = csv_fn.split('_S')[-1].split('_B')[0] # nodes
        datatype = csv_fn.split('_B')[-1].split('_')[1]  # global, local
        sub = '_'.join(csv_fn.split('_B')[-1].split('_')[2:]).replace('.csv','').replace('_{}'.format(thres),'')       # all, seeds

        task = PYS3D.CLASSIFICATION if cf else PYS3D.REGRESSION
        metric = 'auc_micro' if task == PYS3D.CLASSIFICATION else 'r2'

        num_folds = int(arghandler.get('num_folds'))  # n. folds
        max_features = int(arghandler.get('max_features')) # n. max features
        output = os.path.join(output, '{}-FOLDS'.format(num_folds), '{}-MAXFEAT'.format(max_features), sampling, datatype, sub)
        utils.create_paths(output)

        opt = arghandler.get('opt')
        ext = 'png' if 'figext' not in arghandler.arguments else arghandler.get('figext')

        classes = ['blue','red'] if datatype == 'local' and task == PYS3D.CLASSIFICATION else [r'$\geq{}$'.format(thres), '<{}'.format(thres)] if datatype == 'global' and task == PYS3D.CLASSIFICATION else None

        initialize_logger(output, '{}_S{}_T{}_U{}'.format(opt, sampling, datatype, sub), os.path.basename(__file__).split('.')[0])
        logging.info('\nParameters passed:\n{}\n'.format('\n'.join(['-{}:{}'.format(k,v) for k,v in arghandler.arguments.items()])))
        logging.info('\nParameters inferred:\n- dataset:{}\n- sampling:{}\n- datatype:{}\n- sub:{}\n- cf:{}\n-threshold:{}\n- task:{}\n- metric:{}\n- classes:{}'.format(dataset, sampling, datatype, sub, cf, thres,task, metric, classes))

        # SPLITING DATA: TRAINING AND TESTING SETS
        if opt == 's3d-folds':
            split_data(csv_fn, dataset, output, cf, num_folds, num_jobs)

        # CROSS-VALIDATION: PARAMETER TUNNING
        elif opt == 's3d-crossval':
            s3d_cross_validation(dataset, output, cf, max_features, num_jobs)

        # SELECTING BEST PARAMETERS
        elif opt == 's3d-evaluation':
            s3d_evaluation(dataset, output, cf, metric, num_jobs, ext)

        # VIZUALIZATIONS
        elif opt == 's3d-viz-model':
            s3d_viz_model(dataset, output, cf, num_jobs, ext)

        elif opt == 's3d-viz-binning':
            s3d_viz_binning(dataset, output, cf, num_jobs, ext)

        elif opt == 's3d-features':
            s3d_viz_network_features(dataset, output, cf, num_jobs, ext)

        elif opt == 's3d-performance':
            s3d_viz_performance(dataset, output, cf, metric, num_jobs, ext, classes)

        # # OTHERS
        # elif opt == 's3d-best-performance':
        #     metric = arghandler.get('metric')
        #     s3d_viz_best_performance(path, sampling, metric, njobs)

# SAMPLING=$1
# NFOLDS=$2
# SUB=$3
# if [[ $# -ne 3 ]] ; then
#     echo 'SAMPLING and NFOLDS and SUB arguments are missing.'
#     exit 1
# fi
#
# if [ "$SUB" == "local" ]; then
#     EVAL='auc_micro'
# else
#     EVAL='mae'
# fi
#
# echo $SAMPLING $NFOLDS $SUB $EVAL;
#
# mkdir /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS
# mkdir /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/$SUB
# mkdir /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/$SUB/datasets
# cp -r /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D/datasets/$SUB/* /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/$SUB/datasets/
# mkdir /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/$SUB/$SAMPLING
# rm -rf /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/$SUB/$SAMPLING/*
# python3.5 feature_selection.py -root /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/ -sampling $SAMPLING -opt s3d-folds -sub $SUB -num_folds $NFOLDS -njobs 10;
# python3.5 feature_selection.py -root /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/ -sampling $SAMPLING -opt s3d-crossval -sub $SUB -njobs 10;
# python3.5 feature_selection.py -root /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/ -sampling $SAMPLING -opt s3d-evaluation -sub $SUB -njobs 10 -metric $EVAL;
# python3.5 feature_selection.py -root /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/ -sampling $SAMPLING -opt s3d-viz-model -sub $SUB -njobs 10;
# python3.5 feature_selection.py -root /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/ -sampling $SAMPLING -opt s3d-viz-binning -sub $SUB -dim 4 -njobs 10;
# python3.5 feature_selection.py -root /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/ -sampling $SAMPLING -opt s3d-features -sub $SUB -njobs 10;
# python3.5 feature_selection.py -root /bigdata/lespin/Network-Unbiased-Inference/results/BAHm4_N2000_B0.5_color_prior_bayes_relaxation_n_S3D_nfolds$NFOLDS/ -sampling $SAMPLING -opt s3d-performance -sub $SUB -njobs 10 -metric $EVAL;
