import matplotlib

from libs.utils.arguments import ArgumentsHandler
from libs.utils.loggerinit import *
import operator
import itertools
from sklearn.preprocessing import normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use('Agg')
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import pandas as pd
import sys
from itertools import product
import seaborn as sns
from pylab import rcParams

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

MUST = ['dataset', 'root', 'label', 'LC', 'RC', 'CI', 'seed', 'eval', 'evalclass', 'seedmethod','sampling','stat','RCattributes','LCattributes','ext']

def plot(df,output,prefix,eval,evalclass,b,stat,sampling,ext):
    # matplotlib.style.use('default')
    rcParams.update({'font.size': 14})

    # sns.set(font_scale=1.4)

    # Setting the positions and width for the bars
    ax = sns.heatmap(df, cmap='Blues', linewidths=0.5, annot=True, vmin=0.0, vmax=1.0, fmt='.2f', annot_kws={"size": 10})

    # Set the y axis label
    ax.set_ylabel('\% Seed nodes')
    ax.set_xlabel('\% Homopily')

    # Set the chart's title
    if ext != 'pdf':
        str = '{} {} {} -B{} by {}'.format(stat.upper(), eval.upper(),evalclass,b,sampling)
    else:
        str = 'B{} by {}'.format(b, sampling)

    str = str.replace('_', '\_')
    ax.set_title(str)

    # Adding the legend and showing the plot
    plt.tight_layout()
    fn = os.path.join(output,'{}-B{}-homophily.{}'.format(prefix,b,ext))
    plt.savefig(fn)
    plt.close()

def read(fn):
    if not os.path.exists(fn):
        return None
    with open(fn,'rb') as f:
        data = pickle.load(f)
    logging.info('{} loaded!'.format(fn))
    return data


if __name__ == '__main__':
    arghandler = ArgumentsHandler(MUST)

    arghandler.parse_aguments()
    if(arghandler.are_valid_arguments()):
        root = arghandler.get('root')
        dataset = arghandler.get('dataset')
        label = arghandler.get('label')
        LC = arghandler.get('LC')
        RC = arghandler.get('RC')
        CI = arghandler.get('CI')
        seed = 'y' if arghandler.get('seed').lower()[0] in ['y','t'] else 'n'
        eval = arghandler.get('eval')
        evalclass = arghandler.get('evalclass')
        seedmethod = arghandler.get('seedmethod')
        sampling = arghandler.get('sampling')
        stat = arghandler.get('stat').lower()
        RCattributes = arghandler.get('RCattributes').lower()[0] == 'y'
        ext = arghandler.get('ext').lower()
        evalclassoriginal = str(evalclass)

        sanitycheck = []

        if stat not in ['mean', 'std']:
            print('stat:{} is not implemented'.format(stat))
            sys.exit(0)

        try:
            evalclass = int(evalclass)
        except:
            pass

        if evalclass not in ['overall', 'red', 'blue', 0, 1]:
            print('evalclass:{} is not implemented'.format(evalclass))
            sys.exit(0)

        if dataset.startswith('BAH') and evalclass == 'overall' and eval == 'roc_auc':
            evalclass = 'blue'

        _H = [round(float(h),1) for h in np.arange(0.0, 1.1, 0.1)]
        _B = [round(float(h),1) for h in np.arange(0.5, 1.0, 0.1)]

        if eval not in ['error', 'roc_auc']:
            logging.error('{} is not supported. Try only error and roc_auc.'.format(eval))
            sys.exit(0)

        if os.path.exists(root):
            prefix = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_byhomophily'.format(dataset, eval, 'overall' if eval == 'roc_auc' else evalclass, label, LC, RC, CI, seed, seedmethod, sampling, stat)
            output = arghandler.get_path(os.path.join(root,'{}-comparison-homophily'.format(prefix)))

            initialize_logger(output, prefix, os.path.basename(__file__).split('.')[0])

            folders = [x for x in os.listdir(root)
                       if (x.split('/')[-1].lower().startswith('{}-'.format(dataset.lower())) or x.split('/')[-1].lower().startswith('{}_'.format(dataset.lower())))
                       and '_{}_T{}'.format(label, seedmethod).lower() in x.split('/')[-1].lower()
                       and '_LC{}_'.format(LC) in x.split('/')[-1]
                       and '_RC{}_'.format(RC) in x.split('/')[-1]
                       and '_CI{}_'.format(CI) in x.split('/')[-1]
                       and '_SEED{}_'.format(seed) in x.split('/')[-1]
                       and '_SAMPLING{}'.format(sampling) in x.split('/')[-1]]

            logging.info('{} total folders for {} ({})'.format(len(folders),dataset,label))

            if RCattributes:
                folders = [f for f in folders if '_RCwithAttributes' in f]
                logging.info('{} folders for RCattributes'.format(len(folders)))
            else:
                folders = [f for f in folders if '_RCwithAttributes' not in f]
                logging.info('{} folders with no RCattributes'.format(len(folders)))

            columns = ['pseeds', 'H', 'value']
            for b in _B:
                df = pd.DataFrame(columns=columns)

                _folders = [f for f in folders if '-B{}-H'.format(b) in f]

                logging.info('{} folders for B{} H'.format(len(_folders), b))

                for folder in _folders:
                    path = os.path.join(root,folder)

                    T = path.split('_T')[1].split('_')[0]
                    logging.info('T: {}'.format(T))

                    if T.lower() != seedmethod.lower():
                        continue

                    # B0.5-H0.9-h

                    H = path.split('-H')[1].split('-Pk')[0].split('-PK')[0]
                    logging.info('H: {}'.format(H))

                    H = float(H)
                    logging.info('H (float): {}'.format(H))

                    files = [fn for fn in os.listdir(path) if os.path.isfile(os.path.join(path, fn)) and fn.endswith('_evaluation.pickle')]
                    logging.info('{} *_evaluation.pickle files in {}'.format(len(files), path))

                    for fn in files:
                        pseeds = int(fn.split('_')[0].replace('P', ''))
                        fn = os.path.join(path,fn)
                        data = read(fn)

                        for obj in data:

                            if evalclass not in obj[eval]:
                                logging.warning('{} does not exist | {} only'.format(evalclass,obj[eval].keys()))
                            else:
                                df = df.append(pd.DataFrame([[pseeds, H, obj[eval][evalclass]]], columns=columns), ignore_index=True)

                                if H == 0.1 and b == _B[2] and pseeds in [1,10]:
                                    sanitycheck.append(obj[eval][evalclass])

                if df.isnull().values.all():
                    print('B:{} | all Bs nan'.format(b))
                else:

                    logging.info('Raw Data:\n{}\n'.format(df.head(5)))

                    if stat == 'mean':
                        df = pd.pivot_table(df, values=columns[2], index=[columns[0]], columns=[columns[1]], aggfunc=np.mean)
                    elif stat == 'std':
                        df = pd.pivot_table(df, values=columns[2], index=[columns[0]], columns=[columns[1]], aggfunc=np.std)

                    logging.info('Pivoted:\n{}\n'.format(df.head(5)))

                    df.sort_index(inplace=True)

                    logging.info('index: {}'.format(df.index))
                    logging.info('columns: {}'.format(df.columns))

                    if dataset.startswith('BAH') and evalclassoriginal == 'overall' and eval == 'roc_auc':
                        evalclass = 'overall'

                    df.fillna(-0.01, inplace=True)
                    plot(df,output,prefix,eval,evalclass,b,stat,sampling,ext)

        else:
            logging.error('{} does NOT exists.'.format(root))

        # logging.info('sanity check for H=0.1 and B={} and pseeds [1 or 10]'.format(_B[2]))
        # logging.info(sanitycheck)
        # logging.info(np.mean(sanitycheck))
        # logging.info(np.std(sanitycheck))
