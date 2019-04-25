import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import pickle

from joblib import Parallel, delayed

from libs.learner import Learner
from libs.utils.arguments import ArgumentsHandler
from libs.utils.loggerinit import *
from libs.utils.netscores import get_homophily
from libs.utils.netscores import get_balance
import sys
from collections import Counter

MUST = ['data','pseeds','sampling','LC','RC','CI','run','output','label','seed','RCattributes','LCattributes']

def does_not_exist(output,prefix):
    fn = '{}/{}_evaluation.pickle'.format(output, prefix)
    return not os.path.exists(fn)

def save(data, output, prefix):
    fn = '{}/{}_evaluation.pickle'.format(output,prefix)
    print(fn)
    with open(fn,'wb') as f:
        pickle.dump(data,f)
    logging.info('{} saved!'.format(fn))

def iterate(iteration, datafn, knownfn, pseeds, LC, RC, CI, label, RCattributes, LCattributes, test, ignore, seed, sampling, nodeattributefn, output=None):
    # BAHm4-N2000-B0.5-H0.8-PK8.0-h0.8-k8.0-7.9
    H = float(datafn.split('/')[-1].split('-H')[-1].split('-PK')[0])
    B = float(datafn.split('/')[-1].split('-B')[-1].split('-H')[0])

    logging.info('')
    logging.info('')
    logging.info('==================== ITERATION {} =========='.format(iteration))
    logging.info('Arguments:\niteration:{}\ndatafn:{}\nknownfn:{}\npseeds:{}\nLC:{}\nRC:{}\nCI:{}\nlabel:{}\ntest:{}\nignore:{}\nseed:{}\nsampling:{}\nRCattributes:{}\nLCattributes:{}\noutput:{}\nH:{}'.format(iteration,datafn,knownfn,pseeds,LC,RC,CI,label,test,ignore,seed,sampling,RCattributes,LCattributes,output,H))
    ln = Learner(datafn, knownfn, pseeds, sampling, LC, RC, CI, label, RCattributes, LCattributes, test, ignore, iteration if seed else None)
    ln.set_nodes_attributes_fn(nodeattributefn)
    ln.initialize()
    ln.learn()
    ln.classify()

    try:
        b0, b1 = get_balance(ln.Gseeds, label)
    except Exception as e:
        logging.error('error while: B0, B1 = get_balance')
        logging.error(e)
        b0, b1 = 0,0

    try:
        h00, h11, h01, h10, h = get_homophily(ln.Gseeds, label)
    except Exception as e:
        logging.error('error while: h00, h11, h01, h10, h = get_homophily')
        logging.error(e)
        h00, h11, h01, h10, h = 0, 0, 0, 0, 0

    labels, nodes, accuracy, f1, mae, rmse, cnf_matrix, precision_recall, roc_auc, error, y_true, y_pred, y_prob = ln.evaluation()
    logging.info('- {} ({:.2f}%) seed nodes out of {} total nodes'.format(ln.Gseeds.number_of_nodes(), ln.Gseeds.number_of_nodes() * 100 / ln.G.number_of_nodes(), ln.G.number_of_nodes()))
    logging.info('- mae: {}'.format(mae['overall']))
    logging.info('- rmse: {}'.format(rmse['overall']))
    logging.info('- accuracy: {}'.format(accuracy))
    logging.info('- f1: {}'.format(f1))
    logging.info('- error: \n{}'.format(error))
    logging.info('- rocauc: \n{}'.format(roc_auc))
    logging.info('- confusion: \n{}'.format(cnf_matrix))
    logging.info('===MODEL===')
    logging.info('classprior: \n{}'.format(ln.relational.classprior))
    logging.info('cpa: \n{}'.format(ln.relational.cpa))
    logging.info('cpn: \n{}'.format(ln.relational.cpn))
    logging.info('===SAMPLE===')
    logging.info('b0:{}\tb1:{}'.format(b0,b1))
    logging.info('h00:{}\th11:{}'.format(h00, h11))
    logging.info('h01:{}\th10:{}'.format(h01, h10))
    logging.info('===CLASSIFICATION SAMPLE (top 10)===')

    unlabelled = [n for n in ln.G if n not in ln.Gseeds]
    counts = Counter([ln.G.node[n][ln.label] for n in unlabelled])
    logging.info('ground-truth unlabelled: {}'.format(counts))

    #count proportions on nonseeds
    # if output is not None and ln.ns is not None and sampling == 'lightweight':
    #     ln.ns.set_plotfn(os.path.join(output,'plot_HypRW_{}.pdf'.format(int(float(pseeds)*100))))
    #     ln.ns.plot_apprxn(ln.ns.data_to_plot[0], ln.ns.data_to_plot[1], ln.ns.data_to_plot[2], ln.ns.data_to_plot[3], ln.ns.data_to_plot[4], ln.ns.data_to_plot[5], ln.ns.data_to_plot[6])


    obj = None
    try:
        obj = { 'datafn':datafn,
                'label':label,
                'labels': labels,
                'unlabeled':nodes,
                'N':ln.G.number_of_nodes(),
                'E': ln.G.number_of_edges(),
                'H':H,
                'B': B,
                'sampling':sampling,
                'seeds': ln.Gseeds.nodes(),
                'pseeds':pseeds,
                'seedsN': ln.Gseeds.number_of_nodes(),
                'seedsE': ln.Gseeds.number_of_edges(),
                'seedsH00': h00,
                'seedsH11': h11,
                'seedsH01': h01,
                'seedsH10': h10,
                'seedsB0': b0,
                'seedsB1': b1,
                'class_prior':ln.relational.classprior,
                'class_prior_attributes':ln.relational.cpa,
                'cond_probabilities': ln.relational.cpn,
                'accuracy': accuracy,
                'f1': f1, 'mae': mae, 'rmse': rmse, 'cnf_matrix': cnf_matrix,
                'precision_recall': precision_recall, 'roc_auc': roc_auc, 'error':error,
                'y_true':y_true, 'y_pred':y_pred, 'y_prob':y_prob }

    except Exception as ex:
        print(ex)
        logging.warning('Error return results: {}'.format(ex))

    return obj

if __name__ == '__main__':
    arghandler = ArgumentsHandler(MUST)

    arghandler.parse_aguments()
    if(arghandler.are_valid_arguments()):

        output = arghandler.get_path()
        prefix = arghandler.get_file()
        initialize_logger(output,prefix,os.path.basename(__file__).split('.')[0])
        logging.info('\nParameters passed:\n{}\n'.format(arghandler.arguments))

        datafn, pseeds, LC, RC, CI, label, runs, sampling = arghandler.get('data'), float(arghandler.get('pseeds')), arghandler.get('LC'), arghandler.get('RC'), arghandler.get('CI'), arghandler.get('label'), range(1, int(arghandler.get('run')) + 1), arghandler.get('sampling')
        test = arghandler.get('test').lower()[0] in ['y','t'] if 'test' in arghandler.arguments.keys() else False
        ignore = arghandler.get('ignore') if 'ignore' in arghandler.arguments.keys() else None
        knownfn = arghandler.get('known') if 'known' in arghandler.arguments.keys() and arghandler.get('known') is not None and os.path.exists(arghandler.get('known')) and (arghandler.get('known').endswith('.pickle') or arghandler.get('known').endswith('.gpickle') or arghandler.get('known').endswith('.txt')) else None
        seed = arghandler.get('seed').lower()[0] == 'y'
        RCattributes = arghandler.get('RCattributes').lower()[0] == 'y'
        LCattributes = arghandler.get('LCattributes').lower()[0] == 'y'
        nodeattributefn = arghandler.get('nafn') if 'nafn' in arghandler.arguments.keys() else None

        try: ignore = int(ignore)
        except: pass

        if does_not_exist(output,prefix):
            results = Parallel(n_jobs=len(runs))(delayed(iterate)(iteration, datafn, knownfn, pseeds, LC, RC, CI, label, RCattributes, LCattributes, test, ignore, seed, sampling, nodeattributefn, output) for iteration in runs)
            save(results, output, prefix)
        else:
            logging.info('Nothing to do, this already exists. {} {}. Bye'.format(output,prefix))



