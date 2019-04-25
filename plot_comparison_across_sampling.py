import matplotlib

from libs.utils.arguments import ArgumentsHandler
from libs.utils.loggerinit import *
# import operator
# import itertools
# from sklearn.preprocessing import normalize
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from libs.collective.collectivehandler import Collective

matplotlib.use('Agg')
import pickle
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.pyplot import cm
import pandas as pd
import sys
import six
# from matplotlib import colors as mcolors
# from matplotlib.lines import Line2D
from itertools import product
from itertools import cycle
from pylab import rcParams
from sklearn.metrics import f1_score
from matplotlib.lines import Line2D
import networkx as nx
from collections import Counter
import pylab

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

MUST = ['dataset', 'root', 'label', 'LC', 'RC', 'CI', 'seed', 'eval', 'evalclass', 'RCattributes', 'LCattributes', 'ext']

def render_mpl_table(data, col_width=0.6, row_height=0.5, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0.12, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):

    # headwidth = 6
    data = data.applymap("{0:.2f}".format)

    # tmp = data.columns
    # data['Sampling'] = data.index
    # columns = ['Sampling']
    # columns.extend(tmp)
    # data = data[columns]
    # logging.info('\n{}\n'.format(data.head(5)))

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([6, 1])) * np.array([col_width, row_height])
        # size[0] += headwidth + 1
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, rowLabels=data.index, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)

        # if k[1] == -1:
        #     cell.set_width(headwidth)

        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax



def plot(dfmean, dfstd, output, prefix, eval, H, B, stat, evalclass, ext):
    # matplotlib.style.use('default')
    rcParams.update({'font.size': 18})
    cmap = plt.get_cmap('tab10')
    # Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r,
    # CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys,
    # Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r,
    # Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r,
    # PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu,
    # RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r,
    # Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r,
    # Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r,
    # YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r,
    # bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r,
    # gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow,
    # gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray,
    # gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean,
    # ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral,
    # spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r,
    # terrain, terrain_r, viridis, viridis_r, winter, winter_r

    # colors = cycle([cmap(i) for i in np.linspace(0, 1, len(dfmean.columns))])
    linestyles = cycle(['-', '--', ':'])
    markers = cycle(Line2D.filled_markers)

    columns = sorted(list(dfmean.columns))
    logging.info(columns)

    for c in columns:
        x = dfmean[c].index
        y = dfmean[c].values
        yerr = dfstd[c].values
        ax = errorfill(x, y, yerr, c,
                  color = 'blue' if 'blue' in c.lower() else 'red' if 'red' in c.lower() else 'black' if 'random' in c.lower() else None,
                  linestyle = '--' if 'random' in c.lower() or 'baseline' in c.lower() else next(linestyles),
                  marker=None if 'random' in c.lower() or 'baseline' in c.lower() else next(markers),
                  alpha_fill=0.1,
                  alpha_line=0.3 if 'random' in c.lower() or 'baseline' in c.lower() else 1.0,
                  ax=None) # ,
                  # solid_joinstyle="miter")

    # random
    ax = plt.gca()
    lines = ax.lines
    ax.axhline(y=0.5, color='k', linestyle='-', linewidth='0.1')

    # Set the y axis label
    ax.set_xlabel('\% Seed Nodes')
    str = '{} {}'.format(stat.title(), eval.title())
    str = str.replace('_','\_')
    ax.set_ylabel(str)
    ax.set_ylim(0.40, 1.10)

    # Legend
    # plt.legend(numpoints=1, # Set the number of markers in label
    #            loc="best")  # Set label location
    if ext != 'pdf':
        lgd = ax.legend(numpoints=1,loc='center right', bbox_to_anchor=(1.4, 0.5), prop = {'size': 10})

    # Set the chart's title
    if ext != 'pdf':
        str = '{} {} ({}) H{}-B{}'.format(stat.upper(), eval.upper(), evalclass, H, B)
    else:
        str = 'H{}-B{}'.format(H, B)

    str = str.replace('_', '\_')
    ax.set_title(str)

    # Saving
    # plt.tight_layout()
    fn = os.path.join(output, '{}-H{}-B{}.{}'.format(prefix, H, B, ext))
    if ext != 'pdf':
        plt.savefig(fn, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    plt.close()


    # plot table
    ax = render_mpl_table(dfmean.transpose(), header_columns=0)
    str = '{} {} ({}) H{}-B{}'.format(stat.upper(), eval.upper(), evalclass, H, B)
    str = str.replace('_', '\_')
    ax.set_title(str)
    fn = os.path.join(output, 'table_{}-H{}-B{}.{}'.format(prefix, H, B, ext))
    plt.savefig(fn)
    plt.close()


    # legend
    figlegend = pylab.figure(figsize=(12.7, 1))
    figlegend.legend(lines, columns, 'center', ncol=5)
    fn = os.path.join(output, 'legend.{}'.format(ext))
    figlegend.savefig(fn)
    plt.close()

def errorplot(x,y,yerr,fmt,color,label):
    line, caps, bars = plt.errorbar(
        x,  # X
        y,  # Y
        yerr=yerr,  # Y-errors
        fmt=fmt,  # format line like for plot()
        linewidth=1.5,  # width of plot line
        elinewidth=0.5,  # width of error bar line
        ecolor=color,  # color of error bar
        color=color,
        capsize=3,  # cap length for error bar
        capthick=0.5  # cap thickness for error bar
    )
    plt.setp(line, label=label)  # give label to returned line

def errorfill(x, y, yerr, label, linestyle=None, marker=None, color=None, alpha_fill=0.2, alpha_line=1.0, ax=None, solid_joinstyle=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        # color = ax._get_lines.color_cycle.next()
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label, alpha=alpha_line, linestyle=linestyle, marker=marker, solid_joinstyle=solid_joinstyle)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

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
        stat = 'mean'
        H = None if 'H' not in arghandler.arguments else float(arghandler.get('H'))
        B = None if 'B' not in arghandler.arguments else float(arghandler.get('B'))
        M = None #majority label
        RCattributes = arghandler.get('RCattributes').lower()[0] == 'y'
        LCattributes = arghandler.get('LCattributes').lower()[0] == 'y'
        ext = arghandler.get('ext')

        sanitycheck = []
        baselines = set()

        try:
            evalclass = int(evalclass)
        except:
            pass

        if dataset.startswith('BAH') and (H is None or B is None):
            print('H and B MUST be specified!!!')
            sys.exit(0)

        if evalclass not in ['overall','red','blue', 0,1,2]:
            print('evalclass:{} is not implemented'.format(evalclass))
            sys.exit(0)

        if eval not in ['error', 'f1', 'roc_auc']:
            logging.error('{} is not supported. Try only error and roc_auc'.format(eval))
            sys.exit(0)

        if eval == 'roc_auc':
            evalclass = 'overall'

        if os.path.exists(root):
            prefix = '{}_{}_{}_{}_{}{}_{}{}_{}_{}_{}_sampling'.format(dataset, eval, evalclass, label, LC, '_LCattributes' if LCattributes else '', RC, '_RCattributes' if RCattributes else '', CI, seed, stat)
            output = arghandler.get_path(os.path.join(root, '{}-comparison'.format(prefix)))

            initialize_logger(output, prefix, os.path.basename(__file__).split('.')[0])

            folders = [x for x in os.listdir(root)
                       if (x.split('/')[-1].lower().startswith('{}-'.format(dataset.lower())) or x.split('/')[-1].lower().startswith('{}_'.format(dataset.lower())))
                       and '_{}_T'.format(label).lower() in x.split('/')[-1].lower()
                       and '_LC{}_'.format(LC) in x.split('/')[-1]
                       and '_RC{}_'.format(RC) in x.split('/')[-1]
                       and '_CI{}_'.format(CI) in x.split('/')[-1]
                       and '_SEED{}_'.format(seed) in x.split('/')[-1]
                       and os.path.isdir(os.path.join(root, x))]

            logging.info('{} folders for {} ({})'.format(len(folders),dataset,label))

            columns = ['pseeds', 'sampling', 'value']
            df = pd.DataFrame(columns=columns)

            if RCattributes:
                folders = [f for f in folders if '_RCwithAttributes' in f]
                logging.info('{} folders for RCattributes'.format(len(folders)))
            else:
                folders = [f for f in folders if '_RCwithAttributes' not in f]
                logging.info('{} folders with no RCattributes'.format(len(folders)))


            if LCattributes:
                folders = [f for f in folders if '_LCwithAttributes' in f]
                logging.info('{} folders for LCattributes'.format(len(folders)))
            else:
                folders = [f for f in folders if '_LCwithAttributes' not in f]
                logging.info('{} folders with no LCattributes'.format(len(folders)))



            if H is not None and B is not None:
                _folders = [f for f in folders if '-H{}-'.format(H) in f and '-B{}-'.format(B) in f]
                logging.info('{} folders for H{} B{}'.format(len(_folders), H, B))
            else:
                _folders = folders
                logging.info('{} folders for {}'.format(len(_folders),dataset))


            for folder in _folders:
                path = os.path.join(root, folder)

                T = path.split('_SAMPLING')[1].split('_')[0]
                logging.info('T: {}'.format(T))
                seedmethod = T

                random = ['nodes','nedges','snowball','lightweight']
                topological = ['degreeasc','degreedesc','pagerankasc','pagerankdesc','percolationasc','percolationdesc']
                toexclude = ['edges','lightweight']
                logging.debug('excluding these sampling methods: {}'.format(toexclude))

                if seedmethod.lower() in toexclude:
                    continue

                files = [fn for fn in os.listdir(path) if os.path.isfile(os.path.join(path, fn)) and fn.endswith('_evaluation.pickle')]
                logging.info('{} *_evaluation.pickle files in {}'.format(len(files), path))

                if eval == 'roc_auc' and dataset.startswith('BAH'):
                    evalclass = 'blue'
                elif eval == 'roc_auc' and dataset.startswith('Caltech36'):
                    evalclass = 1
                elif eval == 'roc_auc' and evalclass != 'overall':
                    logging.info('{}: evalclass:{} in {} must be overall'.format(eval,evalclass,dataset))
                    sys.exit(0)


                for fn in files:
                    pseeds = int(fn.split('_')[0].replace('P', ''))

                    if pseeds > 0 and pseeds <= 100:
                        fn = os.path.join(path, fn)
                        data = read(fn)

                        if 'datafn' in data[0]:
                            G=nx.read_gpickle(data[0]['datafn'])

                            try:
                                nodes = [n for n in G.nodes() if G.node[n][label] in [0,'?','0']]
                                G.remove_nodes_from(nodes)
                                nodes = [n for n in G.nodes() if G.degree(n)==0]
                                G.remove_nodes_from(nodes)
                            except:
                                pass


                            M = Counter([G.node[n][label] for n in G.nodes()])
                            N = sum(M.values())
                            logging.info('balance: {}'.format(M))
                            M = M.most_common(1)[0]
                            logging.info('most common balance (B): {}'.format(M))
                            logging.info('N: {}'.format(float(N)))

                            if B is None:
                                B = round(M[1] / float(N),1)
                                logging.info('{} / {} = {}'.format(M[1],float(N),B))

                            M = str(M[0])
                            logging.info('B:{} from network'.format(B))

                            same = sum([1 for edge in G.edges() if G.node[edge[0]][label] == G.node[edge[1]][label]])
                            if H is None:
                                H = round( same / float(G.number_of_edges()),1)

                            logging.info('same {} edges: {} | {} total edges | H:{}'.format(label,same,G.number_of_edges(),H))

                        if M is None:
                            M = 'red'
                            labels = data[0]['labels']
                            if M not in labels:
                                logging.warning('M:{} does not exist'.format(M))
                                sys.exit(0)

                        logging.info('Majority label: {}'.format(M))

                        for obj in data:
                            if evalclass not in obj[eval]:
                                logging.warning('{} does not exist | {} only'.format(evalclass, obj[eval].keys()))
                            else:
                                df = df.append(pd.DataFrame([[pseeds, seedmethod, obj[eval][evalclass]]], columns=columns), ignore_index=True)

                                # sanity check
                                if H is not None and B is not None:
                                    if H == 0.9 and B == 0.9 and pseeds in [1, 10]:
                                        sanitycheck.append(obj[eval][evalclass])

                    # baselines
                    if B is not None and evalclass == 'overall':
                        if pseeds not in baselines and B > 0.5:
                            labels = data[0]['labels']
                            N = len(data[0]['unlabeled'])+len(data[0]['seeds'])
                            N = int(round(N * pseeds / 100.))
                            logging.info('N: {}'.format(N))
                            majority = [1] * int(B*N)
                            majority.extend([0] * (N-int(B*N)))
                            majority = np.asarray(majority)
                            np.random.shuffle(majority)
                            logging.info('majority sum {}, B:{}'.format(majority.sum(),B))
                            #np.random.binomial(1, B, size=N)

                            randomerror = []
                            for i in range(len(data)):
                                random = np.random.binomial(1, 0.5, size=N)

                                if eval == 'error':
                                    score = 1. - float(random.dot(majority))/N
                                elif eval == 'f1':
                                    score = f1_score(majority, random, labels=labels, average='weighted')

                                logging.info('score {}: {}'.format(eval,score))

                                df = df.append(pd.DataFrame([[pseeds, 'zRandom', score]], columns=columns), ignore_index=True)

                                if eval == 'error':
                                    #mejority
                                    df = df.append(pd.DataFrame([[pseeds, 'zBaseline{}'.format(M.title()), 1.0 - B]], columns=columns), ignore_index=True)

                                    #minotirty
                                    m = [str(l) for l in labels if l != M][0]
                                    logging.info('minority label: {}'.format(m))
                                    df = df.append(pd.DataFrame([[pseeds, 'zBaseline{}'.format(m.title()), B]], columns=columns), ignore_index=True)

                                baselines.add(pseeds)


            logging.info(df.head(5))

            ###
            if df.isnull().values.all():
                print('H{} B{}| all Hs nan'.format(H,B))
            else:
                logging.info('Raw Data:\n{}\n'.format(df.head(6)))

                dfmean = pd.pivot_table(df, values=columns[2], index=[columns[0]], columns=[columns[1]], aggfunc=np.mean, dropna=True)
                dfstd = pd.pivot_table(df, values=columns[2], index=[columns[0]], columns=[columns[1]], aggfunc=np.std, dropna=True)

                logging.info('grouped by:\n{}\n'.format(dfmean.head(6)))

                dfmean.sort_index(inplace=True)
                dfstd.sort_index(inplace=True)

                logging.info('index: {}'.format(dfmean.index))
                logging.info('columns: {}'.format(dfmean.columns))

                # df.fillna(-0.1, inplace=True)
                logging.info('Raw Data (aggregated):\n{}\n'.format(dfmean.head(5)))

                if eval == 'roc_auc':
                    evalclass = 'overall'
                    
                plot(dfmean, dfstd, output, prefix, eval, H, B, stat, evalclass, ext)

        else:
            logging.error('{} does NOT exists.'.format(root))

        try:
            if len(sanitycheck) > 0:
                logging.info('sanity check for H=0.9 and B=0.9')
                logging.info(sanitycheck)
                logging.info(np.mean(sanitycheck))
                logging.info(np.std(sanitycheck))
        except:
            pass


#
# def plot2(df, output, prefix, eval, H, B, stat, evalclass):
#     styles = plt.style.available
#     style = 'seaborn-notebook'
#     style = style if style in styles else np.random.choice(styles)
#     plt.style.use(style)
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#     markers = []
#     for m in Line2D.markers:
#         try:
#             if len(m) == 1 and m != ' ':
#                 markers.append(m)
#         except TypeError:
#             pass
#
#     styles = markers + [
#         r'$\lambda$',
#         r'$\bowtie$',
#         r'$\circlearrowleft$',
#         r'$\clubsuit$',
#         r'$\checkmark$']
#
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     i = 0
#     print('')
#     with pd.plotting.plot_params.use('x_compat', True):
#         for c in df.columns:
#             tmp = df[c].dropna(axis=0, inplace=False).sort_index(axis=0, ascending=True, inplace=False)
#             tmp.plot(ax=ax, color=colors[i], marker=styles[i], label=c, legend=True)
#             print(c)
#             print(tmp)
#             print('')
#             i+=1
#
#     # Set the y axis label
#     ax.set_xlabel('# Nodes')
#     ax.set_ylabel('{} {}'.format(stat.title(), eval.title()))
#     ax.set_ylim(0, 1.0)
#
#     # Set the chart's title
#     ax.set_title('{} {} ({}) H{}-B{}'.format(stat.upper(), eval.upper(), evalclass, H, B))
#
#     # Adding the legend and showing the plot
#     plt.tight_layout()
#     fn = os.path.join(output, '{}-H{}-B{}.png'.format(prefix, H, B))
#     plt.savefig(fn)
#     plt.close()
#
# def settings_plots():
#     styles = plt.style.available
#     style = 'seaborn-notebook'
#     style = style if style in styles else np.random.choice(styles)
#     plt.style.use(style)
#     # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#     styles = []
#     for m in Line2D.markers:
#         try:
#             if len(m) == 1 and m != ' ':
#                 styles.append('{}-'.format(m))
#         except TypeError:
#             pass
#
#     styles = sorted(styles)
#     print(styles)
#     return styles
#
# def _plot(dfmean, dfstd, output, prefix, eval, H, B, stat, evalclass):
#
#     matplotlib.style.use('default')
#
#     styles = settings_plots()
#     linestyle = ['-','--',':']
#     markers = ['o','v','*','D']
#     styles = ['{}{}'.format(lm[0],lm[1]) for lm in product(markers,linestyle)]
#
#     # curves
#     # fig, ax = plt.subplots()
#     #
#     # stdev = data.std()
#     #
#     # ax.errorbar(x, means, yerr=stdev, color='red', ls='--', marker='o', capsize=5, capthick=1, ecolor='black')
#     #
#     # ax.set_xlim(xlims)
#     # ax.set_ylim(ylims)
#
#     # cmap = plt.get_cmap('tab20')
#     # colors = [cmap(i) for i in np.linspace(0, 1, len(dfmean.columns))]
#
#     # fig, ax = plt.subplots()
#     # for c in list(dfmean.columns):
#     #     print(c)
#     #     ax.errorbar(dfmean.index,dfmean[c],yerr=dfstd[c], ls=next(linestyle), marker=next(markers), capsize=5, capthick=1, ecolor='black', label=c)
#     # ax.set_xlim(10,90)
#     # ax.set_ylim(0,1)
#
#     # ax = dfmean.plot(legend=True, style=styles, yerr=dfstd)
#     ax = dfmean.plot(legend=True, style=styles)
#
#     # adding baseline
#     # random
#     ax.axhline(y=0.5, color='k', linestyle='-', label='random', linewidth='0.1')
#     # majority
#
#     # Set the y axis label
#     ax.set_xlabel('% Seed Nodes')
#     ax.set_ylabel('{} {}'.format(stat.title(), eval.title()))
#     ax.set_ylim(0, 1.0)
#
#     # Set the chart's title
#     ax.set_title('{} {} ({}) H{}-B{}'.format(stat.upper(), eval.upper(), evalclass, H, B))
#
#     # Adding the legend and showing the plot
#     plt.tight_layout()
#     fn = os.path.join(output, '{}-H{}-B{}.png'.format(prefix, H, B))
#     plt.savefig(fn)
#     plt.close()
#
#     # plot table
#     ax = render_mpl_table(dfmean.transpose(), header_columns=0, col_width=2.0)
#     ax.set_title('{} {} ({}) H{}-B{}'.format(stat.upper(), eval.upper(), evalclass, H, B))
#     plt.tight_layout()
#     fn = os.path.join(output, 'table_{}-H{}-B{}.png'.format(prefix, H, B))
#     plt.savefig(fn)
#     plt.close()
#
# def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
#     ax = ax if ax is not None else plt.gca()
#     if color is None:
#         color = ax._get_lines.color_cycle.next()
#     if np.isscalar(yerr) or len(yerr) == len(y):
#         ymin = y - yerr
#         ymax = y + yerr
#     elif len(yerr) == 2:
#         ymin, ymax = yerr
#     ax.plot(x, y, color=color)
#     ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
