import os
import math
import sympy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from palettable.colorbrewer.diverging import RdBu_11

from utils import empirical

############################################################################################################
# Constants
############################################################################################################

# Available metrics

METRIC = {'rocauc':'ROCAUC', 'ROCAUC': 'ROCAUC', 'pseeds':'pseeds \%',
          
          'bias':r"${bias}=\frac{TPR_{min}}{TPR_{min}+TPR_{maj}}$",
          
          # priors
          'p1':r'$P_{min}$',
          'p0':r'$P_{maj}$',
    
          # Estimation Error: estimation - observed
          'EEp1': r'$P_{min} - \theta_{min}$',
          'EEp0': r'$P_{maj} - \theta_{maj}$',
          'EEcp11': r'$P_{min|min} - \theta_{min|min}$',
          'EEcp01': r'$P_{maj|min} - \theta_{maj|min}$',
          'EEcp00': r'$P_{maj|maj} - \theta_{maj|maj}$',
          'EEcp10': r'$P_{min|maj} - \theta_{min|maj}$',

          # Squared Error: squared estimation error (estimation - observed)^2
          'SEp1': r'$(P_{min} - \theta_{min})^2$',
          'SEp1-s': r'$SE_{min}$',
          
          'SEp0': r'$(P_{maj} - \theta_{maj})^2$',
          
          'SEcp11': r'$(P_{min|min} - \theta_{min|min})^2$',
          'SEcp11-s': r'$SE_{min|min}$',
          
          'SEcp00': r'$(P_{maj|maj} - \theta_{maj|maj})^2$',
          'SEcp00-s': r'$SE_{maj|maj}$',
          
          # Comparing the estimation errors
          'SEcpDiff': r"$(\theta_{maj|maj} - P_{maj|maj})^2 - (\theta_{min|min} - P_{min|min})^2$",
          
          'SEcpSum': r"$(\theta_{maj|maj} - P_{maj|maj})^2 + (\theta_{min|min} - P_{min|min})^2$",
          'SEcpSum-s': r"$SE_{maj|maj} + SE_{min|min} $",
          
          'SE': r"$SE_{min} + SE_{maj|maj} + SE_{min|min} $",
          'SE-s': r"$\sum SE$"
          }


############################################################################################################
# Latex compatible
############################################################################################################

def latex_compatible_text(txt):
    return sympy.latex(sympy.sympify(txt)).replace("_", "\_")

def latex_compatible_dataframe(df, latex=True):
    tmp = df.copy()
    if latex:
        cols = {c:c if c=="N" or c.startswith("MSE") else sympy.latex(sympy.sympify(c)).replace("_","\_") for c in tmp.columns}
        if 'sampling' in tmp.columns:
            tmp.sampling = tmp.apply(lambda row: row.sampling.replace('_', '\_'), axis=1)
    else:
        cols = {c:c for c in tmp.columns}
    cols['rocauc'] = cols['rocauc'].upper()
    tmp.rename(columns=cols, inplace=True)
    return tmp, cols

def unlatexfyme(text):
    return text.replace("_", "").replace("\\", "").replace('{', '').replace('}', '').replace('$','').strip()

############################################################################################################
# Latex tables
############################################################################################################



def plot_empirical_degree_distributions(root, bestk=True, fn=None):
    files = [os.path.join(root,fg) for fg in os.listdir(root) 
             if not fg.startswith('BAH') and 
             not fg.startswith('USF') and
             fg != 'synthetic_evalues.csv']
    
    sorted_files = []
    for d in ['escort','usf','swarth','caltech','wiki','github']:
        tmp = [fn for fn in files if d in fn.lower()]
        if len(tmp) > 0 :
            sorted_files.append(tmp[0])
            
    ndatasets = len(sorted_files)
    
    plt.close()
    fig,axes = plt.subplots(1, ndatasets, figsize=(3*ndatasets, 2), sharex=False, sharey=False)
    #yvalues = [0.0001, 0.0004, 0.0001, 0.0006, 0.002, 0.0000003] # with USF
    yvalues = [0.0001, 0.0001, 0.0006, 0.002, 0.0000003]

    for c, fg in enumerate(sorted_files):
        dataset = fg.split('/')[-1].replace('.gpickle','')
        pl = empirical.get_power_law(fg, bestk)
        pl.plot_pdf(ax=axes[c], linewidth=3, linestyle='-', label='data')
        pl.power_law.plot_pdf(ax=axes[c], linestyle='--', label='power-law')
        s = '$k_{min}=$' + '{:.0f}; '.format(pl.power_law.xmin) + '$\gamma =$' + '{:.2f}'.format(pl.power_law.alpha)
        axes[c].text(s=s, x=pl.power_law.xmin, y=yvalues[c], va='top', ha='left')
        axes[c].set_title(dataset)

    plt.legend(loc="upper right")

    ### space between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.25)

    ### Save fig
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    ###
    plt.show()
    plt.close()



############################################################################################################
# Plots RQ1: Structure vs Performance (ROCAUC)
############################################################################################################

def plot_rocauc_curve(fpr, tpr, rocauc, fn=None):
    plt.close()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % rocauc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_rocauc_vs_homophily_per_B_m_pseeds(df, columns, example=False, fn=None):
    plt.close()
    tmp = df.copy()

    evaluation = columns['rocauc']
    xaxis = columns['H']
    hue = columns['pseeds']
    col = columns['B']
    col_order = sorted(tmp[col].unique(), reverse=True)
    tmp.loc[:,hue] = tmp.loc[:,hue].apply(lambda r: int(r*100))
    tmp.loc[:,'density'] = tmp.apply(lambda r: round(r['density'],3 if r['m'] == 4 else 2), axis=1)
    tmp.rename(columns={'density':'d'}, inplace=True)
    
    
    fg = sns.catplot(data=tmp,
                     x=xaxis,
                     y=evaluation,
                     col=col,
                     row='d', #'m'
                     hue=hue,
                     ci='sd',
                     kind='point',
                     margin_titles=True,
                     height=1.7,
                     aspect=0.9,
                     palette=RdBu_11.mpl_colors,
                     legend=True,
                     col_order=col_order
                     )

    for aa in np.ndenumerate(fg.axes):
        coord = aa[0]

        if coord != (1, 1):
            aa[1].set_xlabel("")
        #else:
        #    aa[1].set_xlabel("$h$")
        ### @todo: uncomment this h and fm but first change througout all the paper.
        #if coord[0] == 0:
        #    aa[1].set_title(aa[1].get_title().replace("B",'$f_m$'))
            
        if coord[0] == 1:
            _set_minimal_xticklabels(aa[1])
            # labels = aa[1].get_xticklabels()  # get x labels
            # for i, l in enumerate(labels):
            #     if (i not in [1, 5, 9]): labels[i] = ''  # skip even labels
            # aa[1].set_xticklabels(labels, rotation=0)

        aa[1].axhline(0.5, lw=0.5, c="grey", ls="--")

        # example
        if example:
            if coord == (0, 0):
                aa[1].annotate('', xy=(9.1, 0.61),
                               xytext=(10, 0.45),
                               arrowprops={'arrowstyle': '-|>',
                                           'lw': 2,
                                           'ec': 'k', 'fc': 'k'},
                               va='center')

    if tmp[columns['m']].nunique() % 2 == 0:
        for ax in fg.axes.flatten():
            ax.set_ylabel('')
        fg.axes[1,0].text(s=evaluation, x=-5.5, y=1.15, rotation=90, va='center')

    if hue == 'pseeds':
        new_title = 'pseeds \%'
        fg._legend.set_title(new_title)


    plt.ylim((0.5-0.20, 1.0+0.20))
    
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
        print("{} saved!".format(fn))
        
    plt.show()
    plt.close()

def plot_rocauc_vs_pseeds_per_H_B_N_m(df, columns, fn=None):
    y = columns['rocauc']
    row = columns['B']
    row_order = sorted(df[row].unique(), reverse=True)
    col = columns['H']
    hue = columns['network_size']
    toplegend = True
    palette = "Paired"

    #1
    tmp = df.copy()
    
    #2 changing network_size from N,m to N,d
    tmp.loc[:,hue] = tmp.apply(lambda row: 'N{}, d{}'.format(row[columns['N']],
                                                             round(row[columns['density']],3 
                                                                   if row[columns['density']] < 0.01 else 2)), axis=1)
    
    #3
    tmp.sort_values(['m','N'], inplace=True)
    _plot_by_pseeds(tmp, y, row, col, hue, hue_order=None, fn=fn, ylabel=(True, True),
                    legend=True, toplegend=toplegend, yticklabels=True, kind="line",
                    logy=False, palette=palette, row_order=row_order, height=1.2, aspect=1)

def plot_rocauc_vs_pseeds_per_H_B_sampling(df, columns, fn=None):
    y = columns['rocauc']
    row = columns['H']
    col = columns['B']
    hue = columns['sampling']

    hue_order = []
    hue_list = df[hue].unique()
    for ho in ['nodes', 'neighbors', 'nedges', 'degree', 'partial\_crawls', 'partial_crawls']:
        if ho in hue_list:
            hue_order.append(ho)

    toplegend = True
    palette = "tab10"
    #2
    _plot_by_pseeds(df, y, row, col, hue, hue_order=hue_order, fn=fn, ylabel=(True, False),
                    legend=False, toplegend=toplegend, yticklabels=True, kind="line", logy=False, palette=palette)


############################################################################################################
# Plots RQ1: Mixed effects model
############################################################################################################

def plot_fixed_effects(fe_params, fn=None):
    plt.close()
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    fe_params.reset_index().plot(ax=ax, kind='bar', x='index', y='LMM')
    # ax.set_title("Fixed Effects")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(fe_params.index.shape[0]))
    ax.set_xticklabels(fe_params.index.values.tolist(), rotation=0)
    ax.set_ylabel("")  # LMM
    ax.axhline(y=0, color='black', linestyle='--', lw=1)
    ax.get_legend().remove()

    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()

def plot_random_effects(random_effects, group_vars, fn=None):
    plt.close()

    tmp = random_effects.copy()
    tmp2 = pd.DataFrame(tmp.index.astype(str).str.split('_').tolist(), columns=group_vars, index=tmp.index)
    tmp = pd.concat([tmp, tmp2], axis=1, sort=False)

    fg = sns.catplot(data=tmp, x='H', y='LMM', hue='pseeds',
                     height=3.0, aspect=1.0,
                     palette=RdBu_11.mpl_colors,
                     kind='swarm'
                     )

    _set_minimal_xticklabels(fg.ax)
    fg.ax.axhline(0.0, ls="--", c='grey', lw=1.0)
    fg.ax.set_ylabel("")

    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()

def plot_fitted_line(mdf, y_observed, fn=None):
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    y_predict = mdf.fittedvalues
    performance = pd.DataFrame()
    performance["residuals"] = mdf.resid.values
    performance["Observed"] = y_observed
    performance["Predicted"] = y_predict
    ax = sns.regplot(x="Predicted", y="Observed", data=performance, ax=ax, truncate=False)

    mae = np.mean(abs(performance["Observed"] - performance["Predicted"]))
    ax.text(x=0.1, y=0.8, s='MAE: {}'.format(round(mae, 4)))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)

    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()

def plot_prediction(X, Y, Z, fe_params, random_effects, fn=None):
    plt.close()
    fig = plt.figure(figsize=(18, 3))
    ax = fig.add_subplot(1, 1, 1)
    if X is not None and Y is not None:
        ax.plot(Y.flatten(), 'o', color='grey', label='Observed', alpha=.5)

        if Z is not None:
            for iname in fe_params.columns.get_values():
                fitted = np.dot(X, fe_params[iname]) + np.dot(Z, random_effects[iname]).flatten()

                mse = np.mean(np.square(Y.flatten() - fitted))
                rmse = np.sqrt(mse)
                mae = np.mean(abs(Y.flatten() - fitted))

                print("The MSE of " + iname + " is " + str(mse))
                print("The RMSE of " + iname + " is " + str(rmse))
                print("The MAE of " + iname + " is " + str(mae))

                ax.plot(fitted, 'x', color='orange', label=iname, alpha=.8)  # lw=1,

        ax.legend(loc=0)
        ax.set_ylabel("Predictions")
        ax.set_xlabel("Observations")

    if fn is not None:
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()

def plot_model_vs_data(df, fn):
    plt.close()
    tmp = df.copy()
    tmp.loc[:,'pseeds'] = tmp.apply(lambda row: int(row.pseeds*100), axis=1)
    tmp.sort_values(['dataset','pseeds'], ascending=True, inplace=True)

    tmp.kind = tmp.apply(lambda row: 'BA-Homophily' if row.kind == 'BAH' else row.kind ,axis=1)
    tmp.sort_values(["H","B"], inplace=True)
    
    x = 'pseeds'
    y = 'ROCAUC'
    col = 'dataset'
    hue = 'kind'
    #colors = sns.color_palette("Paired")
    #subfigurelabel = ['a','b','c','d','e','f']
    
#     fig,axes = plt.subplots(1,tmp[col].nunique(),figsize=(2.2*tmp[col].nunique(),2.0))
#     for c,dataset in enumerate(tmp[col].unique()):
#         for h,kind in enumerate(tmp[hue].unique()):
            
#             if kind == 'empirical':
#                 _data = tmp.query("dataset==@dataset & kind==@kind").copy()
#                 _err = _data.groupby(['dataset','kind','pseeds']).std().reset_index().sort_values('pseeds')
#                 _data = _data.groupby(['dataset','kind','pseeds']).mean().reset_index().sort_values('pseeds')
#                 axes[c].errorbar(_data[x].values, _data[y].values, yerr=_err[y].values,
#                                  color=colors[h], 
#                                  alpha=1.0,
#                                  zorder=100,
#                                  label=kind)
#             else:
#                 for i in tmp['i'].unique():
#                     for epoch in tmp['epoch'].unique():
#                         _data = tmp.query("dataset==@dataset & kind==@kind & i==@i & epoch==@epoch").copy()
#                         axes[c].plot(_data[x].values, _data[y].values,
#                                      color=colors[h], 
#                                      alpha=0.5,
#                                      zorder=1, 
#                                      label=kind)
            
#         axes[c].set_title("{}".format(dataset))
    
    fg = sns.catplot(data=tmp,
                     kind='point',
                     estimator=np.mean,
                     n_boot=1000,
                     height=2.0,
                     aspect=0.85,
                     palette="Paired",
                     col=col, x=x, y=y,
                     hue=hue)

    fg.set_titles("{col_name}")
    subfigurelabel = ['a','b','c','d','e','f']

    for i,ax in enumerate(fg.axes.flatten()):
    #for i in np.arange(tmp[col].nunique()):
        #ax = axes[i]
        ax.axhline(0.5, ls="--", c='grey', lw=1.0)
        ax.set_ylim(0.4, 1.05)
        _set_minimal_xticklabels(ax)

        dataset = ax.get_title()
        _tmp = tmp.query(" dataset==@dataset & kind=='empirical' ")

        # symmetric homophily
        ax.text(s="H={}\nB={}".format(_tmp.H.unique()[0],_tmp.B.unique()[0]),x=1,y=0.8)
        
        # asymmetric hompohily
#         ax.text(s="H={} ({}, {})\nB={}".format(_tmp.H.unique()[0],
#                                                 _tmp.Hmm.unique()[0],_tmp.HMM.unique()[0],
#                                                 _tmp.B.unique()[0]), x=1, y=0.8)
        
        ####@todo: uncomment the following but before change accorddingly in paper
        #ax.text(s="$h={}$\n$f_m={}$".format(_tmp.H.unique()[0],
        #                                    _tmp.B.unique()[0]), x=1 if i!=2 else 0., y=0.8 if i!=2 else 0.8)

        ax.set_title("{}) {}".format(subfigurelabel[i],dataset))

        if i == int(fg.axes.shape[1]/2.):
        #if i == int(tmp[col].nunique()/2.):
            ax.set_xlabel('pseeds \%')
        else:
            ax.set_xlabel('')

    plt.subplots_adjust(wspace=0.1)

    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        #plt.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))

    plt.show()
    plt.close()


############################################################################################################
# Plots RQ2
############################################################################################################

def plot_SE_vs_pseeds_per_H_B_sampling(df, columns, fn=None):
    y = columns['SE']
    row = columns['H']
    col = columns['B']
    hue = columns['sampling']
    x = columns['pseeds']
    hue_order = _sort_sampling_methods(df[hue].unique())

    toplegend = True
    palette = "tab10"
    _plot_by(df, x, y, row, col, hue, hue_order=hue_order, fn=fn, ylabel=(True, True), legend=False, toplegend=toplegend, yticklabels=True, kind="line", logy=False, palette=palette)

def plot_rocauc_vs_SE_per_H_B_sampling(df, columns, fn=None):
    y = columns['rocauc']
    row = columns['H']
    col = columns['B']
    hue = columns['sampling']
    x = columns['SE']
    hue_order = _sort_sampling_methods(df[hue].unique())

    toplegend = True
    palette = "tab10"
    _plot_by(df, x, y, row, col, hue, hue_order=hue_order, fn=fn, ylabel=(True, True), legend=False, toplegend=toplegend, yticklabels=True, kind="scatter", logy=False, palette=palette)

def plot_SEp1_vs_SEcpDiff_per_H_B_sampling(df, columns, fn=None):
    x = columns['SEcpDiff']
    y = columns['SEp1']
    row = columns['H']
    col = columns['B']
    hue = columns['sampling']
    hue_order = _sort_sampling_methods(df[hue].unique())

    toplegend = True
    palette = "tab10"
    _plot_by(df, x, y, row, col, hue, hue_order=hue_order, fn=fn, ylabel=(True, False), legend=True, toplegend=toplegend, yticklabels=True, kind="scatter", logy=False, palette=palette)

def plot_estimation_errors_per_H_B_rocauc(df, columns, metricx, metricy,fn=None):
    x = columns[metricx]
    y = columns[metricy]
    row = columns['H']
    col = columns['B']
    hue = columns['rocauc']
    palette = "BrBG"

    tmp = df.copy()
    tmp.loc[:, hue] = tmp.apply(lambda row: round(row[hue], 1), axis=1)
    col_order = sorted(tmp[col].unique(), reverse=True)

    _plot_by(tmp, x, y, row, col, hue, hue_order=None, fn=fn, ylabel=(True, True), legend=True,
             toplegend=False, yticklabels=True, kind="scatter", logy=False, palette=palette,
             xlabel=True, xlim=(None, None), ylim=(None, None), col_order=col_order, shortaxislabels=True,
             height=1.5, aspect=1.2, grid=True, xlabelpos=(-0.08, -0.2), ylabelpos=(-0.105, 0.05))
    return

def plot_estimation_errors_per_H_B_rocauc_sampling(df, columns, metricx, metricy, sampling=None, fn=None):

    validate_metric(metricx)
    validate_metric(metricy)

    tmp = df.copy()
    
    hue = columns["SE"] if metricx == 'bias' else columns['rocauc'] if metricx == 'SEcpSum' else None
    vmin, vmax = tmp[hue].min(), tmp[hue].max()
    
    if sampling in [None, 'all', 'ALL', 'All']:
        H = 0.5  # not included
        B = 0.3  # not included

        # Selecting only specific types of networks
        tmp = tmp.query("H!=@H & B!=@B")

        # Plotting all sample techniques (one plot for each)
        for sampling in _sort_sampling_methods(df.sampling.unique()):
            ylabel = (sampling == 'nodes', 'partial' in sampling)
            legend = 'partial' in sampling and metricx in ['bias','SEcpSum']
            xlabelpos = (-0.1, -0.3) if metricx == 'SE' else (-0.98, -0.45) if metricx=='bias' else (-0.35, -0.11)
            ylabelpos = (-0.25, 0.21) if metricx == 'SE' else (-0.65, 0.32) if metricx=='bias' else (-0.25, 0.0)
            yticklabels = sampling == 'nodes'
            
            # for CNA2020 applied network science
            ylabel = (sampling in ['nodes','degree'], 'partial' in sampling or 'nedges' in sampling)
            #legend = ('partial' in sampling or 'nedges' in sampling) and metricx in ['bias','SEcpSum']
            legend = True and metricx in ['bias','SEcpSum']
            yticklabels = sampling in ['nodes','degree']
            

            print(sampling)
            _plot_estimation_errors_per_H_B_rocauc_sampling(
                tmp.query('sampling==@sampling'), columns,
                metricx=metricx, metricy=metricy,
                ylabel=ylabel,
                legend=legend,
                yticklabels=yticklabels,
                shortaxislabels=True,
                xlim=(-0.1,1.1) if metricx=='bias' else (-0.03, tmp[columns[metricx]].max()+0.03),
                ylim=(0.1,1.1) if metricy=='rocauc' else (-0.5,0.5) if metricy=='EEp1' else (-0.018, tmp[columns[metricy]].max()+0.03),
                height=1.5, aspect=1.1,
                grid=False,
                fn=fn if fn is None else fn.replace('<sampling>',sampling).replace('\_',''),
                xlabelpos=xlabelpos, ylabelpos=ylabelpos,
                vmin=vmin, vmax=vmax
            )
    else:
        # Plotting only 1 sampling technique (full plot)
        # available sampling methods: nodes, neighbors, nedges, degree, partial
        _plot_estimation_errors_per_H_B_rocauc_sampling(
            tmp.query("sampling.str.startswith(@sampling)", engine='python'), columns,
            metricx=metricx, metricy=metricy,
            ylabel=(True,True),
            legend=True,
            yticklabels=True,
            shortaxislabels=True,
            xlim=(-0.03, tmp[columns[metricx]].max() + 0.03),
            ylim=(0.1,1.1) if metricy=='rocauc' else (-0.018, tmp[columns[metricy]].max() + 0.03),
            height=1.5, aspect=1.2,
            grid=True,
            fn=fn if fn is None else fn.replace('<sampling>',sampling),
            xlabelpos=(-0.3, -0.11), ylabelpos=(-0.2, 0.0))

#def _round_nearest(x, a):
#    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

def _round_nearest(x,a):
    return round(round(x/a)*a ,2)

def _plot_estimation_errors_per_H_B_rocauc_sampling(df, columns, metricx, metricy,
                                                    ylabel=(True,True), legend=True, yticklabels=True,
                                                    shortaxislabels=True,xlim=(None,None),ylim=(None,None),
                                                    height=1.2,aspect=1.2,grid=False,
                                                    fn=None, **kwargs):
    x = columns[metricx]
    y = columns[metricy]
    row = columns['H']
    col = columns['B']
    hue = columns["SE"] if x == 'bias' else columns['rocauc'] if x == 'SEcpSum' else None
    palette = "BrBG" if hue == columns['rocauc'] else 'RdYlGn_r' if hue == columns["SE"] else None

    tmp = df.copy()
    col_order = sorted(tmp[col].unique(), reverse=True)

    if hue is not None:
        if hue.lower() == 'se':
            tmp.loc[:, hue] = tmp.apply(lambda row: round(row[hue], 2), axis=1)   
        elif hue.lower() == 'rocauc':
            tmp.loc[:, hue] = tmp.apply(lambda row: round(row[hue], 1), axis=1)
        
    #tmp.loc[:, hue] = tmp.apply(lambda row: round(row[hue], 1), axis=1)
    #tmp.loc[:, hue] = tmp.apply(lambda row: _round_nearest(row[hue], 0.05), axis=1)
    #if x == 'bias':
    #    tmp.SE = tmp.SE.round(1)

    _plot_by(tmp, x, y, row=row, col=col, hue=hue,
             kind="scatter", fn=fn,
             ylabel=ylabel,
             legend=legend, toplegend=False,
             yticklabels=yticklabels, xlabel=True,
             logy=False,
             height=height,
             aspect=aspect,
             xlim=xlim,
             ylim=ylim,
             shortaxislabels=shortaxislabels,
             grid=grid,
             palette=palette,
             col_order=col_order,
             baselines=True,
             **kwargs)


############################################################################################################
# Plots RQ3
############################################################################################################

def plot_bias_vs_pseeds_per_B_H_sampling(df, columns, fn=None):
    y = columns['bias']
    row = columns['H']
    col = columns['B']
    hue = columns['sampling']
    hue_order = []
    
    tmp = df.copy()
    tmp.loc[:,hue] = tmp.apply(lambda row:'edges' if row[hue]=='nedges' else row[hue], axis=1)
    hue_list = tmp[hue].unique()
    
    for ho in ['nodes', 'neighbors', 'edges', 'degree', 'partial\_crawls', 'partial_crawls']:
        if ho in hue_list:
            hue_order.append(ho)
    
    toplegend = True
    palette = "tab10"

    col_order = sorted(tmp[col].unique(), reverse=True)

    #3
    _plot_by_pseeds(tmp, y, row, col, hue, hue_order=hue_order,
                    fn=fn, ylabel=(True, True),
                    legend=True, toplegend=toplegend, yticklabels=True, kind="bar",
                    logy=False, palette=palette, col_order=col_order)


def plot_y_vs_pseeds_per_B_H_sampling(df, columns, y='rocauc', fn=None):
    y = columns[y]
    row = columns['H']
    col = columns['B']
    hue = columns['sampling']

    hue_order = []
    hue_list = df[hue].unique()
    for ho in ['nodes', 'neighbors', 'nedges', 'degree', 'partial\_crawls', 'partial_crawls']:
        if ho in hue_list:
            hue_order.append(ho)

    toplegend = True
    palette = "tab10"

    tmp = df.copy()
    col_order = sorted(tmp[col].unique(), reverse=True)

    #3
    _plot_by_pseeds(tmp, y, row, col, hue, hue_order=hue_order,
                    fn=fn, ylabel=(True, True),
                    legend=True, toplegend=toplegend, yticklabels=True, kind="bar",
                    logy=False, palette=palette, col_order=col_order)
    
############################################################################################################
# Setup / Handlers
############################################################################################################

def plot_setup(latex=True):
    mpl.rcParams.update(mpl.rcParamsDefault)

    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        rc('text', usetex=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['text.usetex'] = True

        lw = 0.8
        sns.set_context("paper", rc={"lines.linewidth": lw})
    else:
        sns.set_context('paper', font_scale=1.2)

def validate_metric(metric):
    if metric not in METRIC.keys():
        raise ValueError('{} does not exist as a metric. Please use any of the following: \n{}'.format(metric, METRIC.keys()))

def _sort_sampling_methods(sampling_list):
    order = []
    for s in ['nodes', 'neighbors', 'nedges', 'degree', 'partial\_crawls', 'partial_crawls']:
        if s in sampling_list:
            order.append(s)
    return order

def _get_short_axis_label(label):
    if label.lower() in ['rocauc','bias']:
        return label

    if label == "SE":
        s = "\sum SE" #"SE_{min} + SE_{min|min} + SE_{maj|maj}"
        return r'${}$'.format(s)

    if label not in ['SEcpDiff', 'SEcpSum']:
        if 'cp' in label:
            s = label.replace('cp', '_{').replace('11', 'min|min}').replace('00', 'maj|maj}')
        else:
            s = label.replace('p', '_{').replace('1', 'min}').replace('0', 'maj}')
        s = r'${}$'.format(s)
    else:
        if label == 'SEcpDiff':
            s = r'$SE_{maj|maj} - SE_{min|min}$'
        else:
            s = r'$SE_{maj|maj} + SE_{min|min}$'
    return s

def _set_minimal_xticklabels(ax):
    labels = ax.get_xticklabels()  # get x labels
    for i, l in enumerate(labels):
        if (i not in [1, 5, 9]): labels[i] = ''  # skip even labels
    ax.set_xticklabels(labels, rotation=0)

def _plot_lines_simple(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    #mean = kwargs.pop("mean")
    sns.pointplot(data=data, x=x, y=y, ci='sd', estimator=np.mean, ax=ax, **kwargs)
    ax.axhline(y=0.5, color='grey', linestyle=':', lw=0.8, label="random")
    ax.grid(False)

def _plot_bars(x, y, **kwargs):
    width = 0.15

    ax = plt.gca()
    data = kwargs.pop("data")
    g = data.groupby(['B', 'H', 'pseeds', 'sampling'])  # 'N',m
    means = g[y].mean().reset_index()
    errors = g[y].std()
    logy = kwargs.pop("logy")

    #span = {'nodes': width * 0, 'neighbors': width * 1, 'nedges': width * 2, 'degree': width * 3, 'partialcrawls': width * 4}
    span = {'nodes': width * 0, 'edges': width * 1, 'degree': width * 2, 'partialcrawls': width * 3}
    sampling = data.sampling.unique()[0].replace("_", "").replace("\\", "")
    span = span[sampling]

    xticks = np.arange(1, means.pseeds.nunique() + 1, 1)
    ax.bar(xticks + span, means[y], width, yerr=errors, bottom=0, **kwargs)

    ax.set_xticks(xticks + width + width)
    ax.set_xticklabels(sorted(data.pseeds.astype(np.int).unique()))

    if logy:
        ax.set_yscale('symlog')

def _plot_lines(x, y, **kwargs):

    ax = plt.gca()
    data = kwargs.pop("data")
    g = data.groupby(['B', 'H', x, 'sampling'])  # 'N',m
    means = g[y].mean().reset_index()

    if 'errors' in kwargs:
        if not kwargs.pop('errors'):
            ax.plot(means[x], means[y], **kwargs)
            return

    errors = g[y].std()
    ax.errorbar(means[x], means[y], yerr=errors, **kwargs)

def _plot_scatter(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    plt.scatter(data[x], data[y], **kwargs)

def _plot_by_pseeds(df, y, row, col, hue, hue_order, fn=None, ylabel=(True, True), legend=True, toplegend=False, yticklabels=True, kind="line", logy=False, palette=False, col_order=None, row_order=None, height=1.2, aspect=1.2):
    _plot_by(df, 'pseeds', y, row, col, hue,
            hue_order,
            fn,
            ylabel,
            legend,
            toplegend,
            yticklabels,
            kind,
            logy,
            palette,
            col_order=col_order,
            row_order=row_order,
            height=height,
            aspect=aspect
     )

def _plot_by(df, x, y, row, col, hue, hue_order=None, fn=None, ylabel=(True,True), legend=True,
             toplegend=False, yticklabels=True, kind="line", logy=False, palette=False,
             xlabel=True, xlim=(None,None), ylim=(None,None), col_order=None, shortaxislabels=False,
             grid=False, height=1.2, aspect=1.2, baselines=True, row_order=None, **kwargs):

    plt.close()
    baseline = {'ROCAUC': 0.5, 'bias': 0.5, 'SE': 0, 'EE':0}

    tmp = df.copy()
    tmp.loc[:,'pseeds'] = tmp.apply(lambda row: int(round(row['pseeds']*100,0)), axis=1)

    fg = sns.FacetGrid(data=tmp, col=col, row=row, hue=hue,
                       hue_order=hue_order,
                       col_order=col_order,
                       row_order=row_order,
                       margin_titles=True,
                       height=height if tmp[col].nunique() > 1 else 2,
                       aspect=aspect if tmp[col].nunique() > 1 else 0.75,
                       dropna=False,
                       palette=palette)

    if kind == 'bar':
        fg = fg.map_dataframe(_plot_bars, x, y, logy=logy)
    elif kind == 'line':
        fg = fg.map_dataframe(_plot_lines, x, y, marker='o', lw=1.0, alpha=1.0)
    elif kind == 'scatter':
        if 'vmin' not in kwargs:
            vmin,vmax = 0.0,1.0
        else:
            vmin,vmax = kwargs['vmin'],kwargs['vmax']
        vmin,vmax=None,None
        fg = fg.map_dataframe(_plot_scatter, x, y, marker='o', lw=1.0, alpha=0.5 if x in ['SE','bias'] else 1.0, 
                              vmin=vmin, vmax=vmax)

    x = unlatexfyme(x)
    y = unlatexfyme(y)

    for ax in fg.axes.flatten():
                
        if baselines:
            if 'SE' in x or 'EE' in x:
                ax.axvline(baseline[x[:2]], lw=1.0, ls='--', c='pink')
            if 'SE' in y or 'EE' in y:
                ax.axhline(baseline[y[:2]], lw=1.0, ls='--', c='pink')
            if 'bias' in x:
                ax.axvline(baseline[x], lw=1.0, ls='--', c='pink')

    if shortaxislabels:
        x = _get_short_axis_label(x)
        y = _get_short_axis_label(y)

    if legend:
        if not toplegend:
        
            fg.add_legend()
            
#             lp = lambda i: plt.plot([], color=cmap(norm(i)), marker="o", ls="", ms=10, alpha=0.5)[0]
#             labels = np.arange(0,7.5,0.5)
#             h = [lp(i) for i in labels]
#             g.fig.legend(handles=h, labels=labels, fontsize=9)

            if hue=='SE':
                new_title = r'$\sum SE$'
                fg._legend.set_title(new_title)
        else:
            fg.axes[0, 0].legend(loc='lower left',
                                 bbox_to_anchor=(-0.08, 1.3, 0.1, 1),  # -0.25
                                 borderaxespad=0,
                                 labelspacing=0,
                                 handlelength=1,
                                 frameon=False,
                                 ncol=df[hue].nunique())

    for ax in fg.axes.flatten():

        if baselines:
            try:
                ax.axhline(baseline[y], lw=1.0,
                           ls='-' if y not in ['bias','ROCAUC'] else '--',
                           c='pink' if y not in ['bias'] else 'grey')
            except Exception as ex:
                #print(ex)
                pass

        ax.set_xlabel('')
        ax.set_ylabel('')

        if y in ['ROCAUC', 'bias']:
            ax.set_ylim((-0.1, 1.1))

        if xlim[0] is not None:
            ax.set_xlim(xlim)
        if ylim[0] is not None:
            ax.set_ylim(ylim)

        if logy:
            ax.set_yscale('log')

        if grid:
            ax.grid(True, lw=0.5, ls='--')

    # xlabel
    try:
        if xlabel:
            if x == 'bias':
                #s = r"$bias=\frac{CC_{min}}{CC_{min}+CC_{maj}}$"
                s = r"${bias}=\frac{TPR_{min}}{TPR_{min}+TPR_{maj}}$"
                if col is None:
                    fg.ax.set_xlabel(s=s,fontsize=13)
                else:
                    xx, yy = kwargs['xlabelpos']
                    c = int(tmp[col].nunique()/2)
                    fg.axes[-1, c].text(s=s, x=xx, y=yy, fontsize=13)
            else:

                s = x if x not in METRIC else METRIC[x]

                if col is None:
                    fg.axes[-1, 0].set_xlabel(s)

                elif tmp[col].nunique() % 2 == 0:
                    c = int(tmp[col].nunique()/2)
                    xx,yy = kwargs['xlabelpos']
                    fg.axes[-1, c].text(s=s, x=xx, y=yy)

                    #for c in np.arange(tmp[col].nunique()):
                    #    fg.axes[-1, c].set_xlabel(x if x not in METRIC else METRIC[x])
                else:
                    fg.axes[-1, int(tmp[col].nunique()/2)].set_xlabel(s)

    except Exception as ex:
        print(ex)

    # ylabel
    try:

        if ylabel[0]:

            if y == 'bias':
                s = r"${bias}=\frac{TPR_{min}}{TPR_{min}+TPR_{maj}}$"
                fg.axes[int(round(tmp[row].nunique()/2,0))-1, 0].set_ylabel(s, fontsize=13)
            else:
                if row is None:
                    fg.axes[0, 0].set_ylabel(y if y not in METRIC else METRIC[y])
                elif tmp[row].nunique() % 2 == 0:
                    r = int(round(tmp[row].nunique()/2,0))-1
                    s = y if y not in METRIC else METRIC[y]

                    xx, yy = kwargs['ylabelpos']
                    fg.axes[r, 0].text(s=s, x=xx, y=yy, rotation=90)
                    #for r in np.arange(tmp[row].nunique()):
                    #    fg.axes[r, 0].set_ylabel(y if y not in METRIC else METRIC[y])
                else:
                    fg.axes[int(round(tmp[row].nunique()/2,0))-1, 0].set_ylabel(y if y not in METRIC else METRIC[y])

    except Exception as ex:
        print(ex)


    # ylabel on the right
    if not ylabel[1]:
        for r in np.arange(0, df[row].nunique()):
            fg.axes[r, -1].texts = fg.axes[r, -1].texts[1:]

    # yticklabels
    if not yticklabels:
        for r in np.arange(0, df[row].nunique()):
            for c in np.arange(0, df[col].nunique()):
                fg.axes[r, c].set_yticklabels([])

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print("{} saved!".format(fn))

    plt.show()
    plt.close()




####
# new code:
####

def new_plot_estimation_errors_per_H_B_rocauc_sampling(data, columns, col, row, hue, x, y, sampling, ylabel=(True,True), legend=True, fn=None):
    print(sampling)
    rows = data[columns[row]].nunique()
    cols = data[columns[col]].nunique()
    if hue is not None:
        palettes = {'SE':"RdYlGn_r", 'rocauc':"BrBG"}
        vmin,vmax,nse = data[columns[hue]].min(), data[columns[hue]].max(), data[columns[hue]].nunique()
        colorse = [sns.color_palette(palettes[hue], n_colors=nse),sorted(data[columns[hue]].unique().astype(str))]
        colors = {se:c for c,se in zip(*colorse)}
    else:
        vmin,vmax,nse = None, None, None

    xmin,xmax = data[columns[x]].min(), data[columns[x]].max()
    ymin,ymax = data[columns[y]].min(), data[columns[y]].max()
    xmid = (xmax+xmin)/2
    ymid = (ymax+ymin)/2
    
    ### figure
    plt.close()
    fig,axes = plt.subplots(rows,cols,figsize=(3.5,3),sharex=True, sharey=True)

    i = -1
    n = 2
    tmp = data.query("sampling==@sampling").copy()
    
    if hue:
        tmp = tmp.sort_values([columns[row],columns[col],columns[hue]],ascending=[True,False,hue=='SE']).copy()
    else:
        tmp = tmp.sort_values([columns[row],columns[col]],ascending=[True,False]).copy()
        
    for _, df1 in tmp.groupby([row,col],sort=False):
        i += 1
        r = int(i/n)
        c = i%n

        if hue:            
            for _, df2 in df1.groupby([columns[hue]], sort=True):  
                sc = axes[r,c].scatter(df2[columns[x]], df2[columns[y]], 
                                       c=colors[str(df2[columns[hue]].unique()[0])] if hue else None, 
                                       s=35, alpha=0.5, 
                                       vmin=vmin, vmax=vmax)
        else:
            sc = axes[r,c].scatter(df1[columns[x]], df1[columns[y]], 
                                   s=35, alpha=0.5, 
                                   vmin=vmin, vmax=vmax)
        
        if hue:
            axes[r,c].text(x=xmid,y=ymid,s=r'$\overline{'+columns[hue]+'}'+'={}$'.format(round(df1[columns[hue]].mean(),2)), 
                           fontsize=8, va='center', ha='center', 
                           bbox=dict(facecolor='white', edgecolor='white', pad=1))

        if r==0:
            # columns
            axes[r,c].set_title("{} = {}".format(columns[col], df1[columns[col]].unique()[0]))

        if c==1 and ylabel[1]:
            # y-label (right)
            pos = xmax + (0.2 if x=='bias' else 0.07 if x=='SEcpSum' else 0.06 if x=='SE' else 0.05 if x=='SEcp00' else 0)
            axes[r,c].text(x=pos, y=ymid, s="{} = {}".format(columns[row], df1[columns[row]].unique()[0]), 
                           va='center', ha='center', rotation=-90)

        ### right and top borders
        axes[r,c].spines['right'].set_visible(False)
        axes[r,c].spines['top'].set_visible(False)
        
        ### baselines
        axes[r,c].axhline(0.5 if y in ['bias','rocauc'] else 0, c='pink', lw=1.0, ls='--')
        axes[r,c].axvline(0.5 if x in ['bias','rocauc'] else 0, c='pink', lw=1.0, ls='--')
        
        ### x and y limits
        smooth=0.1 if x in ['bias','rocauc'] else 0.03
        axes[r,c].set_xlim((xmin-smooth,xmax+smooth))
        smooth=0.1 if y in ['bias','rocauc'] else 0.03
        axes[r,c].set_ylim((ymin-smooth,ymax+smooth))

    ### unique legend for all:
    if legend and hue:
        handles=[Line2D([0], [0], marker='o', color=c, label=se, markersize=5, linestyle='', alpha=0.5) for c,se in zip(*colorse)]
        steps = 2 if hue=='SE' else 1
        hues = '{}-s'.format(hue)
        title =  METRIC[hues] if hues in METRIC else METRIC[hue] if hue in METRIC else hue
        pos = (1.15,2.2) if hue == 'SE' else (1.7,1.7) if hue == 'rocauc' else (0,0)
        plt.legend(handles=handles[::steps], bbox_to_anchor=pos, 
                   borderaxespad=0, frameon=False, title=title)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)

    ### y-labels left
    if ylabel[0]:
        ys = '{}-s'.format(y)
        s =  METRIC[ys] if ys in METRIC else METRIC[y] if y in METRIC else y
        pos = (-0.55,0.2) if x=='bias' else (-0.2,0.2) if x=='SE' else (-0.18,-0.05) if y=='SEp1' else (-0.12,-0.05) if x=='SEcp00' else (0,0) 
        axes[0,0].text(x=pos[0], y=pos[1], s=s, rotation=90, va='center', ha='center', fontsize=13)

    ### x-labels
    xs = '{}-s'.format(x)
    s =  METRIC[xs] if xs in METRIC else METRIC[x] if x in METRIC else x
    pos = (-0.16,-0.2) if x=='bias' else (-0.02,-0.12) if x=='SEcpSum' else (-0.02,-0.15) if x=='SE' else (-0.03,-0.20) if x=='SEcp00' else (0,0) 
    axes[1,1].text(x=pos[0], y=pos[1], s=s, rotation=0, va='center', ha='center', fontsize=13)

    ### save figure
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    if fn is not None:
        fn = fn.replace("\_",'')
        plt.savefig(fn, bbox_inches='tight')
        print("{} saved!".format(fn))

    ### show
    plt.show()
    plt.close()
    
    
###########################################################################################    
# Some old code
###########################################################################################
    
    # tmp = df.copy()
    #
    # evaluation = columns['bias']
    # xaxis = columns['pseeds']
    # hue = columns['sampling']
    #
    # fg = sns.catplot(data=tmp,
    #                  x=xaxis,
    #                  y=evaluation,
    #                  col=columns['B'],
    #                  row=columns['H'],
    #                  hue=hue,
    #                  ci='sd',
    #                  kind='bar',
    #                  margin_titles=True,
    #                  height=1.7,
    #                  aspect=0.9,
    #                  legend=True,
    #                  )
    #
    # for aa in np.ndenumerate(fg.axes):
    #     coord = aa[0]
    #
    #     if coord != (1, 1):
    #         aa[1].set_xlabel("")
    #
    #     if coord[0] == 1:
    #         labels = aa[1].get_xticklabels()  # get x labels
    #         for i, l in enumerate(labels):
    #             if (i not in [1, 5, 9]): labels[i] = ''  # skip even labels
    #         aa[1].set_xticklabels(labels, rotation=0)
    #
    #     aa[1].axhline(0.5, lw=0.5, c="grey", ls="--")
    #     aa[1].set_ylim((0,1))
    #
    # plt.subplots_adjust(hspace=0.05, wspace=0.05)
    #
    # if fn is not None:
    #     plt.savefig(fn, bbox_inches='tight')
    # plt.show()
    # plt.close()



    # tmp = df.copy()
    #
    # evaluation = 'rocauc'
    # xaxis = columns['pseeds']
    # hue = columns['network_size']
    #
    # hue_order = tmp[hue].unique()
    # fg = sns.FacetGrid(tmp[[columns[c] for c in ['network_size', 'H', 'B', evaluation, 'pseeds']]],
    #                    col=columns["H"], row=columns['B'],
    #                    hue=hue, hue_order=hue_order, palette='Paired',
    #                    sharex=True, sharey=True,
    #                    height=1, aspect=0.8,
    #                    margin_titles=True, legend_out=True)
    # fg = fg.map_dataframe(_plot_lines, columns[xaxis], columns[evaluation], mean=tmp[columns[evaluation]].mean())
    #
    # labels = hue_order
    # colors = sns.color_palette('Paired').as_hex()[:len(labels)]
    # handles = [patches.Patch(color=col, label=lab) for col, lab in zip(colors, labels)]
    # ncols = tmp[columns['network_size']].nunique()
    # fg.fig.legend(handles=handles, title=hue,
    #               bbox_to_anchor=(0.5 - (0.032 * ncols), 0.98, 0.93, 0.18),
    #               loc='lower left', ncol=ncols)
    #
    # # 4 networkls size: bbox_to_anchor=(0.5-(0.056*ncols), 0.98, 0.93, 0.18),
    # # "Network size, minimum degree"
    # # bbox_to_anchor = (x, y, width, height)
    # # loc = lower left (from top-left corner)
    #
    # for aa in np.ndenumerate(fg.axes):
    #     coord = aa[0]
    #     if coord != (1, 0):
    #         aa[1].set_ylabel("")
    #     if coord != (2, 5):
    #         aa[1].set_xlabel("")
    #     if coord[0] == 2:
    #         labels = aa[1].get_xticklabels()  # get x labels
    #         for i, l in enumerate(labels):
    #             if (i not in [1, 5, 9]): labels[i] = ''  # skip even labels
    #         aa[1].set_xticklabels(labels, rotation=0)
    #
    # plt.subplots_adjust(hspace=0.05, wspace=0.05)
    #
    # if fn is not None:
    #     fg.savefig(fn, bbox_inches='tight')
    #     print("{} saved!".format(fn))
    #
    # plt.show()
    # plt.close()


# def plot_estimation_errors_per_rocauc_sampling(df, columns, metricx, metricy,fn=None):
#     x = columns[metricx]
#     y = columns[metricy]
#     row = None
#     col = columns['sampling']
#     hue = columns['rocauc']
#     palette = "BrBG"
#
#     tmp = df.copy()
#     sampling_order = _sort_sampling_methods(tmp['sampling'].unique())
#     #tmp = tmp.groupby(['sampling','pseeds']).mean().reset_index()
#     tmp.loc[:, hue] = tmp.apply(lambda row: round(row[hue], 1), axis=1)
#
#     _plot_by(tmp, x, y, row=row, col=col, hue=hue, hue_order=None,col_order=sampling_order,
#              kind="scatter", fn=fn,
#              ylabel=(True,True),
#              legend=True, toplegend=False,
#              yticklabels=True, xlabel=True,
#              logy=False,
#              height=2.0,
#              aspect=0.9,
#              xlim=(-0.3,0.6),
#              ylim=(-0.6,0.6),
#              palette=palette)
