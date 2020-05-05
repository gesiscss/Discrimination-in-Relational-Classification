import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy
from matplotlib import rc
from palettable.colorbrewer.diverging import RdBu_11

############################################################################################################
# Constants
############################################################################################################

# Available metrics

METRIC = {'ROCAUC': 'ROCAUC', 'bias': 'Bias',

          # Estimation Error: estimation - observed
          'EEp1': r'$P_{min} - \theta_{min}$',
          'EEp0': r'$P_{maj} - \theta_{maj}$',
          'EEcp11': r'$P_{min|min} - \theta_{min|min}$',
          'EEcp01': r'$P_{maj|min} - \theta_{maj|min}$',
          'EEcp00': r'$P_{maj|maj} - \theta_{maj|maj}$',
          'EEcp10': r'$P_{min|maj} - \theta_{min|maj}$',

          # Squared Error: squared estimation error (estimation - observed)^2
          'SEp1': r'$(P_{min} - \theta_{min})^2$',
          'SEp0': r'$(P_{maj} - \theta_{maj})^2$',
          'SEcp11': r'$(P_{min|min} - \theta_{min|min})^2$',
          'SEcp00': r'$(P_{maj|maj} - \theta_{maj|maj})^2$',

          # Comparing the estimation errors
          'SEcpDiff': r"$(\theta_{maj|maj} - P_{maj|maj})^2 - (\theta_{min|min} - P_{min|min})^2$",
          'SEcpSum': r"$(\theta_{maj|maj} - P_{maj|maj})^2 + (\theta_{min|min} - P_{min|min})^2$",
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

    fg = sns.catplot(data=tmp,
                     x=xaxis,
                     y=evaluation,
                     col=columns['B'],
                     row=columns['m'],
                     hue=hue,
                     ci='sd',
                     kind='point',
                     margin_titles=True,
                     height=1.7,
                     aspect=0.9,
                     palette=RdBu_11.mpl_colors,
                     legend=True,
                     )

    for aa in np.ndenumerate(fg.axes):
        coord = aa[0]

        if coord != (1, 1):
            aa[1].set_xlabel("")

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

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_rocauc_vs_pseeds_per_H_B_N_m(df, columns, fn=None):
    y = columns['rocauc']
    row = columns['B']
    col = columns['H']
    hue = columns['network_size']
    toplegend = True
    palette = "Paired"
    _plot_by_pseeds(df, y, row, col, hue, hue_order=None, fn=fn, ylabel=(True, True), legend=True, toplegend=toplegend, yticklabels=True, kind="line", logy=False, palette=palette)

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
    _plot_by_pseeds(df, y, row, col, hue, hue_order=hue_order, fn=fn, ylabel=(True, False), legend=False, toplegend=toplegend, yticklabels=True, kind="line", logy=False, palette=palette)


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

    x = 'pseeds'
    y = 'ROCAUC'
    fg = sns.catplot(data=tmp,
                     kind='point',
                     height=2.0,
                     aspect=0.85,
                     palette="Paired",
                     col='dataset', x=x, y=y,
                     hue='kind')

    fg.set_titles("{col_name}")
    subfigurelabel = ['a','b','c','d','e','f']

    for i,ax in enumerate(fg.axes.flatten()):
        ax.axhline(0.5, ls="--", c='grey', lw=1.0)
        ax.set_ylim(0.4, 1.05)
        _set_minimal_xticklabels(ax)

        dataset = ax.get_title()
        _tmp = tmp.query(" dataset==@dataset & kind=='empirical' ")

        ### todo: show asymmetric homophily
        #ax.text(s="H={}\nB={}".format(_tmp.H.unique()[0],_tmp.B.unique()[0]),x=1,y=0.8)
        ax.text(s="H={} ({}, {})\nB={}".format(_tmp.H.unique()[0],
                                              _tmp.Hmm.unique()[0],_tmp.HMM.unique()[0],
                                              _tmp.B.unique()[0]), x=1, y=0.8)

        ax.set_title("{}) {}".format(subfigurelabel[i],dataset))

        if i == int(fg.axes.shape[1]/2.):
            ax.set_xlabel('pseeds')
        else:
            ax.set_xlabel('')

    plt.subplots_adjust(wspace=0.1)

    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
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

    _plot_by(tmp, x, y, row, col, hue, hue_order=None, fn=fn, ylabel=(True, True), legend=True,
             toplegend=False, yticklabels=True, kind="scatter", logy=False, palette=palette,
             xlabel=True, xlim=(None, None), ylim=(None, None), col_order=None, shortaxislabels=True,
             height=1.5, aspect=1.2, grid=True, xlabelpos=(-0.08, -0.2), ylabelpos=(-0.105, 0.05))
    return

def plot_estimation_errors_per_H_B_rocauc_sampling(df, columns, metricx, metricy, sampling=None, fn=None):

    validate_metric(metricx)
    validate_metric(metricy)

    tmp = df.copy()

    if sampling in [None, 'all', 'ALL', 'All']:
        H = 0.5  # not included
        B = 0.3  # not included

        # Selecting only specific types of networks
        tmp = tmp.query("H!=@H & B!=@B")

        # Plotting all sample techniques (one plot for each)
        for sampling in _sort_sampling_methods(df.sampling.unique()):
            ylabel = (sampling == 'nodes', 'partial' in sampling)
            legend = 'partial' in sampling
            yticklabels = sampling == 'nodes'

            print(sampling)
            _plot_estimation_errors_per_H_B_rocauc_sampling(
                tmp.query('sampling==@sampling'), columns,
                metricx=metricx, metricy=metricy,
                ylabel=ylabel,
                legend=legend,
                yticklabels=yticklabels,
                shortaxislabels=True,
                xlim=(-0.03, tmp[columns[metricx]].max()+0.03),
                ylim=(-0.018, tmp[columns[metricy]].max()+0.03),
                height=1.5, aspect=1.1,
                grid=True,
                fn=fn.replace('<sampling>',sampling).replace('\_',''),
                xlabelpos=(-0.35, -0.11), ylabelpos=(-0.25, 0.0))
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
            ylim=(-0.018, tmp[columns[metricy]].max() + 0.03),
            height=1.5, aspect=1.2,
            grid=True,
            fn=fn.replace('<sampling>',sampling),
            xlabelpos=(-0.3, -0.11), ylabelpos=(-0.2, 0.0))

def _plot_estimation_errors_per_H_B_rocauc_sampling(df, columns, metricx, metricy,
                                                    ylabel=(True,True), legend=True, yticklabels=True,
                                                    shortaxislabels=True,xlim=(None,None),ylim=(None,None),
                                                    height=1.2,aspect=1.2,grid=False,
                                                    fn=None, **kwargs):
    x = columns[metricx]
    y = columns[metricy]
    row = columns['H']
    col = columns['B']
    hue = columns['rocauc']
    palette = "BrBG"

    tmp = df.copy()
    tmp.loc[:, hue] = tmp.apply(lambda row: round(row[hue], 1), axis=1)

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
    hue_list = df[hue].unique()
    for ho in ['nodes', 'neighbors', 'nedges', 'degree', 'partial\_crawls', 'partial_crawls']:
        if ho in hue_list:
            hue_order.append(ho)

    toplegend = True
    palette = "tab10"
    _plot_by_pseeds(df, y, row, col, hue, hue_order=hue_order, fn=fn, ylabel=(True, True), legend=True, toplegend=toplegend, yticklabels=True, kind="bar", logy=False, palette=palette)



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
    span = {'nodes': width * 0, 'nedges': width * 1, 'degree': width * 2, 'partialcrawls': width * 3}
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
    ax.scatter(data[x], data[y], **kwargs)

def _plot_by_pseeds(df, y, row, col, hue, hue_order, fn=None, ylabel=(True, True), legend=True, toplegend=False, yticklabels=True, kind="line", logy=False, palette=False):
    _plot_by(df, 'pseeds', y, row, col, hue, hue_order, fn, ylabel, legend, toplegend, yticklabels, kind, logy, palette)

def _plot_by(df, x, y, row, col, hue, hue_order=None, fn=None, ylabel=(True,True), legend=True,
             toplegend=False, yticklabels=True, kind="line", logy=False, palette=False,
             xlabel=True, xlim=(None,None), ylim=(None,None), col_order=None, shortaxislabels=False,
             grid=False, height=1.2, aspect=1.2, **kwargs):

    plt.close()
    baseline = {'ROCAUC': 0.5, 'bias': 0.5, 'SE': 0, 'EE':0}

    tmp = df.copy()
    tmp.loc[:,'pseeds'] = tmp.apply(lambda row: int(round(row['pseeds']*100,0)), axis=1)

    fg = sns.FacetGrid(data=tmp, col=col, row=row, hue=hue,
                       hue_order=hue_order,
                       col_order=col_order,
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
        fg = fg.map_dataframe(_plot_scatter, x, y, marker='o', lw=1.0, alpha=1.0, vmin=0, vmax=1)

    x = unlatexfyme(x)
    y = unlatexfyme(y)

    for ax in fg.axes.flatten():
        if 'SE' in x or 'EE' in x:
            ax.axvline(baseline[x[:2]], lw=0.5, ls='--', c='red')
        if 'SE' in y or 'EE' in y:
            ax.axhline(baseline[x[:2]], lw=0.5, ls='--', c='red')

    if shortaxislabels:
        x = _get_short_axis_label(x)
        y = _get_short_axis_label(y)

    if legend:
        if not toplegend:
            fg.add_legend()
        else:
            fg.axes[0, 0].legend(loc='lower left',
                                 bbox_to_anchor=(-0.08, 1.3, 0.1, 1),  # -0.25
                                 borderaxespad=0,
                                 labelspacing=0,
                                 handlelength=1,
                                 frameon=False,
                                 ncol=df[hue].nunique())

    for ax in fg.axes.flatten():
        try:
            ax.axhline(baseline[y], lw=1, ls='-' if y != 'bias' else '--', c='red' if y!='bias' else 'grey')
        except:
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
                print('uno')
                fg.axes[-1,int(tmp[col].nunique()/2)].set_xlabel(r"$bias=\frac{CC_{min}}{CC_{min}+CC_{maj}}$", fontsize=13)
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
                fg.axes[int(round(tmp[row].nunique()/2,0))-1, 0].set_ylabel(r"$bias=\frac{CC_{min}}{CC_{min}+CC_{maj}}$", fontsize=13)
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
