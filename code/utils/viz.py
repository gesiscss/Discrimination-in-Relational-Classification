import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sympy
from matplotlib import rc
from palettable.colorbrewer.diverging import RdBu_11


def latex_compatible_text(txt):
    return sympy.latex(sympy.sympify(txt)).replace("_", "\_")

def latex_compatible_dataframe(df, latex=True):
    tmp = df.copy()
    if latex:
        cols = {c:c if c=="N" else sympy.latex(sympy.sympify(c)).replace("_","\_") for c in tmp.columns}
        if 'sampling' in tmp.columns:
            tmp.sampling = tmp.apply(lambda row: row.sampling.replace('_', '\_'), axis=1)
    else:
        cols = {c:c for c in tmp.columns}
    cols['rocauc'] = cols['rocauc'].upper()
    tmp.rename(columns=cols, inplace=True)
    return tmp, cols

def plot_rocauc_curve(fpr, tpr, rocauc, fn=None):
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

def _plot_lines(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    mean = kwargs.pop("mean")
    sns.pointplot(data=data, x=x, y=y, ci='sd', estimator=np.mean, ax=ax, **kwargs)
    ax.axhline(y=0.5, color='grey', linestyle=':', lw=0.8, label="random")
    ax.grid(False)

def plot_rocauc_vs_homophily_per_B_m_pseeds(df, columns, fn=None):

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
            labels = aa[1].get_xticklabels()  # get x labels
            for i, l in enumerate(labels):
                if (i not in [1, 5, 9]): labels[i] = ''  # skip even labels
            aa[1].set_xticklabels(labels, rotation=0)

        aa[1].axhline(0.5, lw=0.5, c="grey", ls="--")

        # example
        if coord == (0, 0):
            aa[1].annotate('', xy=(9.1, 0.61),
                           xytext=(10, 0.45),
                           arrowprops={'arrowstyle': '-|>',
                                       'lw': 2,
                                       'ec': 'k', 'fc': 'k'},
                           va='center')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_rocauc_vs_pseeds_per_B_N_m(df, columns, fn=None):

    tmp = df.copy()

    evaluation = 'rocauc'
    xaxis = columns['pseeds']
    hue = columns['network_size']

    hue_order = tmp[hue].unique()
    fg = sns.FacetGrid(tmp[[columns[c] for c in ['network_size', 'H', 'B', evaluation, 'pseeds']]],
                       col=columns["H"], row=columns['B'],
                       hue=hue, hue_order=hue_order, palette='Paired',
                       sharex=True, sharey=True,
                       height=1, aspect=0.8,
                       margin_titles=True, legend_out=True)
    fg = fg.map_dataframe(_plot_lines, columns[xaxis], columns[evaluation], mean=tmp[columns[evaluation]].mean())

    labels = hue_order
    colors = sns.color_palette('Paired').as_hex()[:len(labels)]
    handles = [patches.Patch(color=col, label=lab) for col, lab in zip(colors, labels)]
    ncols = tmp[columns['network_size']].nunique()
    fg.fig.legend(handles=handles, title=hue,
                  bbox_to_anchor=(0.5 - (0.032 * ncols), 0.98, 0.93, 0.18),
                  loc='lower left', ncol=ncols)

    # 4 networkls size: bbox_to_anchor=(0.5-(0.056*ncols), 0.98, 0.93, 0.18),
    # "Network size, minimum degree"
    # bbox_to_anchor = (x, y, width, height)
    # loc = lower left (from top-left corner)

    for aa in np.ndenumerate(fg.axes):
        coord = aa[0]
        if coord != (1, 0):
            aa[1].set_ylabel("")
        if coord != (2, 5):
            aa[1].set_xlabel("")
        if coord[0] == 2:
            labels = aa[1].get_xticklabels()  # get x labels
            for i, l in enumerate(labels):
                if (i not in [1, 5, 9]): labels[i] = ''  # skip even labels
            aa[1].set_xticklabels(labels, rotation=0)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if fn is not None:
        fg.savefig(fn, bbox_inches='tight')
        print("{} saved!".format(fn))

    plt.show()
    plt.close()

def plot_bias_vs_pseeds_per_B_H_sampling(df, columns, fn=None):

    tmp = df.copy()

    evaluation = columns['bias']
    xaxis = columns['pseeds']
    hue = columns['sampling']

    fg = sns.catplot(data=tmp,
                     x=xaxis,
                     y=evaluation,
                     col=columns['B'],
                     row=columns['H'],
                     hue=hue,
                     ci='sd',
                     kind='bar',
                     margin_titles=True,
                     height=1.7,
                     aspect=0.9,
                     legend=True,
                     )

    for aa in np.ndenumerate(fg.axes):
        coord = aa[0]

        if coord != (1, 1):
            aa[1].set_xlabel("")

        if coord[0] == 1:
            labels = aa[1].get_xticklabels()  # get x labels
            for i, l in enumerate(labels):
                if (i not in [1, 5, 9]): labels[i] = ''  # skip even labels
            aa[1].set_xticklabels(labels, rotation=0)

        aa[1].axhline(0.5, lw=0.5, c="grey", ls="--")
        aa[1].set_ylim((0,1))

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_fixed_effects(output=None):
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

    if output is not None:
        fn = os.path.join(output, 'fixed_effects.pdf')
        fig.savefig(fn, bbox_inches='tight')
        print('{} saved!'.format(fn))
    plt.show()
    plt.close()



#     _plot_by_pseeds(df, y, hue_order, fn=None, ylabel=(True, True), legend=True, toplegend=False, ytickslabels=True, bars=False, logy=False)
#
# def _plot_by_pseeds(df, y, hue_order, fn=None, ylabel=(True,True), legend=True, toplegend=False, ytickslabels=True, bars=False, logy=False):
#     tmp, cols = _prepare_plot(df)
#
#     baseline = {'rocauc': 0.5, 'bias': 0.5, 'zscore': 0}
#     ymetric = {'rocauc': 'ROCAUC', 'bias': 'Bias', 'zscore': 'z-score'}
#
#     if bars:
#         tmp = tmp.query("pseeds < 40")
#
#     fg = sns.FacetGrid(data=tmp, col='B', row='H', hue='sampling',
#                        hue_order=hue_order,
#                        margin_titles=True,
#                        height=1.2 if tmp.H.nunique() > 1 else 2,
#                        aspect=1.2 if tmp.H.nunique() > 1 else 0.75,
#                        dropna=False,
#                        )
#
#     if bars:
#         fg = fg.map_dataframe(_plot_bars, 'pseeds', y, logy=logy)
#     else:
#         fg = fg.map_dataframe(_plot_lines, 'pseeds', y, marker='o', lw=1.0, alpha=1.0)
#
#     if legend:
#         if not toplegend:
#             fg.add_legend()
#         else:
#             fg.axes[0, 0].legend(loc='lower left',
#                                  bbox_to_anchor=(-0.08, 1.3, 0.1, 1),  # -0.25
#                                  borderaxespad=0,
#                                  labelspacing=0,
#                                  handlelength=1,
#                                  frameon=False,
#                                  ncol=df.sampling.nunique())
#
#     for ax in fg.axes.flatten():
#         try:
#             ax.axhline(baseline[y], lw=1, ls='--', c='grey')
#         except:
#             pass
#
#         ax.set_xlabel('')
#         ax.set_ylabel('')
#
#         # if not bars:
#         if ylabel in ['rocauc', 'bias']:
#             ax.set_ylim((-0.1, 1.1))
#
#         if logy:
#             ax.set_yscale('log')
#
#     try:
#         fg.axes[2, 1].set_xlabel('pseeds')
#         if ylabel[0]:
#             if y == 'bias':
#                 fg.axes[1, 0].set_ylabel(r"$bias=\frac{CC_{min}}{CC_{min}+CC_{maj}}$", fontsize=13)
#             else:
#                 fg.axes[1, 0].set_ylabel(ymetric[y])
#         if not ylabel[1]:
#             fg.axes[0, 2].texts = []
#             fg.axes[1, 2].texts = []
#             fg.axes[2, 2].texts = []
#         if not ytickslabels:
#             for r in [1, 2]:
#                 for c in [0, 1, 2]:
#                     fg.axes[r, c].set_yticklabels([])
#     except:
#         fg.axes[0, 1].set_xlabel('pseeds')
#         if y == 'bias':
#             fg.axes[0, 0].set_ylabel(r"$bias=\frac{CC_{min}}{CC_{min}+CC_{maj}}$", fontsize=13)
#         else:
#             try:
#                 fg.axes[0, 0].set_ylabel(ymetric[y])
#             except:
#                 fg.axes[0, 0].set_ylabel(y)
#         pass
#
#     plt.subplots_adjust(hspace=0.05, wspace=0.05)
#
#     if fn is not None:
#         fg.savefig(fn, bbox_inches='tight')
#         print("{} saved!".format(fn))
#     plt.show()
#     plt.close()
#
#     def _plot_bars(x, y, **kwargs):
#         width = 0.2
#
#         ax = plt.gca()
#         data = kwargs.pop("data")
#         g = data.groupby(['B', 'H', 'pseeds', 'sampling'])  # 'N',m
#         means = g[y].mean().reset_index()
#         errors = g[y].std()
#         logy = kwargs.pop("logy")
#
#         span = {'nodes': width * 0, 'nedges': width * 1, 'degree': width * 2, 'partialcrawls': width * 3}
#         sampling = data.sampling.unique()[0].replace("_", "").replace("\\", "")
#         span = span[sampling]
#
#         xticks = np.arange(1, means.pseeds.nunique() + 1, 1)
#         ax.bar(xticks + span, means[y], width, yerr=errors, bottom=0, **kwargs)
#
#         ax.set_xticks(xticks + width)
#         ax.set_xticklabels(sorted(data.pseeds.astype(np.int).unique()))
#
#         if logy:
#             ax.set_yscale('symlog')
#
#     def _plot_lines(x, y, **kwargs):
#
#         ax = plt.gca()
#         data = kwargs.pop("data")
#         g = data.groupby(['B', 'H', 'pseeds', 'sampling'])  # 'N',m
#         means = g[y].mean().reset_index()
#
#         if 'errors' in kwargs:
#             if not kwargs.pop('errors'):
#                 ax.plot(means.pseeds, means[y], **kwargs)
#                 return
#
#         errors = g[y].std()
#         ax.errorbar(means.pseeds, means[y], yerr=errors, **kwargs)

