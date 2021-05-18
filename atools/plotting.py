#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions for base plots.

Author: Andrew Tarzia

Date Created: 25 Mar 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def colors_i_like(palette=None):
    """
    A list of colours I like to choose from.

    palette options:
        None
        Base
        IBM
        Wong
        Tol
        CB_pairs

    """

    if palette is None:
        return [
            '#FA7268', '#F8A72A', '#DAF7A6', '#900C3F', '#6BADB0',
            '#DB869D', '#F6D973', 'mediumvioletred',
            'skyblue', 'gold', 'palegreen', 'coral',
        ]
    elif palette == 'Base':
        return [
            '#D81B60', '#1E88E5', '#FFC107', '#FE6100', '#004D40'
        ]
    elif palette == 'IBM':
        return [
            '#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'
        ]
    elif palette == 'Wong':
        return [
            '#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
            '#0072B2', '#D55E00', '#CC79A7'
        ]
    elif palette == 'Tol':
        return [
            '#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77',
            '#CC6677', '#AA4499', '#882255',
        ]
    elif palette == 'CB_pairs':
        return [
            '#FFC20A', '#0C7BDC', '#994F00', '#006CD1', '#E1BE6A',
            '#40B0A6', '#E66100', '#5D3A9B', '#1AFF1A', '#4B0092',
            '#FEFE62', '#D35FB7', '#005AB5', '#DC3220', '#1A85FF',
            '#D41159',
        ]


def markers_i_like():
    """
    A list of markers I like to choose from.

    """
    return [
        'o', 'X', 's', 'P', 'h', 'D', 'd', 'p', 'v', '^', '<', '>'
    ]


def parity_plot(X, Y, xtitle, ytitle, lim, c=None, marker=None):
    """
    Make parity plot.

    """
    if c is None:
        C = colors_i_like()[2]
    else:
        C = c
    if marker is None:
        M = 'o'
    else:
        M = marker
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X, Y, c=C, edgecolors='k',
               marker=M, alpha=1.0, s=80)
    ax.plot(np.linspace(min(lim) - 1, max(lim) + 1, 2),
            np.linspace(min(lim) - 1, max(lim) + 1, 2),
            c='k', alpha=0.4)
    # Set number of ticks for x-axis
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel(xtitle, fontsize=16)
    ax.set_ylabel(ytitle, fontsize=16)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    return fig, ax


def scatter_plot(
    X,
    Y,
    xtitle,
    ytitle,
    xlim,
    ylim,
    title=None,
    c='firebrick',
    edgecolors='k',
    marker='o',
    alpha=1.0,
    s=80,
    Z=None,
    cmap=None
):
    """
    Make scatter plot.

    Parameters
    ----------
    X : :class:``

    Y : :class:``

    xtitle : :class:``

    ytitle : :class:``

    xlim : :class:``

    ylim : :class:``

    title : :class:``

    c : :class:``

    edgecolors : :class:``

    marker : :class:``

    alpha : :class:``

    s : :class:``

    Z : :class:``

    cmap : :class:`dict`
        Dictionary containing information for cmap.
        Example:
        {
            'mid_point': 0.5,
            'cmap': cm.Purples,
            'ticks': [0, .50, 1.00],
            'labels': [
                '0',
                '20',
                '40'
            ],
            'cmap_label': 'flex',
        }

        This requries that the `Z` argument is on the range of 0 to 1.
        Use 'labels' to define the relationship between cmap and
        Z value.


    Returns
    -------

    fig

    ax

    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if cmap is None and Z is None:
        ax.scatter(
            X, Y,
            c=c,
            edgecolors=edgecolors,
            marker=marker,
            alpha=alpha,
            s=s
        )
    else:
        cmp = define_plot_cmap(
            fig, ax,
            mid_point=cmap['mid_point'],
            cmap=cmap['cmap'],
            ticks=cmap['ticks'],
            labels=cmap['labels'],
            cmap_label=cmap['cmap_label']
        )
        ax.scatter(
            X, Y,
            c=cmp(Z),
            edgecolors=edgecolors,
            marker=marker,
            alpha=alpha,
            s=s
        )
    # Set number of ticks for x-axis
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel(xtitle, fontsize=16)
    ax.set_ylabel(ytitle, fontsize=16)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title, fontsize=16)
    return fig, ax


def histogram_2D(
    X_data,
    Y_data,
    x_bins,
    y_bins,
    xtitle,
    ytitle,
    xlim,
    ylim,
    cmap,
    density=False
):
    """
    2D-histogram.

    """

    norm = colors.Normalize()

    fig, ax = plt.subplots(figsize=(8, 5))

    hist = ax.hist2d(
        X_data,
        Y_data,
        bins=[x_bins, y_bins],
        range=[xlim, ylim],
        density=density,
        norm=norm,
        cmap=cmap
    )

    # Set number of ticks for x-axis
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel(xtitle, fontsize=16)
    ax.set_ylabel(ytitle, fontsize=16)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.ax.set_ylabel('count', fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    return fig, ax


def histogram_plot_N(
    Y,
    X_range,
    width,
    alpha,
    color,
    edgecolor,
    xtitle,
    labels=None,
    density=False,
    N=1
):
    """
    Make histogram plot with 1 distribution.

    """

    fig, ax = plt.subplots(figsize=(8, 5))
    X_bins = np.arange(X_range[0], X_range[1], width)
    if N == 1:
        hist, bin_edges = np.histogram(
            a=Y,
            bins=X_bins,
            density=density
        )
        ax.bar(
            bin_edges[:-1],
            hist,
            align='edge',
            alpha=alpha,
            width=width,
            color=color,
            edgecolor=edgecolor
        )
    else:
        for i_ in range(N):
            if type(color) is not list or len(Y) != N:
                raise ValueError(
                    'Make sure color and Y are of length N'
                )
            hist, bin_edges = np.histogram(
                a=Y[i_],
                bins=X_bins,
                density=density
            )
            if labels[i_] is None:
                label = ''
            else:
                label = labels[i_]
            ax.bar(
                bin_edges[:-1],
                hist,
                align='edge',
                alpha=alpha[i_],
                width=width,
                color=color[i_],
                edgecolor=edgecolor,
                label=label
            )
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel(xtitle, fontsize=16)
    if density is False:
        ax.set_ylabel('count', fontsize=16)
    elif density is True:
        ax.set_ylabel('frequency', fontsize=16)
    ax.set_xlim(X_range)
    if N > 1 and labels[0] is not None:
        ax.legend(fontsize=16)
    return fig, ax


def histogram_w_scatters_plot(
    X,
    X_range,
    X_width,
    X_title,
    Y,
    Y_range,
    Y_width,
    Y_title,
    scat_alpha=1.0,
    s=80,
    hist_alpha=1.0,
    color='k',
    edgecolor='k',
    marker='o',
    density=False,
):
    """
    Make scatter plot histogram plot distribution on both axes.

    """

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    X_bins = np.arange(X_range[0], X_range[1], X_width)
    Y_bins = np.arange(Y_range[0], Y_range[1], Y_width)

    fig = plt.figure(figsize=(8, 8))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(
        direction='in',
        top=True,
        right=True,
        labelsize=16,
    )
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(
        direction='in',
        labelbottom=False,
        labelsize=16,
    )
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(
        direction='in',
        labelleft=False,
        labelsize=16,
    )

    ax_scatter.scatter(
        X, Y,
        c=color,
        edgecolors=edgecolor,
        marker=marker,
        alpha=scat_alpha,
        s=s
    )
    ax_scatter.set_xlabel(X_title, fontsize=16)
    ax_scatter.set_ylabel(Y_title, fontsize=16)
    ax_scatter.set_xlim(X_range)
    ax_scatter.set_ylim(Y_range)

    ax_histx.hist(
        X,
        bins=X_bins,
        density=density,
        alpha=hist_alpha,
        color=color,
        edgecolor=edgecolor,
    )
    ax_histy.hist(
        Y,
        bins=Y_bins,
        orientation='horizontal',
        density=density,
        alpha=hist_alpha,
        color=color,
        edgecolor=edgecolor,
    )
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    return fig


def flat_line(ax, x, y, w=0, C='k', m='x'):
    ax.plot([x - w, x, x + w], [y, y, y], c=C)
    ax.scatter(x, y, marker=m, c=C)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0,
                    name='shiftedcmap'):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    From Stack Exchange:
        https://stackoverflow.com/questions/7404116/
        defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap


def define_plot_cmap(fig, ax, mid_point, cmap, ticks, labels,
                     cmap_label):
    """
    Define cmap shifted to midpoint and plot colourbar

    """
    new_cmap = shiftedColorMap(
        cmap,
        midpoint=mid_point,
        name='shifted'
    )
    X = np.linspace(0, 1, 256)
    cax = ax.scatter(-X-100, -X-100, c=X, cmap=new_cmap)
    cbar = fig.colorbar(cax, ticks=ticks, spacing='proportional')
    cbar.ax.set_yticklabels(labels, fontsize=16)
    cbar.set_label(cmap_label, fontsize=16)
    return new_cmap
