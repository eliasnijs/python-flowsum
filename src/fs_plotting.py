from math import pi

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def fs_plot_star_chart(ax, max_weight, angles, values, colors, legend):
    """
    Plot a star (radar/spider) chart on a given matplotlib Axes.

    Parameters:
    ax (matplotlib.axes.Axes): The Axes object to draw the star chart on.
    max_weight (float): The maximum value for the radial axis, defining the
                        outermost circle of the star chart.
    angles (list of float): The angles in radians at which to place each axis of the
                            star chart. The last angle should be the same as the
                            first to close the plot.
    values (list of float): The values to plot for each axis. The order of the
                            values should correspond to the order of the angles.
    colors (list of str): The colors to use for each axis. The order of the colors
                          should correspond to the order of the angles and values.
    legend (list): A list to which the filled areas of the plot will be appended for
                   later use in a legend.

    This function creates a star chart where each axis represents a different
    category, and the length of the axis represents the value for that category. The
    axes are arranged in a circular fashion, and consecutive axes are connected to
    each other, forming a star shape.

    The chart is drawn on the given Axes object, and the filled areas of the plot
    are added to the given legend list.

    The grid and the border (spine) of the polar plot are semi-transparent to
    improve readability.
    """
    ax.set_rorigin(0)
    ax.set_ylim(0, max_weight)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_facecolor("white")
    ax.grid(alpha=0.25)
    ax.spines["polar"].set_alpha(0.25)
    for k in range(len(angles) - 1):
        ax.plot(
            [0, angles[k], angles[k + 1], 0],
            [0, values[k], values[k], 0],
            color=colors[k],
            zorder=10,
        )
        (fill,) = ax.fill(
            [angles[k], angles[k + 1], 0],
            [values[k], values[k], 0],
            color=colors[k],
            alpha=0.5,
            zorder=10,
        )
        legend.append(fill)


def fs_plot_som(fs, save=None, show=True, show_clusters=True):
    """
    Plots the flowsom SOM as a grid of star charts.
    Each neuron in the SOM is represented as a star chart, with the weights
    of the neuron serving as dimensions.
    """
    plt.clf()

    # Get the weights of the SOM
    weights = fs.som.get_weights()
    max_weight = np.max(weights)

    # Get the column names from the DataFrame to use as labels
    labels = fs.data.columns.values.tolist()

    # Define the properties for the radar chart
    num_vars = weights.shape[-1]
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    labels += labels[:1]

    fig, ax = plt.subplots(
        fs.som_param.shape[0],
        fs.som_param.shape[1],
        subplot_kw=dict(polar=True),
        figsize=(fs.som_param.shape[0] * 2, fs.som_param.shape[1] * 2),
    )
    fig.suptitle("FlowSOM: Grid")

    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    hcc_colors = plt.cm.tab20(np.linspace(0, 1, fs.hcc_param.n_clusters))

    legend_fills = []
    for i in range(fs.som_param.shape[0]):
        for j in range(fs.som_param.shape[1]):
            values = np.concatenate((weights[i, j], weights[i, j][:1]))
            i_cluster = fs.hcc[i * fs.som_param.shape[0] + j]
            fs_plot_star_chart(
                ax[i, j], max_weight, angles, values, colors, legend_fills
            )

    if show_clusters:
        ax_hcc = fig.add_subplot(111)
        ax_hcc.set_facecolor("#00000000")
        ax_hcc.axis("off")
        for i, ax_rows in enumerate(ax):
            for j, ax_node in enumerate(ax_rows):
                pos = ax_node.get_position()
                i_cluster = fs.hcc[i * fs.som_param.shape[0] + j] - 1
                circle = plt.Rectangle(
                    (pos.x0 - 0.005, pos.y0 - 0.005),
                    pos.width + 0.01,
                    pos.height + 0.01,
                    # (pos.x0 + pos.width / 2.0, pos.y0 + pos.height / 2.0),
                    # min(pos.width, pos.height) / 2.0,
                    edgecolor=(*hcc_colors[i_cluster][:3], 0.4),
                    facecolor=(*hcc_colors[i_cluster][:3], 0.15),
                    zorder=20,
                    transform=fig.transFigure,
                )
                ax_hcc.add_artist(circle)

    legend = fig.legend(
        legend_fills, labels[:-1], loc="upper right", title="Startcharts Legend"
    )
    for line in legend.get_lines():
        line.set_linewidth(5)

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)


def fs_plot_mst(fs, save=None, show=True, show_clusters=True):
    plt.clf()

    weights = fs.som.get_weights()
    weights = weights.reshape(-1, weights.shape[-1])
    max_weight = np.max(weights)

    labels = fs.data.columns.values.tolist()

    num_vars = weights.shape[-1]
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    labels += labels[:1]

    mst = fs.mst.toarray()
    G = nx.from_numpy_array(mst)
    pos = nx.kamada_kawai_layout(G)

    min_x = min(x for x, _ in pos.values())
    max_x = max(x for x, _ in pos.values())
    min_y = min(y for _, y in pos.values())
    max_y = max(y for _, y in pos.values())

    scale = 0.9
    offset = (1 - scale) / 2
    pos = {
        node: (
            scale * (x - min_x) / (max_x - min_x) + offset,
            scale * (y - min_y) / (max_y - min_y) + offset,
        )
        for node, (x, y) in pos.items()
    }

    gs = gridspec.GridSpec(1000, 1000)
    fig = plt.figure(figsize=(25, 25))
    fig.suptitle("FlowSOM: MST")

    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    hcc_colors = plt.cm.tab20(np.linspace(0, 1, fs.hcc_param.n_clusters))

    legend_fills = []
    for i_node, (x, y) in pos.items():
        grid_x = int(x * 1000)
        grid_y = int(y * 1000)
        ax = plt.subplot(
            gs[grid_y - 15 : grid_y + 15, grid_x - 15 : grid_x + 15], polar=True
        )
        ax.set_autoscale_on(False)
        values = np.concatenate((weights[i_node], weights[i_node][:1]))
        fs_plot_star_chart(ax, max_weight, angles, values, colors, legend_fills)

    if show_clusters:
        ax_hcc = fig.add_subplot(111)
        ax_hcc.set_facecolor("#00000000")
        ax_hcc.axis("off")
        for node, (x, y) in pos.items():
            i_cluster = fs.hcc[node] - 1
            circle = plt.Circle(
                (x, 1 - y),
                0.02,
                edgecolor=(*hcc_colors[i_cluster][:3], 0.4),
                facecolor=(*hcc_colors[i_cluster][:3], 0.15),
                zorder=20,
            )
            ax_hcc.add_artist(circle)

    ax_edges = fig.add_subplot(111)
    ax_edges.set_facecolor("#00000000")
    ax_edges.set_xticklabels([])
    ax_edges.set_yticklabels([])
    ax_edges.axis("off")
    for edge in G.edges:
        node1, node2 = edge
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        ax_edges.plot(
            [x1, x2],
            [1 - y1, 1 - y2],
            color="k",
            linewidth=0.5,
            transform=plt.gca().transAxes,
        )

    legend = fig.legend(
        legend_fills, labels[:-1], loc="upper right", title="Startcharts Legend"
    )
    for line in legend.get_lines():
        line.set_linewidth(5)

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)


def fs_plot_feature_planes(fs, save=None, show=True):
    plt.clf()
    weights = fs.som.get_weights()
    for i, f in enumerate(list(fs.data.columns)):
        ax = plt.subplot(4, 4, i + 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off")
        plt.title(f)
        plt.pcolor(weights[:, :, i].T, cmap="plasma")
    plt.tight_layout()
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
