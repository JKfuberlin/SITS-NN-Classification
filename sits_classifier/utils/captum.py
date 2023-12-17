### all timeseries visualization convenience functions are adopted from:
### https://github.com/pytorch/captum/blob/010f76dc36a68d62cc7ad59fc6582cfaa1d19008/captum/attr/_utils/visualization.py#L3
### only slight changes are made, credits and many thanks to "smaeland"

from enum import Enum
from matplotlib.pyplot import axis, figure
from matplotlib import cm, colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import ndarray
from typing import Any, Iterable, List, Optional, Tuple, Union

class TimeseriesVisualizationMethod_cs(Enum):
    overlay_individual = 1
    overlay_combined = 2
    colored_graph = 3

class VisualizeSign_cs(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4

def _cumulative_sum_threshold_cs(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def _normalize_scale_cs(attr: ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)
def _normalize_attr_cs(
    attr: ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign_cs[sign] == VisualizeSign_cs.all:
        threshold = _cumulative_sum_threshold_cs(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold_cs(attr_combined, 100 - outlier_perc)
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold_cs(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold_cs(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale_cs(attr_combined, threshold)

def visualize_timeseries_attr_cs(
    attr: ndarray,
    data: ndarray,
    x_values: Optional[ndarray] = None,
    method: str = "individual_channels",
    sign: str = "absolute_value",
    channel_labels: Optional[List[str]] = None,
    channels_last: bool = True,
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    outlier_perc: Union[int, float] = 2,
    cmap: Union[None, str] = None,
    alpha_overlay: float = 0.7,
    show_colorbar: bool = False,
    title: Union[None, str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
    **pyplot_kwargs,
):
    r"""
    Visualizes attribution for a given timeseries data by normalizing
    attribution values of the desired sign (positive, negative, absolute value,
    or all) and displaying them using the desired mode in a matplotlib figure.
    Args:
        attr (numpy.ndarray): Numpy array corresponding to attributions to be
                    visualized. Shape must be in the form (N, C) with channels
                    as last dimension, unless `channels_last` is set to True.
                    Shape must also match that of the timeseries data.
        data (numpy.ndarray): Numpy array corresponding to the original,
                    equidistant timeseries data. Shape must be in the form
                    (N, C) with channels as last dimension, unless
                    `channels_last` is set to true.
        x_values (numpy.ndarray, optional): Numpy array corresponding to the
                    points on the x-axis. Shape must be in the form (N, ). If
                    not provided, integers from 0 to N-1 are used.
                    Default: None
        method (str, optional): Chosen method for visualizing attributions
                    overlaid onto data. Supported options are:
                    1. `overlay_individual` - Plot each channel individually in
                        a separate panel, and overlay the attributions for each
                        channel as a heat map. The `alpha_overlay` parameter
                        controls the alpha of the heat map.
                    2. `overlay_combined` - Plot all channels in the same panel,
                        and overlay the average attributions as a heat map.
                    3. `colored_graph` - Plot each channel in a separate panel,
                        and color the graphs according to the attribution
                        values. Works best with color maps that does not contain
                        white or very bright colors.
                    Default: `overlay_individual`
        sign (str, optional): Chosen sign of attributions to visualize.
                    Supported options are:
                    1. `positive` - Displays only positive pixel attributions.
                    2. `absolute_value` - Displays absolute value of
                        attributions.
                    3. `negative` - Displays only negative pixel attributions.
                    4. `all` - Displays both positive and negative attribution
                        values.
                    Default: `absolute_value`
        channel_labels (list[str], optional): List of labels
                    corresponding to each channel in data.
                    Default: None
        channels_last (bool, optional): If True, data is expected to have
                    channels as the last dimension, i.e. (N, C). If False, data
                    is expected to have channels first, i.e. (C, N).
                    Default: True
        plt_fig_axis (tuple, optional): Tuple of matplotlib.pyplot.figure and axis
                    on which to visualize. If None is provided, then a new figure
                    and axis are created.
                    Default: None
        outlier_perc (float or int, optional): Top attribution values which
                    correspond to a total of outlier_perc percentage of the
                    total attribution are set to 1 and scaling is performed
                    using the minimum of these values. For sign=`all`, outliers
                    and scale value are computed using absolute value of
                    attributions.
                    Default: 2
        cmap (str, optional): String corresponding to desired colormap for
                    heatmap visualization. This defaults to "Reds" for negative
                    sign, "Blues" for absolute value, "Greens" for positive sign,
                    and a spectrum from red to green for all. Note that this
                    argument is only used for visualizations displaying heatmaps.
                    Default: None
        alpha_overlay (float, optional): Alpha to set for heatmap when using
                    `blended_heat_map` visualization mode, which overlays the
                    heat map over the greyscaled original image.
                    Default: 0.7
        show_colorbar (bool): Displays colorbar for heat map below
                    the visualization.
        title (str, optional): Title string for plot. If None, no title is
                    set.
                    Default: None
        fig_size (tuple, optional): Size of figure created.
                    Default: (6,6)
        use_pyplot (bool): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
                    Default: True.
        pyplot_kwargs: Keyword arguments forwarded to plt.plot, for example
                    `linewidth=3`, `color='black'`, etc
    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.
    """

    # Check input dimensions
    assert len(attr.shape) == 2, "Expected attr of shape (N, C), got {}".format(
        attr.shape
    )
    assert len(data.shape) == 2, "Expected data of shape (N, C), got {}".format(
        attr.shape
    )

    # Convert to channels-first
    if channels_last:
        attr = np.transpose(attr)
        data = np.transpose(data)

    num_channels = attr.shape[0]
    timeseries_length = attr.shape[1]

    if num_channels > timeseries_length:
        warnings.warn(
            "Number of channels ({}) greater than time series length ({}), "
            "please verify input format".format(num_channels, timeseries_length)
        )

    num_subplots = num_channels
    if (
        TimeseriesVisualizationMethod_cs[method]
        == TimeseriesVisualizationMethod_cs.overlay_combined
    ):
        num_subplots = 1
        attr = np.sum(attr, axis=0)  # Merge attributions across channels

    ### the next bit contains a small bug fix by myself
    if x_values is not None:
        if channels_last:
            assert (
                x_values.shape[0] == timeseries_length
            ), "x_values must have same length as data"
        else:
            assert (
                    x_values.shape[0] == num_channels
            ), "x_values must have same length as data"
    else:
        x_values = np.arange(timeseries_length)

    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(
                figsize=fig_size, nrows=num_subplots, sharex=True
            )
        else:
            plt_fig = Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots(nrows=num_subplots, sharex=True)

    if not isinstance(plt_axis, ndarray):
        plt_axis = np.array([plt_axis])

    norm_attr = _normalize_attr_cs(attr, sign, outlier_perc, reduction_axis=None)

    # Set default colormap and bounds based on sign.
    if VisualizeSign_cs[sign] == VisualizeSign_cs.all:
        default_cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white", "green"]
        )
        vmin, vmax = -1, 1
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.positive:
        default_cmap = "Greens"
        vmin, vmax = 0, 1
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.negative:
        default_cmap = "Reds"
        vmin, vmax = 0, 1
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.absolute_value:
        default_cmap = "Blues"
        vmin, vmax = 0, 1
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    cmap = cmap if cmap is not None else default_cmap
    cmap = cm.get_cmap(cmap)
    cm_norm = colors.Normalize(vmin, vmax)

    def _plot_attrs_as_axvspan(attr_vals, x_vals, ax):

        half_col_width = (x_values[1] - x_values[0]) / 2.0
        for icol, col_center in enumerate(x_vals):
            left = col_center - half_col_width
            right = col_center + half_col_width
            ax.axvspan(
                xmin=left,
                xmax=right,
                facecolor=(cmap(cm_norm(attr_vals[icol]))),
                edgecolor=None,
                alpha=alpha_overlay,
            )

    if (
        TimeseriesVisualizationMethod_cs[method]
        == TimeseriesVisualizationMethod_cs.overlay_individual
    ):

        for chan in range(num_channels):

            plt_axis[chan].plot(x_values, data[chan, :], **pyplot_kwargs)
            if channel_labels is not None:
                plt_axis[chan].set_ylabel(channel_labels[chan])

            _plot_attrs_as_axvspan(norm_attr[chan], x_values, plt_axis[chan])

        plt.subplots_adjust(hspace=0)

    elif (
        TimeseriesVisualizationMethod_cs[method]
        == TimeseriesVisualizationMethod_cs.overlay_combined
    ):

        # Dark colors are better in this case
        # green, re1, re2 and so on = cm.Dark2.colors # unpacking the tuple, then cols = (green, re1, re2, and so on...)
        cycler = plt.cycler("color", cm.Dark2.colors)
        plt_axis[0].set_prop_cycle(cycler)

        for chan in range(num_channels):
            label = channel_labels[chan] if channel_labels else None
            plt_axis[0].plot(x_values, data[chan, :], label=label, **pyplot_kwargs)

        _plot_attrs_as_axvspan(norm_attr, x_values, plt_axis[0])

        ### legend position changed because of bad experience with "best"
        plt_axis[0].legend(loc="upper left")

    elif (
        TimeseriesVisualizationMethod_cs[method]
        == TimeseriesVisualizationMethod_cs.colored_graph
    ):

        for chan in range(num_channels):

            points = np.array([x_values, data[chan, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, norm=cm_norm, **pyplot_kwargs)
            lc.set_array(norm_attr[chan, :])
            plt_axis[chan].add_collection(lc)
            plt_axis[chan].set_ylim(
                1.2 * np.min(data[chan, :]), 1.2 * np.max(data[chan, :])
            )
            if channel_labels is not None:
                plt_axis[chan].set_ylabel(channel_labels[chan])

        plt.subplots_adjust(hspace=0)

    else:
        raise AssertionError("Invalid visualization method: {}".format(method))

    plt.xlim([x_values[0], x_values[-1]])

    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis[-1])
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.4)
        colorbar_alpha = alpha_overlay
        if (
            TimeseriesVisualizationMethod_cs[method]
            == TimeseriesVisualizationMethod_cs.colored_graph
        ):
            colorbar_alpha = 1.0
        plt_fig.colorbar(
            cm.ScalarMappable(cm_norm, cmap),
            orientation="horizontal",
            cax=colorbar_axis,
            alpha=colorbar_alpha,
        )
    if title:
        plt_axis[0].set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis