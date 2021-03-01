""" Tools for bivariate colormaps. """

import warnings
import numpy as np
import pandas as pd
from pandas.api import types
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage.interpolation import zoom


def custom_xycmap(corner_colors=("grey", "green", "blue", "red"), n=(5, 5)):
    """Creates a colormap according to specified colors and n categories.

    Args:
        corner_colors: List or Tuple of four matplotlib colors. See recognized
            formats at https://matplotlib.org/stable/api/colors_api.html.
        n: Tuple containing the number of columns and rows (x, y).

    Returns:
        Custom two-dimensional colormap in np.ndarray.

    Raises:
        ValueError: If less than two columns or rows are passed.
    """
    xn, yn = n
    if xn < 2 or yn < 2:
        raise ValueError("Expected n >= 2 categories.")

    color_array = np.array(
        [
            [
                list(colors.to_rgba(corner_colors[0])),
                list(colors.to_rgba(corner_colors[1])),
            ],
            [
                list(colors.to_rgba(corner_colors[2])),
                list(colors.to_rgba(corner_colors[3])),
            ],
        ],
    )
    zoom_factor_x = xn / 2  # Divide by the original two categories.
    zoom_factor_y = yn / 2
    zcolors = zoom(color_array, (zoom_factor_y, zoom_factor_x, 1), order=1)

    return zcolors


def mean_xycmap(xcmap=plt.cm.Greens, ycmap=plt.cm.Reds, n=(5, 5)):
    """Creates a colormap according to two colormaps and n categories.

    Args:
        xcmap: Matplotlib colormap along the x-axis. Defaults to Greens.
        ycmap: Matplotlib colormap along the y-axis. Defaults to Reds.
        n: Tuple containing the number of columns and rows (x, y).

    Returns:
        Custom two-dimensional colormap in np.ndarray.

    Raises:
        ValueError: If less than two columns or rows are passed.
    """
    xn, yn = n
    if xn < 2 or yn < 2:
        raise ValueError("Expected n >= 2 categories.")

    sy, sx = np.mgrid[0:yn, 0:xn]

    # Rescale the mock series into the colormap range (0, 255).
    xvals = np.array(255 * (sx - sx.min()) / (sx.max() - sx.min()), dtype=int)
    yvals = np.array(255 * (sy - sy.min()) / (sy.max() - sy.min()), dtype=int)

    xcolors = xcmap(xvals)
    ycolors = ycmap(yvals)

    # Take the mean of the two colormaps.
    zcolors = np.sum([xcolors, ycolors], axis=0) / 2

    return zcolors


def bivariate_color(
    sx, sy, cmap, xlims=None, ylims=None, xbins=None, ybins=None
):
    """Creates a color series for a combination of two series.

    Args:
        sx: Initial pd.Series to plot.
        sy: Secondary pd.Series to plot.
        cmap: A two-dimensional colormap in np.ndarray.
        xlims: Optional tuple specifying limits to the x-axis.
        ylims: Optional tuple specifying limits to the y-axis.
        xbins: Optional iterable containing bins for the x-axis.
        ybins: Optional iterable containing bins for the y.axis.

    Returns:
        pd.Series of assigned colors per cmap provided.

    Raises:
        TypeError: If sx or sy is not numeric or categorical.
        ValueError: If the colormap axis length does not match the length of
            n categories in supplied categorical Series.
        RuntimeError: If limits are supplied for a categorical Series.
    """
    x_numeric = types.is_numeric_dtype(sx)
    y_numeric = types.is_numeric_dtype(sy)
    x_categorical = types.is_categorical_dtype(sx)
    y_categorical = types.is_categorical_dtype(sy)

    msg = (
        "The provided {s} is not numeric or categorical. If {s} contains "
        "categories, transform the series to (ordered) pd.Categorical first."
    )
    if not x_numeric and not x_categorical:
        raise TypeError(msg.format(s="sx"))
    if not y_numeric and not y_categorical:
        raise TypeError(msg.format(s="sy"))

    # If categorical, the number of categories have to equal the cmap shape.
    if x_categorical:
        if len(sx.categories) != cmap.shape[1]:
            raise ValueError(
                f"Length of x-axis colormap ({cmap.shape[1]}) does not match "
                f"the length of categories in sx ({len(sx.categories)}). "
                "Adjust the n of your cmap."
            )
    if y_categorical:
        if len(sy.categories) != cmap.shape[0]:
            raise ValueError(
                f"Length of x-axis colormap ({cmap.shape[0]}) does not match "
                f"the length of categories in sy ({len(sy.categories)}). "
                "Adjust the n of your cmap."
            )

    # If numeric, use min/max to mock a series for the bins.
    if x_numeric:
        xmin, xmax = (sx.min(), sx.max()) if xlims is None else xlims
        if xbins is None:
            _, xbins = pd.cut(
                pd.Series([xmin, xmax]), cmap.shape[1], retbins=True
            )
    else:
        if xlims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sx: the xticks of the "
                "cmap are indivisible. Instead, limit your data to the "
                "categories and adjust the n of cmap accordingly."
            )
        if xbins is not None:
            raise RuntimeError(
                "Cannot apply bins to a categorical sx: the xticks of the "
                "cmap are indivisible."
            )

    if y_numeric:
        ymin, ymax = (sy.min(), sy.max()) if ylims is None else ylims
        if ybins is None:
            _, ybins = pd.cut(
                pd.Series([ymin, ymax]), cmap.shape[0], retbins=True
            )
    else:
        if ylims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sy: the yticks of the "
                "cmap are indivisible. Instead, limit your data to the "
                "categories and adjust the n of cmap accordingly."
            )
        if ybins is not None:
            raise RuntimeError(
                "Cannot apply bins to a categorical sy: the yticks of the "
                "cmap are indivisible."
            )

    def _bin_value(x, bins):
        if bins.min() > x:
            return 0  # First index.
        if bins.max() < x:
            return len(bins[:-1]) - 1  # Last index.
        for i, v in enumerate(bins[:-1]):
            rangetest = v < x <= bins[i + 1]  # pd.cut right=True by default.
            if rangetest:
                return i
        return np.nan

    def _return_color(x, y, cmap):
        if np.isnan(x) or np.isnan(y):
            return (0.0, 0.0, 0.0, 0.0)  # Transparent white if one is np.nan.
        xidx = _bin_value(x, xbins) if x_numeric else x
        yidx = _bin_value(y, ybins) if y_numeric else y
        return tuple(cmap[yidx, xidx])

    sx = pd.Series(sx.codes) if x_categorical else sx
    sy = pd.Series(sy.codes) if y_categorical else sy

    df = pd.DataFrame([sx, sy]).T
    colors = df.apply(
        lambda g: _return_color(g[df.columns[0]], g[df.columns[1]], cmap),
        axis=1,
    )

    return colors


def bivariate_legend(
    ax,
    sx,
    sy,
    cmap,
    alpha=1,
    xlims=None,
    ylims=None,
    xlabels=None,
    ylabels=None,
):
    """Plots bivariate cmap onto an ax to use as a legend.

    Args:
        ax: Matplotlib ax to plot into.
        sx: Initial pd.Series to plot.
        sy: Secondary pd.Series to plot.
        cmap: A two-dimensional colormap in np.ndarray.
        alpha: Optional alpha (0-1) to pass to imshow.
        xlims: Optional tuple specifying limits to the x-axis, if numeric.
        ylims: Optional tuple specifying limits to the y-axis, if numeric.
        xlabels: Optional list of ordered labels for the bins along x.
        ylabels: Optional list of ordered labels for the bins along y.

    Returns:
        An ax containing the plotted cmap and relevant tick labels.

    Raises:
        TypeError: If sx or sy is not numeric or categorical.
        RuntimeError: If the length of labels is less than the number of ticks.
    """
    x_numeric = types.is_numeric_dtype(sx)
    y_numeric = types.is_numeric_dtype(sy)
    x_categorical = types.is_categorical_dtype(sx)
    y_categorical = types.is_categorical_dtype(sy)

    msg = (
        "The provided {s} is not numeric or categorical. If {s} contains "
        "categories, transform the series to (ordered) pd.Categorical first."
    )
    if not x_numeric and not x_categorical:
        raise TypeError(msg.format(s="sx"))
    if not y_numeric and not y_categorical:
        raise TypeError(msg.format(s="sy"))

    # Bin series for ticklabels if numeric, get categories if categorical.
    if x_numeric:
        xmin, xmax = (sx.min(), sx.max()) if xlims is None else xlims
        _, xbins = pd.cut(pd.Series([xmin, xmax]), cmap.shape[1], retbins=True)
        if xlabels is None:
            xlabels = [f"{np.round(i, 2)}" for i in xbins]
    else:
        if xlabels is None:
            xlabels = sx.categories
        if xlims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sx: the xticks of the "
                "cmap are indivisible. Instead, limit your data to the "
                "categories and adjust the n of cmap accordingly."
            )

    if y_numeric:
        ymin, ymax = (sy.min(), sy.max()) if ylims is None else ylims
        _, ybins = pd.cut(pd.Series([ymin, ymax]), cmap.shape[0], retbins=True)
        if ylabels is None:
            ylabels = [f"{np.round(i, 2)}" for i in ybins]
    else:
        if ylabels is None:
            ylabels = sy.categories
        if ylims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sy: the yticks of the "
                "cmap are indivisible. Instead, limit your data to the "
                "categories and adjust the n of cmap accordingly."
            )

    # Start building the plot here.
    ax.imshow(cmap, alpha=alpha, origin="lower")

    # Center ticks if categorical.
    if x_categorical:
        xticks = np.arange(0, cmap.shape[1], 1)
    else:
        xticks = np.arange(-0.5, cmap.shape[1], 1)
    ax.set_xticks(xticks)

    if y_categorical:
        yticks = np.arange(0, cmap.shape[0], 1)
    else:
        yticks = np.arange(-0.5, cmap.shape[0], 1)
    ax.set_yticks(yticks)

    # Check whether tick labels match the number of ticks. Cut if longer.
    if len(xlabels) > len(xticks):
        warnings.warn(
            f"More xlabels ({len(xlabels)}) than the number of xticks "
            f"({len(xticks)}). The labels were cut to match length."
        )
        xlabels = xlabels[: -(len(xlabels) - len(xticks))]
    if len(xlabels) < len(xticks):
        raise RuntimeError(
            f"Less xlabels ({len(xlabels)}) than the number of xticks "
            f"({len(xticks)}) on your colormap."
        )

    if len(ylabels) > len(yticks):
        warnings.warn(
            f"More ylabels ({len(ylabels)}) than the number of yticks "
            f"({len(yticks)}). The labels were cut to match length."
        )
        ylabels = ylabels[: -(len(ylabels) - len(yticks))]
    if len(ylabels) < len(yticks):
        raise RuntimeError(
            f"Less ylabels ({len(ylabels)}) than the number of yticks "
            f"({len(yticks)}) on your colormap."
        )

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    return ax
