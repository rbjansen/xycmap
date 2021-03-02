# xycmap
> Bivariate colormap solutions.

This package makes it easy to create custom two-dimensional colormaps, apply them to your series, and add bivariate color legends to your plots.

![example](https://user-images.githubusercontent.com/31345940/109506935-7b7ad100-7a9e-11eb-868f-899804e05bf6.png)

## Installation

`pip install xycmap`

## Usage

Make a custom interpolated colormap by specifying four corner colors (see recognized formats [here](https://matplotlib.org/stable/api/colors_api.html)), and dimensions `n`:

```python
import xycmap
corner_colors = ("lightgrey", "green", "blue", "red")
n = (5, 5)  # x, y
cmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)
```

![custom_xycmap](https://user-images.githubusercontent.com/31345940/109507925-8c781200-7a9f-11eb-9a2d-32c19b07a1c0.png)

Or make a colormap by mixing two matplotlib colormaps, and specifying dimensions `n`:

```python
import matplotlib.pyplot as plt
xcmap = plt.cm.rainbow
ycmap = plt.cm.Greys
n = (5, 5)  # x, y
cmap = xycmap.mean_xycmap(xcmap=xcmap, ycmap=ycmap, n=n)
```

![mean_xycmap](https://user-images.githubusercontent.com/31345940/109420855-d647f600-79d4-11eb-8b3a-f50505fcc44a.png)

With that in place, apply the colormap to two series that are numeric or categorical:

```python
colors = xycmap.bivariate_color(sx=sx, sy=sy, cmap=cmap)
```

Note that you can apply limits to the axes, as well as pass custom bins for the axes (if numerical). See the docstring for details.

Then simply pass `colors` to your plot. To add a legend, create a new ax and run `bivariate_legend()` into the ax with the same parameters as `bivariate_color()`, e.g.:

```python
cax = fig.add_axes([1, 0.25, 0.5, 0.5])
cax = xycmap.bivariate_legend(ax=cax, sx=sx, sy=sy, cmap=cmap)
```

## Meta

Remco Bastiaan Jansen – r.b.jansen.uu@gmail.com - [https://github.com/rbjansen](https://github.com/rbjansen)

Distributed under the MIT license. See `LICENSE` for more information.
