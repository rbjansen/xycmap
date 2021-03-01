""" Unit tests of xycmap functions."""

import unittest
import numpy as np
import pandas as pd
import xycmap


class TestBivariateColormap(unittest.TestCase):
    """Unit tests of xycmap functions."""

    def test_dims_custom_xycmap(self):
        """Test for expected dimensions of n-zoomed cmap."""
        cmap = xycmap.custom_xycmap(n=(10, 10))
        self.assertEqual(cmap.shape, (10, 10, 4))

    def test_bivariate_color_picks_default(self):
        """Test for expected color indices."""
        cmap = xycmap.custom_xycmap(n=(2, 2))
        sx = pd.Series([1, 3, 5])
        sy = pd.Series([3, 9, 10])
        expected_colors = pd.Series(
            [
                tuple(cmap[0][0]),  # y, x
                tuple(cmap[1][0]),
                tuple(cmap[1][1]),
            ]
        )
        self.assertTrue(
            xycmap.bivariate_color(sx, sy, cmap).equals(expected_colors)
        )

    def test_bivariate_color_picks_outer_minmax(self):
        """Test for expected color indices if outer lims are applied."""
        cmap = xycmap.custom_xycmap(n=(2, 2))
        sx = pd.Series([3, 6, 10])  # 6 should be right-most if minned on 0.
        sy = pd.Series([3, 2, 10])
        expected_colors = pd.Series(
            [
                tuple(cmap[0][0]),
                tuple(cmap[0][1]),
                tuple(cmap[1][1]),
            ]
        )
        self.assertTrue(
            xycmap.bivariate_color(
                sx, sy, cmap=cmap, xlims=(0, 10), ylims=(0, 10)
            ).equals(expected_colors)
        )

    def test_bivariate_color_picks_inner_minmax(self):
        """Test for expected color indices if inner lims are applied."""
        cmap = xycmap.custom_xycmap(n=(5, 5))
        sx = pd.Series([3, 6, 10])
        sy = pd.Series([3, 2, 10])  # 3, 2 first index; 10 last index.
        expected_colors = pd.Series(
            [
                tuple(cmap[0][1]),
                tuple(cmap[0][2]),
                tuple(cmap[4][4]),
            ]
        )
        self.assertTrue(
            xycmap.bivariate_color(
                sx, sy, cmap=cmap, xlims=(0, 10), ylims=(5, 8)
            ).equals(expected_colors)
        )

    def test_edge_bin_cases(self):
        """Test for expected color indices on edge cases."""
        cmap = xycmap.mean_xycmap(n=(5, 5))
        sx = pd.Series([3.9, 6, 8.1])
        sy = pd.Series([3.9, 6, 8.1])
        expected_colors = pd.Series(
            [
                tuple(cmap[1][1]),
                tuple(cmap[2][2]),
                tuple(cmap[4][4]),
            ]
        )
        colors = xycmap.bivariate_color(
            sx=sx, sy=sy, cmap=cmap, ylims=(0, 10), xlims=(0, 10)
        )
        self.assertTrue(colors.equals(expected_colors))

    def test_bivariate_color_picks_categorical(self):
        """Test for expected color indices if data is categorical."""
        sx = pd.Categorical(
            ["low", "mid", "high"],
            categories=["low", "mid", "high"],
            ordered=True,
        )
        sy = pd.Series([0, 5, 10])
        cmap = xycmap.mean_xycmap(n=(len(sx.categories), 5))
        expected_colors = pd.Series(
            [
                tuple(cmap[0][0]),
                tuple(cmap[2][1]),
                tuple(cmap[4][2]),
            ]
        )
        colors = xycmap.bivariate_color(sx=sx, sy=sy, cmap=cmap, ylims=(0, 10))
        self.assertTrue(colors.equals(expected_colors))

    def test_missing_color_picks(self):
        """Test for expected colors if data contains missing values."""
        cmap = xycmap.custom_xycmap(n=(2, 2))
        sx = pd.Series([1, np.nan, 5, 9])
        sy = pd.Series([3, 9, 10, np.nan])
        expected_colors = pd.Series(
            [
                tuple(cmap[0][0]),
                (0.0, 0.0, 0.0, 0.0),
                tuple(cmap[1][0]),
                (0.0, 0.0, 0.0, 0.0),
            ]
        )
        self.assertTrue(
            xycmap.bivariate_color(sx, sy, cmap).equals(expected_colors)
        )


if __name__ == "__main__":
    unittest.main()
