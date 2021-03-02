""" Setup file for xycmap. """
from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="xycmap",
    version="1.0.1",
    description="Bivariate colormap solutions",
    keywords="visualization, colormap, color, bivariate, two-dimensional",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rbjansen/xycmap",
    author="Remco Bastiaan Jansen",
    author_email="r.b.jansen.uu@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    packages=["xycmap"],
    install_requires=["pandas", "numpy", "scipy", "matplotlib"]
)
