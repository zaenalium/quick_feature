from setuptools import setup, find_packages
#from version import find_version
from codecs import open
from os import path
import re

#here = path.abspath(path.dirname(__file__))



VERSION_PATH = "quick_feature/VERSION"
with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()


setup(
        name = 'quick_feature',
        author = 'Ahmad Zaenal',
        description = 'Fast and Efficient Feature engineering with Polars based dataframe',
        long_description = 'Fast and Efficient Feature engineering with Polars based dataframe',
        license = 'MIT',
        project_urls = {'Github': 'https://github.com/zaenalium/quick_feature', 'Documentation': 'https://quick_feature.readthedocs.io/en/latest/'},
        include_package_data=True,
        version=__version__,
        packages =  ['quick_feature'], #find_packages(),
        author_email='ahmadzaenal125@gmail.com',
        keywords='feature engineering, encoding, discretisation',  # Optional
        install_requires=['numpy>=1.18.2', 'pandas>=2.2.0', 'scikit-learn>=1.4.0', 'scipy>=1.4.1',
                           'statsmodels>=0.11.1', 'polars>=1.8.2', 'numpydoc', 'sphinx', 'sphinx_autodoc_typehints', 'sphinx_rtd_theme'],  # Optional
     )