from setuptools import setup, find_packages
#from version import find_version
from codecs import open
from os import path
import re

#here = path.abspath(path.dirname(__file__))



VERSION_PATH = "fast_feature/VERSION"
with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()


setup(
        name = 'fast_feature',
        author = 'Ahmad Zaenal',
        description = 'Fast and Efficient Feature engineering with Polars based dataframe',
        long_description = 'Fast and Efficient Feature engineering with Polars based dataframe',
        license = 'MIT',
        project_urls = {'Github': 'https://github.com/zaenalium/fast_feature', 'Documentation': 'https://fast_feature.readthedocs.io/en/latest/'},
        include_package_data=True,
        version=__version__,
        packages =  ['fast_feature'], #find_packages(),
        author_email='ahmadzaenal125@gmail.com',
        keywords='feature engineering, encoding, discretisation',  # Optional
        install_requires=['numpy','pandas>=0.25.0','matplotlib','scikit-learn>=0.19.1', 'statsmodels', 'patsy'],  # Optional
     )