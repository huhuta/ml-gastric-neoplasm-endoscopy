import os
from glob import glob
import sys

from setuptools import setup, find_packages

required_packages = [
    'docopt',
    'matplotlib',
    'scipy',
    'docopt',
    'sklearn',
    'pandas',
]

setup(
    name="ml",
    version='0.0.1',
    description="",
    packages=find_packages('src'),
    package_dir={
        '': 'src'},
    py_modules=[
        os.path.splitext(
            os.path.basename(path))[0] for path in glob('src/*.py')],
    author="huhuta",
    url='https://github.com/huhuta/ml-gastric-neoplasm-endoscopy',
    license="MIT",
    install_requires=required_packages,
    entry_points={
        'console_scripts': ['mlcli=ml.cli.main:main'],
    })
