import sys
from setuptools import setup

try:
    with open('README.rst') as readme:
        README = readme.read()
except IOError:
    README = ''

setup(
    name='kingsnake',
    version='0.0.0',
    packages=['kingsnake'],
    install_requires=[
        'scrapy',
        'xmltodict',
        'pymongo',
        'scrapy-mongodb',
    ],
    author='Dirley Rodrigues',
    author_email='dirleyrls@gmail.com',
    long_description=README,
)
