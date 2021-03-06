from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='proptest',
      packages=[package for package in find_packages()
                if package.startswith('lintest')],
      install_requires=[
              'numpy',
              'matplotlib',
              'jupyter'],
      description="Test if a real function is (almost) linear by querying",
      author="Matteo Papini",
      url='https://github.com/T3p/linearity-test',
      author_email="matteo.papini@polimi.it",
      version="0.1")
