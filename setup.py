from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='lunar-crater-age-classifier',
      version="0.0.1",
      description="API for predicting lunar crater age using a classifier model",
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False,
      install_requires=requirements)
