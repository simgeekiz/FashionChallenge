from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['sklearn', 'numpy>=1.13.3', 'pandas', 'keras', 'tensorflow', 'h5py', 'pillow', 'google-gax<=0.13.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='iMaterialist Challenge (Fashion) model on Cloud ML Engine'
)
