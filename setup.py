from setuptools import setup, find_packages
import os

# parse requirements.txt to find dependencies
library_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{library_folder}/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='kinetic-analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=install_requires,
    author= ['Duncan Muir', 'Nicholas Freitas'],
    author_email="duncan.muir@ucsf.edu",
#    description='what does this do?',
    long_description='https://github.com/pinneylab/kinetic_analysis',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)