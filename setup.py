from setuptools import setup, find_packages

setup(
    name='kinetic-analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'scikit-learn'
    ],
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