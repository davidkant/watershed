from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

requirements = [
    'numpy',
    'sklearn',
    'skimage',
    'scipy',
]

setup(
    name='watershed',
    version='0.0.1',
    description='Waterhsed segmentation',
    long_description=readme,
    author='David Kant',
    author_email='dkant@ucsc.edu',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=requirements
)
