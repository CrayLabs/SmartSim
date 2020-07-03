from setuptools import setup, find_packages

setup(
    name='SmartSim',
    version='0.1.0',
    author=['Sam Partee', "Matt Ellis", "Andrew Shao", "Audrey Pratt", "Alex Heye"],
    author_email="spartee@hpe.com",
    packages=find_packages(),
    long_description=open('README.md').read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7'
)