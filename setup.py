#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from pip._internal.req import parse_requirements

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

install_reqs = parse_requirements("requirements.txt",session="hack")
requirements = [ir.requirement for ir in install_reqs]


test_requirements = ['pytest>=3', ]

setup(
    name='ntrfc',
    author="Malte Nyhuis",
    author_email='nyhuis@tfd.uni-hannover.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Numerical Test Rig for Cascades. A workflows-library for cfd-analysis of cascade-flows",
    entry_points={
        'console_scripts': [
            'ntrfc=ntrfc.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ntrfc',
    packages=find_packages(include=['ntrfc', 'ntrfc.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MaNyh/ntrfc',
    version='0.1.0',
    zip_safe=False,
)
