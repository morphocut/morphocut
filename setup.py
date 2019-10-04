from setuptools import find_packages, setup

import versioneer

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name='morphocut',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Simon-Martin Schroeder',
    author_email="martin.schroeder@nerdluecht.de",
    description="Image processing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/morphocut/morphocut",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scikit-image',
        'pandas',
        'tqdm',
        'scipy',
    ],
    python_requires='>=3.5',
    extras_require={
        'tests': ['pytest'],
        'docs':
            [
                'sphinx >= 1.4',
                'sphinx_rtd_theme',
                'sphinxcontrib-programoutput',
            ]
    },
    entry_points={},
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ],
)
