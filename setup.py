from setuptools import find_packages, setup

import versioneer

with open("README.rst", "r") as fp:
    LONG_DESCRIPTION = fp.read()

setup(
    name="morphocut",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Simon-Martin Schroeder",
    author_email="martin.schroeder@nerdluecht.de",
    description="Image processing pipeline",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    url="https://github.com/morphocut/morphocut",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scikit-image>=0.19",
        "pandas",
        "tqdm",
        "scipy",
        "deprecated",
        # Required to display exception notes while Python<3.11 is supported.
        "exceptiongroup",
    ],
    python_requires=">=3.9",
    extras_require={
        "tests": [
            # Pytest
            "pytest",
            "pytest-cov",
            "timer-cm",
            # Coverage
            "codecov",
            # Optional dependencies
            "parse",
            "matplotlib",  # For FontManager in scalebar
            "h5py",
            "pydot",
        ],
        "docs": [
            "sphinx~=7.3",
            "sphinx_rtd_theme",
            "sphinxcontrib-programoutput",
            "sphinx-autodoc-typehints>=1.10.0",
            # See https://github.com/morphocut/morphocut/issues/89
            "Jinja2<3.1",
        ],
    },
    entry_points={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ],
)
