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
    install_requires=["numpy", "scikit-image>=0.16.0", "pandas", "tqdm", "scipy"],
    python_requires=">=3.6",
    extras_require={
        "tests": [
            # Pytest
            "pytest",
            "pytest-cov",
            "timer-cm",
            # Optional dependencies
            "parse",
        ],
        "docs": [
            "sphinx >= 1.4",
            "sphinx_rtd_theme",
            "sphinxcontrib-programoutput",
            "sphinx-autodoc-typehints>=1.10.0",
        ],
    },
    entry_points={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ],
)
