from distutils.core import setup

with open("README_pypi.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='physDBD',
    version='0.2.0',
    packages=['physDBD','physDBD/gauss','physDBD/pca'],
    author='Oliver K. Ernst',
    author_email='oernst@ucsd.edu',
    url='https://github.com/smrfeld/phys_dbd',
    python_requires='>=3.7',
    description="Physics-based modeling of reaction networks with TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown"
    )