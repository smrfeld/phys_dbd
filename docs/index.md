# Physics-based dynamic PCA for modeling stochastic reaction networks with TensorFlow

This is the source repo. for the `physDBD` Python package. It allows the creation of physics-based machine learning models in `TensorFlow` for modeling stochastic reaction networks.

<img src="figures/fig_1.png" alt="drawing" width="800"/>

## API Documentation

[See the API documentation here.](modules.rst)

## Quickstart

1. Install:
    ```
    pip install physDBD
    ```
2. See the [example notebook](example/).

3. Read the [documentation](...).

## About

This repo. implements a TensorFlow package for modeling stochastic reaction networks with a dynamic PCA model. Please see [this] paper for technical details:
```
XXX
```
The original implementation in the paper is written in Mathematica and can be found [here](https://github.com/smrfeld/physics-based-ml-reaction-networks). The Python package developed here translates these methods to `TensorFlow`.

The only current supported probability distribution is the Gaussian distribution defined by PCA; more general Gaussian distributions are a work in progress.

## Requirements

* `TensorFlow 2.5.0` or later. *Note: later versions not tested.*
* `Python 3.7.4` or later.

## Installation

Use `pip`:
```
pip install physDBD
```
Alternatively, clone this repo. and use the provided `setup.py`:
```
python setup.py install
```

## Example

See the notebook in the [example](example/) directory.

## Tests

Tests are run using `pytest` and are located in [tests](tests/).

## Citing

Please cite the following paper:
```
XXX
```