.. physDBD documentation master file, created by
   sphinx-quickstart on Thu Jun 17 14:20:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Physics-based dynamic PCA models in TensorFlow
==============================================

.. image:: figures/fig_1.png
  :width: 800
  :alt: Reaction model image

This is the source repo. for the `physDBD Python package <https://github.com/smrfeld/phys_dbd>`_. 
It allows the creation of physics-based machine learning models in `TensorFlow` for modeling stochastic reaction networks.

Quickstart
==========

1. Install:

.. code-block:: python

   pip install physDBD
    
2. See the :doc:`Quickstart </quickstart>`.

3. See the example notebook in the example folder of the `GitHub repo <https://github.com/smrfeld/phys_dbd>`_.

4. Scan the :ref:`api_ref`.

About
=====

This package for TensorFlow implements modeling stochastic reaction networks 
with a dynamic PCA model. `Please see this paper for technical details: <https://arxiv.org/abs/2109.05053>`_

`O. K. Ernst, T. Bartol, T. Sejnowski and E. Mjolsness. Physics-based machine learning for modeling stochastic IP3-dependent calcium dynamics. arXiv:2109.05053`

The original implementation in the paper is written in 
`Mathematica` and can be found `here <https://github.com/smrfeld/physics-based-ml-reaction-networks>`_. 
The Python package developed here translates these methods to `TensorFlow`.

The only current supported probability distribution is the Gaussian distribution defined by PCA; more general Gaussian distributions are a work in progress.

Requirements
============

* `TensorFlow 2.5.0` or later. *Note: later versions not tested.*
* `Python 3.7.4` or later.

Installation
============

Either: use `pip`:

.. code-block:: python

   pip install physDBD

Or alternatively, clone this `repo. from GitHub <https://github.com/smrfeld/phys_dbd>`_ and use the provided `setup.py`:

.. code-block:: python

   python setup.py install

API Documentation
=================

See the :ref:`api_ref`.

Example
=======

See the notebook in the example directory in `GitHub repo. <https://github.com/smrfeld/phys_dbd>`_

Citing
======

`Please cite this paper: <https://arxiv.org/abs/2109.05053>`_

`O. K. Ernst, T. Bartol, T. Sejnowski and E. Mjolsness. Physics-based machine learning for modeling stochastic IP3-dependent calcium dynamics. arXiv:2109.05053`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart.md
   modules.rst