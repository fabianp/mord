.. mord documentation master file, created by
   sphinx-quickstart on Tue Jan  6 09:55:06 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mord: Ordinal Regression in Python
===================================

mord is a Python package that implements some ordinal regression methods following the scikit-learn API.

.. image:: ordinal_1.png
   :align: center


What is ordinal regression ?
-----------------------------

Ordinal Regression denotes a family of statistical learning methods in which the goal is to predict a variable that is discrete and ordered. For example, predicting the movie rating on a scale of 1 to 5 starts can be considered an ordinal regression task.

In this package we provide different models for the ordinal regression task. We categorize them between :ref:`threshold-based`, :ref:`regression-based` and :ref:`classification-based`.


.. _threshold-based:

Threshold-based models
----------------------
Threshold-based loss functions partition the real like into `k` segments and then

  * :class:`mord.LogisticIT`
  * :class:`mord.LogisticAT`


.. _regression-based:

Regression-based models
-----------------------

  * :class:`mord.OrdinalRidge`
  * :class:`mord.LAD`

.. _classification-based:

Classification-based models
---------------------------

  * :class:`mord.MulticlassLogistic`



Citing
======

If you find this software useful, please consider citing:

`Fabian Pedregosa-Izquierdo. Feature extraction and supervised learning on fMRI: from practice to theory. PhD thesis. <https://tel.archives-ouvertes.fr/tel-01100921>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

