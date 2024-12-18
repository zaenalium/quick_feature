.. -*- mode: rst -*-

.. currentmodule:: quick_feature.wrappers

Scikit-learn Wrapper
====================

quick_feature's Scikit-learn wrappers wrap Scikit-learn transformers allowing their
implementation only on a selected subset of features.

.. toctree::
   :maxdepth: 1

   Wrapper

Other wrappers
~~~~~~~~~~~~~~

The :class:`SklearnTransformerWrapper()` offers a similar function to the
`ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_
class available in Scikit-learn. They differ in the implementation to select the
variables.