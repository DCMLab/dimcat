.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/dimcat.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/dimcat
    .. image:: https://readthedocs.org/projects/dimcat/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://dimcat.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/dimcat/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/dimcat
    .. image:: https://img.shields.io/conda/vn/conda-forge/dimcat.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/dimcat
    .. image:: https://pepy.tech/badge/dimcat/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/dimcat
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/dimcat

.. image:: https://img.shields.io/pypi/v/dimcat.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/dimcat/

.. image:: https://readthedocs.org/projects/dimcat/badge/?version=latest
    :target: https://dimcat.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


======
DiMCAT
======


    DIgital Musicology Corpus Analysis Toolkit


A Python library for processing and analyzing notated music on a very large scale. It is under heavy development and
has just seen its v1.0.0 alpha release. The library is developed by the Digital and Cognitive Musicology Lab at the
École Polytechnique Fédérale de Lausanne (EPFL) in Switzerland and a white paper has been published as

    Hentschel, J., McLeod, A., Rammos, Y., & Rohrmeier, M. (2023). Introducing DiMCAT for processing and analyzing notated music on a very large scale. Proceedings of the 24th International Society for Music Information Retrieval Conference, 516–523. https://ismir2023program.ismir.net/poster_52.html



Installation
============

DiMCAT is available on PyPI and can be installed via pip:

.. code-block:: bash

    pip install dimcat

Quickstart
==========

DiMCAT compiles frictionless datapackages. To play around with the alpha release, we recommend downloading the package
which corresponds to the DCML corpora that are currently public. The package consists of two files:

* `dcml_corpora.zip <https://github.com/DCMLab/dcml_corpora/releases/download/v2.0/dcml_corpora.zip>`__ (data)
* `dcml_corpora.json <https://github.com/DCMLab/dcml_corpora/releases/download/v2.0/dcml_corpora.datapackage.json>`__ (metadata)

The data package can be loaded into DiMCAT as follows:

.. code-block:: python

    from dimcat import Dataset

    D = Dataset.from_package("dcml_corpora.datapackage.json")


Acknowledgements
================

Development of this software tool was supported by the Swiss National Science Foundation within the
project “Distant Listening – The Development of Harmony over Three Centuries (1700–2000)”
(Grant no. 182811). This project is being conducted at the Latour Chair in Digital and Cognitive
Musicology, generously funded by Mr. Claude Latour.

The software project has been set up using PyScaffold 4.2.1.

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/
