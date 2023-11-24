===================
Contributing & Help
===================

Welcome to ``DiMCAT`` contributor's guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but `other kinds of contributions`_ are also
appreciated.

If you are new to using git_ or have never collaborated in a project previously,
please have a look at `contribution-guide.org`_. Other resources are also
listed in the excellent `guide created by FreeCodeCamp`_ [#contrib1]_.

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, `Python Software
Foundation's Code of Conduct`_ is a good reference in terms of behavior
guidelines.


Issue Reports
=============

If you experience bugs or general issues with ``DiMCAT``, please have a look
on the `issue tracker`_. If you don't see anything useful there, please feel
free to fire an issue report.

.. tip::
   Please don't forget to include the closed issues in your search.
   Sometimes a solution was already reported, and the problem is considered
   **solved**.

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.


Documentation Improvements
==========================

You can help improve ``DiMCAT`` docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

``DiMCAT`` documentation uses Sphinx_ as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution.

Documentation pages are written in reStructuredText_ (as are the docstrings that are automatically compiled to the
API docs) or as [MyST notebooks](https://myst-nb.readthedocs.io/en/latest/authoring/basics.html) that are run and
rendered to HTML when building the docs (see also the syntax guide at MyST_).

.. tip::
  Please notice that the `GitHub web interface`_ provides a quick way of
  propose changes in ``DiMCAT``'s files. While this mechanism can
  be tricky for normal code contributions, it works perfectly fine for
  contributing to the docs, and can be quite handy.

  If you are interested in trying this method out, please navigate to
  the ``docs`` folder in the source repository_, find which file you
  would like to propose changes and click in the little pencil icon at the
  top, to open `GitHub's code editor`_. Once you finish editing the file,
  please write a message in the form at the bottom of the page describing
  which changes have you made and what are the motivations behind them and
  submit your proposal.

When working on documentation changes in your local machine, you can
compile them using |tox|_::

    tox -e docs

and use Python's built-in web server for a preview in your web browser
(``http://localhost:8000``)::

    python3 -m http.server --directory 'docs/_build/html'


Code Contributions
==================

.. admonition:: TL;DR

   * Fork the repository.
   * (Create a virtual environment, :ref:`see below <virtenv>`).
   * Head into the local clone of your fork and hit ``pip install -e ".[dev]"`` (where ``.`` is the current directory).
   * Install the precommit hooks via ``pre-commit install``.
   * Implement the changes and create a Pull Request against the ``development`` branch.
   * Thank you!


Submit an issue
---------------

Before you work on any non-trivial code contribution it's best to first create
a report in the `issue tracker`_ to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

.. _virtenv:

Create an environment
---------------------

Before you start coding, we recommend creating an isolated `virtual
environment`_ to avoid any problems with your installed Python packages.
This can easily be done via either |virtualenv|_::

    virtualenv <PATH TO VENV>
    source <PATH TO VENV>/bin/activate

or Miniconda_::

    conda create -n dimcat python=3 six virtualenv pytest pytest-cov
    conda activate dimcat

Clone the repository
--------------------

#. Create an user account on |the repository service| if you do not already have one.
#. Fork the project repository_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on |the repository service|.
#. Clone this copy to your local disk::

    git clone git@github.com:YourLogin/dimcat.git
    cd dimcat

#. You should run::

    pip install -U pip -e ".[dev]"

   to be able to import the package under development in the Python REPL.

#. Install |pre-commit|_::

    pip install pre-commit
    pre-commit install

   ``DiMCAT`` comes with a lot of hooks configured to automatically help the
   developer to check the code being written.

Implement your changes
----------------------

#. Create a branch to hold your changes::

    git checkout -b my-feature

   and start making changes. Never work on the main branch!

#. Start your work on this branch. Don't forget to add docstrings_ to new
   functions, modules and classes, especially if they are part of public APIs.

#. Add yourself to the list of contributors in ``AUTHORS.rst``.

#. When you’re done editing, do::

    git add <MODIFIED FILES>
    git commit

   to record your changes in git_.

   Please make sure to see the validation messages from |pre-commit|_ and fix
   any eventual issues.
   This should automatically use flake8_/black_ to check/fix the code style
   in a way that is compatible with the project.

   .. important:: Don't forget to add unit tests and documentation in case your
      contribution adds an additional feature and is not just a bugfix.

      Moreover, writing a `descriptive commit message`_ is highly recommended.
      In case of doubt, you can check the commit history with::

         git log --graph --decorate --pretty=oneline --abbrev-commit --all

      to look for recurring communication patterns.

#. Please check that your changes don't break any unit tests with::

    tox

   (after having installed |tox|_ with ``pip install tox`` or ``pipx``).

   You can also use |tox|_ to run several other pre-configured tasks in the
   repository. Try ``tox -av`` to see a list of the available checks.

Submit your contribution
------------------------

#. If everything works fine, push your local branch to |the repository service| with::

    git push -u origin my-feature

#. Go to the web page of your fork and click |contribute button|
   to send your changes for review.

   Find more detailed information in `creating a PR`_. You might also want to open
   the PR as a draft first and mark it as ready for review after the feedbacks
   from the continuous integration (CI) system or any required fixes.



DiMCAT architecture
-------------------

1. The library is called DiMCAT and has three high-level objects:

   a. :class:`~.DimcatObject` ("object"): the base class for all objects that manages object creation and serialization and subclass registration.
      The DimcatObject class has a class attribute called _registry that is a dictionary of all subclasses of DimcatObject.
      Each DimcatObject has a nested class called Schema that inherits from DimcatSchema.
   #. :class:`~.DimcatSchema` ("schema"): the base class for all nested Schema classes, inheriting from marshmallow.Schema.
      The Schema defines the valid values ranges for all attributes of the DimcatObject and how to serialize and deserialize them.
   #. :class:`~.DimcatConfig` ("config"): a DimcatObject that can represent a subset of the attributes of another DimcatObject and instantiate it using the .create() method.
      It derives from MutableMapping and used for communicating about and checking the compatibility of DimcatObjects.

#. The nested Schema corresponding to each DimcatObject is instantiated as a singleton and can be retrieved via the class attribute :attr:`~.DimcatObject.schema`.
   Using this Schema, a DimcatObject can be serialized to and deserialized from:

   a. a dictionary using the :meth:`~.DimcatObject.to_dict` and :meth:`~.DimcatObject.from_dict` methods.
   #. a DimcatConfig object using the :meth:`~.DimcatObject.to_config` and :meth:`~.DimcatObject.from_config` methods.
   #. a JSON string using the :meth:`~.DimcatObject.to_json` and :meth:`~.DimcatObject.from_json` methods.
   #. a JSON file using the :meth:`~.DimcatObject.to_json_file` and :meth:`~.DimcatObject.from_json_file` methods.

   In the following, by "serialized object" we mean its representation as a DimcatConfig if not otherwise specified.

#. All objects that are neither a schema nor a config are one of the two following subclasses of DimcatObject:

   a. :class:`~.Data`: a DimcatObject that represents a dataset, a subset of a dataset, or a an individual resource such as a dataframe.
   #. :class:`~.PipelineStep`: a DimcatObject that accepts a Data object as input and returns a Data object as output.

#. The principal Data object is called :class:`~.Dataset` and is the one that users will interact with the most.
   The Dataset provides convenience methods that are equivalent to applying the corresponding PipelineStep.
   Every PipelineStep applied to it will return a new Dataset that can be serialized and deserialized to re-start the pipeline from that point.
   To that aim, every Dataset stores a serialization of the applied PipelineSteps and of the original Dataset that served as initial input.
   This initial input is specified as a :class:`~.DimcatCatalog` which is a collection of :class:`DimcatPackages <.data.dataset.base.DimcatPackage>`,
   each of which is a collection of :class:`DimcatResources <.data.resources.base.DimcatResource>`,
   as defined by the `Frictionless Data specifications <https://frictionlessdata.io>`__.
   The preferred structure of a DimcatPackage is a .zip and a datapackage.json file, where the former contains one or several .tsv files (resources) described in the latter.
   Since the data that DiMCAT transforms and analyzes comes from very heterogeneous sources, each original corpus is pre-processed and stored as a `frictionless.Package <https://framework.frictionlessdata.io/docs/framework/package.html>`__ together with the metadata relevant for reproducing the pre-processing.
#. It follows that the Dataset is mainly a container for :class:`DimcatResources <.data.resources.base.DimcatResource>` namely:

   a. Facets, i.e. the resources described in the original datapackage.json. They aim to stay as faithful as possible to the original data, applying only mild standardization and normalization.
      All Facet resources come with several columns that represent timestamps both in absolute and in musical time, allowing for the alignment of different corpora.
      The `Frictionless resource <https://framework.frictionlessdata.io/docs/framework/resource.html>`__ descriptors listed in the datapackage.json contain both the column schema and the piece IDs that are present in each of the facets.
   #. :class:`Features <.data.resources.features.Feature>`, i.e. resources derived from Facets by applying PipelineSteps. They are standardized objects that are requested by the PipelineSteps to compute statistics and visualizations.
      To allow for straightforward serialization of the Dataset, all Feature resources are represented as a DimcatCatalog called `outputs`, which can be stored as .tsv files in one or several .zip files.

#. A :class:`~.DimcatResource` functions similarly to the `frictionless.Resource <https://framework.frictionlessdata.io/docs/framework/resource.html>`__ that it wraps, meaning that it grants access to the metadata without having to load the dataframes into memory.
   It can be instantiated in two different ways, either from a resource descriptor or from a dataframe.
   At any given moment, the :attr:`~.DimcatResource.status` attribute returns an Enum value reflecting the availability and state of the/a dataframe.
   When a Dataset is serialized, all dataframes from the outputs catalog that haven't been stored to disk yet are written into one or several .zip files so that they can be referenced by resource descriptors.
#. One of the most important methods, used by most PipelineSteps, is :meth:`.Dataset.get_feature`, which accepts a Feature config and returns a Feature resource.
   The Feature config is a :class:`~.DimcatConfig` that specifies the type of Feature to be returned and the parameters to be used for its computation. Furthermore, it is also used

   a. to determine for each piece in every loaded DimcatPackage an Availability value, ranging from not available over available with heavy computation to available instantly.
   #. to determine whether the Feature resource had already been requested and stored in the outputs catalog.


Coding Conventions
------------------

Please make sure to run ``pre-commit install`` in your local clone of the repository. This way, many coding
conventions are automatically applied before each commit!

Commit messages
~~~~~~~~~~~~~~~

``DiMCAT`` uses `Conventional Commits <https://www.conventionalcommits.org/>`__ to determine the next SemVer version number. Please make sure to prefix each
message with one of:

+-------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+
| Commit Type | Title                    | Description                                                                                                 | SemVer |
+=============+==========================+=============================================================================================================+========+
| `feat`      | Features                 | A new feature                                                                                               | MINOR  |
| `fix`       | Bug Fixes                | A bug Fix                                                                                                   | PATCH  |
| `docs`      | Documentation            | Documentation only changes                                                                                  | PATCH  |
| `style`     | Styles                   | Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)      | PATCH  |
| `refactor`  | Code Refactoring         | A code change that neither fixes a bug nor adds a feature                                                   | PATCH  |
| `perf`      | Performance Improvements | A code change that improves performance                                                                     | PATCH  |
| `test`      | Tests                    | Adding missing tests or correcting existing tests                                                           | PATCH  |
| `build`     | Builds                   | Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)         | PATCH  |
| `ci`        | Continuous Integrations  | Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs) | PATCH  |
| `chore`     | Chores                   | Other changes that don't modify src or test files                                                           | PATCH  |
| `revert`    | Reverts                  | Reverts a previous commit                                                                                   | PATCH  |
+-------------+--------------------------+-------------------------------------------------------------------------------------------------------------+--------+

In the case of breaking changes, which result in a new major version, please add a ``!`` after the type, e.g., ``refactor!:``.
This type of commit message needs to come with a body, starting with ``BREAKING CHANGE:``, which explains in great detail everything
that will not be working anymore.

Internal imports
~~~~~~~~~~~~~~~~

The top level of the `src/dimcat` directory consists of the two packages ``data`` and ``steps`` and a couple of
files which, here, we call ``base``.

* All modules can import from ``base`` and ``data``.
* ``data`` modules should not import from ``steps``. Whenever a step is needed, its constructor can be retrieved using :func:`dimcat.base.get_class` function.
* All modules can import from ``dimcat.utils`` except for ``dimcat.base``. Likewise, the ``base`` module of any package cannot import from its sibling ``.utils``.
  This makes it possible to pull up those elements would otherwise be defined in one of the adjacent modules (which generally do import ``.utils``),
  but which are required by one or several utility functions. For example, the Enum ``dimcat.data.resources.base.FeatureName``, conceptually, belongs into
  ``dimcat.data.resources.dc``, where the ``Feature`` class is defined. However, since ``dimcat.data.resources.utils.feature_specs2config()`` needs to import the Enum,
  it is moved up to ``dimcat.data.resources.base``. Hence, no base module shall ever import from ``.utils``. Any utility functions it requires can go into the ``utils``
  of its parent -- or ``dimcat.utils``.

Order of module members
~~~~~~~~~~~~~~~~~~~~~~~

* imports
* constants
* one or several groups, enclosed in ``# region <name>`` and ``# endregion <name>`` comments:

  * classes
  * functions

Order of attributes and methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each bullet point represents an alphabetically sorted group of attributes or methods. Private attributes and methods
are sorted as if they didn't have a leading underscore.

* class members

  * class variables
  * ``@staticmethod``
  * @classmethod
    @property
  * ``@classmethod``
  * nested classes (esp. ``Schema()``)

* instance members

  * ``__init__()``
  * magic methods
  * ``@property`` and setters
  * public and private methods




Troubleshooting
---------------

The following tips can be used when facing problems to build or test the
package:

#. Make sure to fetch all the tags from the upstream repository_.
   The command ``git describe --abbrev=0 --tags`` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   ``.eggs``, as well as the ``*.egg-info`` folders in the ``src`` folder or
   potentially in the root of your project.

#. Sometimes |tox|_ misses out when new dependencies are added, especially to
   ``setup.cfg`` and ``docs/requirements.txt``. If you find any problems with
   missing dependencies when running a command with |tox|_, try to recreate the
   ``tox`` environment using the ``-r`` flag. For example, instead of::

    tox -e docs

   Try running::

    tox -r -e docs

#. Make sure to have a reliable |tox|_ installation that uses the correct
   Python version (e.g., 3.7+). When in doubt you can run::

    tox --version
    # OR
    which tox

   If you have trouble and are seeing weird errors upon running |tox|_, you can
   also try to create a dedicated `virtual environment`_ with a |tox|_ binary
   freshly installed. For example::

    virtualenv .venv
    source .venv/bin/activate
    .venv/bin/pip install tox
    .venv/bin/tox -e all

#. `Pytest can drop you`_ in an interactive session in the case an error occurs.
   In order to do that you need to pass a ``--pdb`` option (for example by
   running ``tox -- -k <NAME OF THE FALLING TEST> --pdb``).
   You can also setup breakpoints manually instead of using the ``--pdb`` option.


Maintainer tasks
================

Releases
--------


If you are part of the group of maintainers and have correct user permissions
on PyPI_, the following steps can be used to release a new version for
``DiMCAT``:

#. Make sure all unit tests are successful.
#. Tag the current commit on the main branch with a release tag, e.g., ``v1.2.3``.
#. Push the new tag to the upstream repository_, e.g., ``git push upstream v1.2.3``
#. Clean up the ``dist`` and ``build`` folders with ``tox -e clean``
   (or ``rm -rf dist build``)
   to avoid confusion with old builds and Sphinx docs.
#. Run ``tox -e build`` and check that the files in ``dist`` have
   the correct version (no ``.dirty`` or git_ hash) according to the git_ tag.
   Also check the sizes of the distributions, if they are too big (e.g., >
   500KB), unwanted clutter may have been accidentally included.
#. Run ``tox -e publish -- --repository pypi`` and check that everything was
   uploaded to PyPI_ correctly.



.. [#contrib1] Even though, these resources focus on open source projects and
   communities, the general ideas behind collaborating with other developers
   to collectively create software are general and can be applied to all sorts
   of environments, including private companies and proprietary code bases.


.. <-- start -->

.. |the repository service| replace:: GitHub
.. |contribute button| replace:: "Create pull request"

.. _repository: https://github.com/DCMLab/dimcat
.. _issue tracker: https://github.com/DCMLab/dimcat/issues
.. <-- end -->


.. |virtualenv| replace:: ``virtualenv``
.. |pre-commit| replace:: ``pre-commit``
.. |tox| replace:: ``tox``


.. _black: https://pypi.org/project/black/
.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _flake8: https://flake8.pycqa.org/en/stable/
.. _git: https://git-scm.com
.. _GitHub's fork and pull request workflow: https://guides.github.com/activities/forking/
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _MyST: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _PyScaffold's contributor's guide: https://pyscaffold.org/en/stable/contributing.html
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tox: https://tox.wiki/en/stable/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/

.. _GitHub web interface: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
.. _GitHub's code editor: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
