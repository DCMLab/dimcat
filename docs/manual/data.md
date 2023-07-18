---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.6
kernelspec:
  display_name: dimcat
  language: python
  name: dimcat
---

```{code-cell} ipython3
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os
from dimcat.data.resource.base import Resource, PathResource
CORPUS_PATH = os.path.abspath(os.path.join("..", "..", "unittest_metacorpus"))
assert os.path.isdir(CORPUS_PATH)
sweelinck_dir = os.path.join(CORPUS_PATH, "sweelinck_keyboard")
```

# Data

## Resource

A resource is a combination of a file and its descriptor.
It allows for interacting with the file without having to "touch" it by interacting with its descriptor only.
The descriptor comes in form of a dictionary and is typically stored next to the file in JSON or YAML format.

DiMCAT follows the [Frictionless specification](https://specs.frictionlessdata.io/) for describing resources.
There are two types of resources:

* [PathResource](PathResource): Stands for a resource on local disk or on the web.
* [DimcatResource](DimcatResource): A [Frictionless Tabular Data Resource](https://specs.frictionlessdata.io/tabular-data-resource/).

They can be instantiated from a single filepath using the constructors

* `.from_resource_path()` which takes the path to the resource file to be described
* `.from_descriptor_filepath()` which takes a filepath pointing to a JSON or YAML file containing a resource descriptor

### Resource.from_resource_path()

The `sweelinck_keyboard` repository contains a single MuseScore file (in the folder "MS3") and several TSV files extracted from it.
Let's load it:

```{code-cell} ipython3
score_resource = os.path.join(sweelinck_dir, "MS3", "SwWV258_fantasia_cromatica.mscx")
PathResource.from_resource_path(score_resource)
```

```{code-cell} ipython3
notes_descriptor_path = os.path.join(sweelinck_dir, "notes", "SwWV258_fantasia_cromatica.notes.resource.json")
notes_path_resource = Resource.from_descriptor_path(notes_descriptor_path)
notes_path_resource = PathResource.from_descriptor_path(notes_descriptor_path)
notes_path_resource
```

```{code-cell} ipython3

notes_resource = Resource.from_descriptor_path(notes_descriptor_path)
notes_resource
```

```{code-cell} ipython3

```
