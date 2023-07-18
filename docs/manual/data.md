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
import frictionless as fl
from dimcat.base import deserialize_json_file
from dimcat.data.resource.base import Resource, PathResource
from dimcat.data.resource.dc import DimcatResource
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

Let's exemplify looking at the

### PathResource

The `sweelinck_keyboard` repository contains a single MuseScore file (in the folder "MS3") and several TSV files extracted from it.
Let's load it:

```{code-cell} ipython3
score_resource = os.path.join(sweelinck_dir, "MS3", "SwWV258_fantasia_cromatica.mscx")
score_resource = PathResource.from_resource_path(score_resource)
score_resource.get_path_dict()
```

The dictionary returned by `.get_path_dict()` tell us everything we need to know to handle the resource physically:

* `basepath` is an absolute directory
* `filepath` is the filepath (which can include subfolders), relative to the `basepath`
* `normpath` is the full path to the resource and defined as `basepath/filepath` (both need to be specified)
* `innerpath`: when `normpath` points to a .zip file, innerpath is the relative filepath of the resource within the ZIP archive
* `descriptor_filename` stores the name of a descriptor when it deviates from the default `<resource_name>.resource.json`. Cannot include subfolders since it is expected to be stored in `basepath` (otherwise, the relative `filepath` stored in the descriptor would resolve incorrectly)
* `descriptor_path`: defined by `basepath/descriptor_filename`

Here, the descriptor_path corresponds to the default, which does not currently point to an existing file:

```{code-cell} ipython3
score_resource.descriptor_exists
```

It can be created using `.store_descriptor()`:

```{code-cell} ipython3
score_descriptor_path = score_resource.store_descriptor()
score_resource.descriptor_exists
```

To underline the functionality of the path resource, even the new descriptor can be treated as a resource:

```{code-cell} ipython3
PathResource.from_resource_path(score_descriptor_path)
```

Which is different from creating the original PathResource from the created descriptor:

```{code-cell} ipython3
PathResource.from_descriptor_path(score_descriptor_path)
```

Note that the `descriptor_filename` is now set to keep track of the existing one the resource originates from.

By the way, the descriptors written to disk qualify as "normal" DimcatConfigs (see ???)...

```{code-cell} ipython3
deserialize_json_file(score_descriptor_path)
```

... and at the same time as valid Frictionless descriptors that can be validated using its commandline tool or Python library:

```{code-cell} ipython3
fl.validate(score_descriptor_path)
```

This is also what the property `is_valid` uses under the hood:

```{code-cell} ipython3
score_resource.is_valid
```

The status of a PathResource is always and unchangeably `PATH_ONLY`, with a value one above `EMPTY`:

```{code-cell} ipython3
score_resource.status
```

The path components cannot be modified because it would invalidate the relations with other path components:

```{code-cell} ipython3
base_path_level_up = os.path.dirname(score_resource.basepath)
score_resource.basepath = base_path_level_up
```

We can either create the same resource once more, specifying the new basepath at once:

```{code-cell} ipython3
PathResource.from_descriptor_path(score_descriptor_path)
```

### DimcatResource

A DimcatResource is both a Resource in the above sense and a wrapped dataframe.
Let's create one from a TSV resource descriptor:

```{code-cell} ipython3
notes_descriptor_path = os.path.join(sweelinck_dir, "notes", "SwWV258_fantasia_cromatica.notes.resource.json")
notes_resources = DimcatResource.from_descriptor_path(notes_descriptor_path)
notes_resources
```

```{code-cell} ipython3
notes_resources.df
```

```{code-cell} ipython3
type(notes_resources)
```

Note that

```{code-cell} ipython3
score_resource.resource_exists
```

```{code-cell} ipython3
score_resource.is_serialized
```

```{code-cell} ipython3
score_resource.status
```

```{code-cell} ipython3
score_resource.to_dict()
```

```{code-cell} ipython3
score_resource.to_dict(pickle=True)
```

```{code-cell} ipython3
score_resource.to_config().create()
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
