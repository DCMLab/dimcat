---
jupytext:
  execution:
    allow_errors: true
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: dimcat
  language: python
  name: dimcat
---

# DiMCAT Errors

This notebook is a collection of errors. It serves as a resource to look up explanations for various kinds of errors and
for other parts of the documentation to link to.

```{code-cell}
import dimcat as dc

package_path = "dcml_corpora.datapackage.json"
dataset = dc.Dataset.from_package(package_path)
dataset
```

## DimcatConfig Errors

### Invalid dtype

```{code-cell}
:tags: [raises-exception]

config = dc.DimcatConfig(dtype="Bananas")
```

### Invalid option

The [Counter](analyzers.counters.Counter) analyzer has no option called `invalid_option`.

```{code-cell}
:tags: [raises-exception]

config = dc.DimcatConfig(dtype="Counter", invalid_option="Notes")
```

The [Notes](resources.features.Notes) features has an option called `format`, but it does not accept the value `"bananas"`.

```{code-cell}
:tags: [raises-exception]

config = dc.DimcatConfig(dtype="Notes", format="bananas")
```

```{code-cell}

```
