---
jupytext:
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

# Quick demo

## Import dimcat and load data

```{code-cell} ipython3
import dimcat as dc
from dimcat.data import resources
from dimcat.steps import analyzers, extractors, groupers

package_path = "/home/laser/git/dimcat/docs/mwe/dcml_corpora.datapackage.json"
dataset = dc.Dataset.from_package(package_path)
dataset
```

## Show metadata

```{code-cell} ipython3
dataset.get_metadata()
```

## Counting notes

### Variant 1: Extract feature, apply Counter

Here we pass the extracted notes to the counter.

```{code-cell} ipython3
notes = dataset.get_feature("notes")
result = analyzers.Counter().process(notes)
result.plot()
```

The `FeatureExtractor` is added to the dataset implicitly, but not the `Counter` because it's applied only to the extracted feature:

```{code-cell} ipython3
dataset
```

### Variant 2: Imply feature extraction in the analyzer

Here we pass the dataset to the counter.

```{code-cell} ipython3
counter = analyzers.Counter(features="notes")
analyzed_dataset = counter.process(dataset)
analyzed_dataset.get_result().plot()
```

Applying an `Analyzer` to a `Dataset` yields an `AnalyzedDataset` that includes one `Result` resource per analyzed `Feature`.
Both are to be found in the respective packages in the outputs catalog:

```{code-cell} ipython3
analyzed_dataset
```

### Variant 3: Define a Pipeline with FeatureExtractor and Counter

```{code-cell} ipython3
pipeline = dc.Pipeline([
    extractors.FeatureExtractor("notes"),
    analyzers.Counter()
])
analyzed_dataset = pipeline.process(dataset)
analyzed_dataset.get_result().plot()
```

## Grouped note counts

Let's define a CustomPieceGrouper from random piece groups:

```{code-cell} ipython3
n_groups = 10
n_members = 10

piece_index = resources.PieceIndex.from_resource(notes)
grouping = {f"group_{i}": piece_index.sample(n_members) for i in range(1, n_groups + 1)}
grouper = groupers.CustomPieceGrouper.from_grouping(grouping)
grouper
```

### Applying the grouper to the analysis result

```{code-cell} ipython3
grouped_result = grouper.process(result)
grouped_result
```

```{code-cell} ipython3
grouped_result.plot()
```

### Including the grouper in the analysis pipeline

```{code-cell} ipython3
pipeline.add_step(grouper)
pipeline.info()
```

```{code-cell} ipython3
analyzed_dataset = pipeline.process(dataset)
analyzed_dataset.get_result().plot()
```

## Assembling the Pipeline from DimcatConfig objects

Serialization of any DimcatObject uses the `DimcatConfig` object.

```{code-cell} ipython3
step_configs = [
    dict(dtype="FeatureExtractor", features=[dict(dtype="Notes", format="FIFTHS")]),
    dict(dtype="Counter"),
    dict(dtype='CustomPieceGrouper', grouped_pieces=grouping)
]
pl = dc.Pipeline.from_step_configs(step_configs)
pl
```

```{code-cell} ipython3
resulting_dataset = pl.process(dataset)
resulting_dataset.get_result().plot()
```

```{code-cell} ipython3

```
