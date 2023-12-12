---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: dimcat
  language: python
  name: dimcat
---

# Quick demo

## Import dimcat and load data

```{code-cell}
import dimcat as dc
from dimcat.data import resources
from dimcat.steps import analyzers, extractors, groupers

package_path = "dcml_corpora.datapackage.json"
dataset = dc.Dataset.from_package(package_path)
dataset
```

## Show metadata

```{code-cell}
dataset.get_metadata()
```

## Counting notes

### Variant 1: Extract feature, apply Counter

Here we pass the extracted notes to the counter.

```{code-cell}
notes = dataset.get_feature("notes")
result = analyzers.Counter().process(notes)
result.plot()
```

The `FeatureExtractor` is added to the dataset's pipeline implicitly, but the `Counter` is not because it's applied only to the extracted feature:

```{code-cell}
dataset
```

The pitch-class distributions shown by `.plot()` correspond to the current **unit of analysis**, which defaults to the piece-level.
Results also come with a second plotting method, `.plot_grouped()`. Since no groupers have been applied, the entire dataset is treated as a single group:

```{code-cell}
result.plot_grouped()
```

### Variant 2: Imply feature extraction in the analyzer

Here we pass the dataset to the counter.

```{code-cell}
counter = analyzers.Counter(features="notes")
analyzed_dataset = counter.process(dataset)
analyzed_dataset.get_result().plot()
```

Applying an `Analyzer` to a `Dataset` yields an `AnalyzedDataset` that includes one `Result` resource per analyzed `Feature`.
Both are to be found in the respective packages in the outputs catalog:

```{code-cell}
analyzed_dataset
```

### Variant 3: Define a Pipeline with FeatureExtractor and Counter

```{code-cell}
pipeline = dc.Pipeline([
    extractors.FeatureExtractor("notes"),
    analyzers.Counter()
])
analyzed_dataset = pipeline.process(dataset)
analyzed_dataset.get_result().plot()
```

## Grouped note counts

Let's define a CustomPieceGrouper from random piece groups:

* We create a `PieceIndex`, which is essentially a fancy list of piece ID tuples.
* From this, we sample `n_groups` groups of `n_members` piece tuples each. A `grouping` is a mapping of group names to piece IDs.
* Then, we set up a `CustomPieceGrouper` from the grouping. Inspecting it, we see that it stores a `PieceIndex` in which the first
  level corresponds to the three group names, `group_1`, `group_2`, and `group_3`. Whenever we apply this grouper, it will prepend
  this level to any processed Resource (provided it contains the grouped pieces). This changes the behaviour of the grouped resource,
  e.g. when plotting it.

```{code-cell}
n_groups = 3
n_members = 30

piece_index = resources.PieceIndex.from_resource(notes)
grouping = {f"group_{i}": piece_index.sample(n_members) for i in range(1, n_groups + 1)}
grouper = groupers.CustomPieceGrouper.from_grouping(grouping)
grouper
### Applying the grouper to the analysis result
```

```{code-cell}
grouped_result = grouper.process(result)
grouped_result
```

```{code-cell}
grouped_result.plot_grouped()
```

As promised, the grouped result plots differently: Instead of showing pitch-class distributions for each of the grouped pieces,
(which we can still obtain by calling `.plot()`), it shows the pitch-class distributions for each of the groups.
However, for closer inspection, the area of a circle is not ideal, so let's view it as a bar plot:

```{code-cell}
grouped_result.make_bar_plot()
```

### Step.process(Data) == Data.apply_step(Step)

Above, we have applied **Steps**, an analyzer, a grouper, and a pipeline, to **Data** objects, namely
resources (to the `Notes` feature and to the `Counts` result) and to a dataset containing these resources.
Another way to achieve the same goal is by applying steps to data. Let's start with a fresh dataset and
apply the grouper and the analyzer once more:

```{code-cell}
D = dc.Dataset.from_package(package_path)
analyzed_dataset = D.apply_step(grouper, counter)
analyzed_dataset
```

```{code-cell}
result = analyzed_dataset.get_result()
result
```

```{code-cell}
result.default_groupby
```

```{code-cell}
analyzed_dataset.get_result().make_bar_plot()
```

## Assembling the Pipeline from DimcatConfig objects

Serialization of any DimcatObject uses the `DimcatConfig` object. Each config needs to have at least the key `dtype`,
specifying the name of a DimcatObject. Any other keys need to correspond to init arguments of that object. Wrong keys
or invalid values [are rejected](./errors.md#invalid-option).

Any DimcatObject can be expressed as a config by calling its `.to_config()` method:

```{code-cell}
config = counter.to_config()
config
```

Any config can be used to instantiate a DimcatObject:

```{code-cell}
counter_copy = config.create()
print(f"""The new object and the old object are
equal: {counter == counter_copy}
identical: {counter is counter_copy}""")
```

Wherever DiMCAT operates with configs, it also accepts dictionaries:

```{code-cell}
step_configs = [
    dict(dtype="FeatureExtractor", features=[dict(dtype="Notes", format="FIFTHS")]),
    dict(dtype='CustomPieceGrouper', grouped_units=grouping),
    dict(dtype="Counter")
]
pl = dc.Pipeline.from_step_configs(step_configs)
pl
```

```{code-cell}
resulting_dataset = pl.process(dataset)
resulting_dataset.get_result().make_bar_plot()
```
