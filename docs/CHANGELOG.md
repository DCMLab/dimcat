# Changelog

## [1.1.0](https://github.com/DCMLab/dimcat/compare/v1.0.1...v1.1.0) (2023-11-25)


### Features

* adds ClassVar DimcatResource._default_formatted_column and property DimcatResource.formatted_column to allow producing Results (and plots) with both the original and the formatted values. The properties formatted_column and value_column cannot be set directly (anymore). The former is to be controlled by the parameter 'format', which all Features now accept and serialize, too. Whereas 'value_column' will probably remain immutable. ([fc64c36](https://github.com/DCMLab/dimcat/commit/fc64c366e3ba4b9e4e6e0a53da29db3a5da066d2))
* eliminates results.LineOfFifthsDistribution() by merging the line-of-fifths plotting functionality into Result(), which decides based on the analyzed_resource's 'format' property whether it '.uses_line_of_fifths_colors' or not and, whenever necessary, removes GroupMode.COLOR from the 'group_modes'. This also gets rid of the special result types PitchClassDurations() and ScaleDegreeDurations() and of the special analyzer proportions.ScaleDegreeVectors() ([5390ae7](https://github.com/DCMLab/dimcat/commit/5390ae7b9f6e83179497746e1c802d6caf5c86f3))
* homogenizes code between dimcat.plotting.make_lof_bubble_plot() and make_lof_bar_plot(), making the latter accept an 'x_names_col' argument, too ([ce8ae53](https://github.com/DCMLab/dimcat/commit/ce8ae53161e53550c83d9b60ab02cad43e193d54))
* introduces TypeVar R for DimcatResource ([14e0615](https://github.com/DCMLab/dimcat/commit/14e061577edad83a715f9205fb07d86ebde551ad))
* Results are now initialized (and serialized) with parameters value_column, dimension_column (required) and formatted_column (optional). This allows, for example, for using the values to organize markers along the x-axis (e.g. numerically) while formatted_column may determine how the values are displayed. The dimension column comes from the new ClassVar Analyzer._dimension_column_name ([136833b](https://github.com/DCMLab/dimcat/commit/136833bcce679cf83cd2142bc8e0b802a8f82795))


### Bug Fixes

* extend functions that add convenience columns first check if they aren't already present (e.g. because the Feature is being created during a FeatureTransformation) ([a3e53a4](https://github.com/DCMLab/dimcat/commit/a3e53a4499ce50b1ee9f39389c41c912f472ebd5))
* tighter checks in Analyzer.check_resource() ([fa6575b](https://github.com/DCMLab/dimcat/commit/fa6575b8f0a9850207c85dcb31319579a767835e))


### Documentation

* includes v1.0.0 retrospectively ([479baee](https://github.com/DCMLab/dimcat/commit/479baee8e38ccf27da1f58e08241eaab912d12f5))

## [1.0.1](https://github.com/DCMLab/dimcat/compare/v1.0.0...v1.0.1) (2023-11-24)


### Bug Fixes

* has store_as_json_or_yaml() create target directories automatically ([e5fcf4f](https://github.com/DCMLab/dimcat/commit/e5fcf4f0728e167cbf33c3b0d7729b3e09eb9eda))
* includes the key columns in BassNotes feature ([864edc1](https://github.com/DCMLab/dimcat/commit/864edc1d29c91f29ad52f5a3094d8cbdf1b62020))


### Documentation

* configures please-release-action to use docs/CHANGELOG.md and converts the previous /CHANGELOG.rst ([d9ef418](https://github.com/DCMLab/dimcat/commit/d9ef418677df588a154535b0bc46209ea4fc0568))
* enables the inclusion of markdown documents in the documentation and enables MyST extensions ([86c19a9](https://github.com/DCMLab/dimcat/commit/86c19a9a4d3912c8343cf8c29b29e71816acac87))
* enables unittest_metacorpus submodule for RTD ([15d3317](https://github.com/DCMLab/dimcat/commit/15d3317951352bee86119cfd68b018259b4c1b71))
* includes jupyter_sphinx Sphinx extension for rendering interactive Plotly figures ([d363468](https://github.com/DCMLab/dimcat/commit/d36346895a20c5613d3d6784cacf0aa8f2fded93))
* moves changelog, authors, and license under the top-level heading "Imprint" ([7032162](https://github.com/DCMLab/dimcat/commit/7032162a5624f47c49df0c5bdd021d7879445de6))
* updates docs requirements to the latest release of dimcat[docs] ([dc04b76](https://github.com/DCMLab/dimcat/commit/dc04b763c594d48e98c5b407e23d9c106dfd412f))

## [1.0.0](https://github.com/DCMLab/dimcat/compare/0.3.0...v1.0.0) (2023-11-24)

This release completely invalidates the `0.x.x` releases: A lot of the conception survives, but not much of the original code.

### Included in this release

* A clean package/module structure centered on
  * a few base modules and objects; all other classes are conceptually separated between the two main packages:
  * `data` with its sub-packages
    * catalogs
    * datasets
    * packages
    * resources
  * `steps` with sub-packages
    * analyzers
    * extractors
    * groupers
    * loaders
    * pipelines
    * slicers
* All sub-sub packages can be imported directly and all included classes used directly, regardless of the module they live in. For example, you would

  ```python
  from dimcat import analyzers
  ```

  and then invoke analyzers as in `analyzers.Counter()`
* The basic usage pattern is set up, which lets you process individual resources as in `processed_resource = step.process(resource)` or entire datasets as in `processed_dataset = step.process(dataset)`.
* The skeleton for the documentation is setup and allows for the inclusion of MyST notebooks.




## Version 0.3.0

-   new slicers: MeasureSlicer, ChordFeatureSlicer
-   LocalKeySlicer has become a special case of ChordFeatureSlicer
-   Analyzers call method `post_process()` on the processed data.
    Introduced for enabling:
-   Overall consistent behaviour and naming of MultiIndex levels for all
    DataFrames
-   Corpus objects
    -   now come with the new method `get_facet()` that collects the
        results from `iter_facet()`
    -   come with a more performant slicing facility by slicing
        everything at once using the optimized function
        `ms3.utils.overlapping_chunk_per_interval()` (requires ms3 \>
        0.5.3)
    -   consistently store the sequence of applied PipelineSteps, the
        first being the most recently applied
    -   has a new method `get_previous_pipeline_step` allowing for
        retrieving one of them either by position or by type
-   PitchClassVectors allows for new settings:
    -   `weight_grace_durations` gives grace notes durations which by
        default are zero
    -   `include_empty=True` adds empty rows for segments without note
        occurrence
    -   all of which can be used via the commandline tool `dimcat pcvs`
-   ChordSymbolBigrams now
    -   comes with new setting `dropna` defaulting to True and excluding
        transitions from and to missing values
    -   checks before processing if LocalKeySlicer has been applied
        before
    -   checks before computing (staticmethod) if all transitions
        actually happen within the same localkey
-   All PipelineSteps how have a `filename_factory()` method
-   TSVWriter now
    -   automatically creates meaningful filenames from all applied
        PipelineSteps by calling their `filename_factory()`
    -   comes with the new settings `round` and `fillna`
-   Unittests store and check the git revision of the
    `unittest_metacorpus` repo, for which they are supposed to work

## Version 0.2.0

-   new slicers: NoteSlicer, LocalKeySlicer
-   new groupers: PieceGrouper, CorpusGrouper, YearGrouper, ModeGrouper
-   new filter: IsAnnotatedFilter
-   new writer: TSVWriter
-   installs system-wide command `dimcat pcvs`
-   consistent naming of DataFrame MultiIndex levels
-   more consistent interface, more abstractions

## Version 0.1.0

-   new analyzers: TPCrange, PitchClassVectors, ChordSymbolUnigrams,
    ChordSymbolBigrams
-   installs system-wide commands `dimcat unigrams` and `dimcat bigrams`
    for creating TSV files
