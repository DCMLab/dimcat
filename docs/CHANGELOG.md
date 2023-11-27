# Changelog

## [2.0.0](https://github.com/DCMLab/dimcat/compare/v1.1.0...v2.0.0) (2023-11-27)


### ⚠ BREAKING CHANGES

* the Enums HarmonyLabelsFormat and NotesFormat are losing the formats that are currently not implemented
* renaming of arguments and properties: grouped_pieces => grouped_units; piece_groups => grouping. This enables future subclasses of MappingGrouper able to flexibly group both pieces and slices, depending on the 'smallest_unit'

### Features

* adds CadenceFormat, CadenceCounter() analyzer and CadenceCounts() result type where .plot_grouper() defaults to .make_pie_chart() ([2d56ce1](https://github.com/DCMLab/dimcat/commit/2d56ce1294ee269889246c8e15c21d7ed97ea346))
* adds CriterionGrouper() and its first subclass HasCadenceAnnotations() ([94290f2](https://github.com/DCMLab/dimcat/commit/94290f2321011e06ef3b85fdaac9955a7dec88e6))
* adds method DimcatIndex.filter() ([ff8deba](https://github.com/DCMLab/dimcat/commit/ff8deba40bfe41bf583a369057f0ac38af2083f6))
* adds new feature CadenceLabels and lets all Feature._format_dataframe() end on the new ._sort_columns() ([5873550](https://github.com/DCMLab/dimcat/commit/587355099d028c23110e8788199ae752bc6507fb))
* adds property Dataset.extractable_features ([961532d](https://github.com/DCMLab/dimcat/commit/961532d9dc19cb0eb83a23732a54129cb7eac10a))
* adds Result.make_ranking_table() and allows Results.combine_results() to re-combine if the groups are a subset of the columns ([4d0f313](https://github.com/DCMLab/dimcat/commit/4d0f313f25960620c2fb7eac88fa78e9a1921c89))
* adds two new methods to Package, .get_resource() and .replace_resource() ([8f74b10](https://github.com/DCMLab/dimcat/commit/8f74b10f2c4ab5d5e50ec1664e0e758b753f0571))
* both Dataset and DimcatResource now have a method .apply_steps() that creates a Pipeline and applies it, and .apply_step() that creates a step directly, without turning it into a Pipeline first. DimcatResources no get methods .make_bar_plot(), .make_bubble_plot(), and .make_pie_chart() which follow the same principle as .plot() and .plot_grouped(): Apply the specified PipelineSteps or the default Analyzer and call the respective method on the result. ([eef8e1b](https://github.com/DCMLab/dimcat/commit/eef8e1bd7a1437de82da6997fd682776dc93436f))
* enables Result.make_pie_chart() ([e156d45](https://github.com/DCMLab/dimcat/commit/e156d4518111a72358141582123bab7af756585e))
* enables two new BassNotesFormat values: SCALE_DEGREE_MAJOR and SCALE_DEGREE_MINOR ([d52b8b0](https://github.com/DCMLab/dimcat/commit/d52b8b0811990f2e9e2699d62d514817b2916b29))
* factors out MappingGrouper() from CustomPieceGrouper() ([4a1ebf5](https://github.com/DCMLab/dimcat/commit/4a1ebf55966d90c049744325510a291c4ffe04e2))
* feature pass their format arg to super().__init__(); removes unused format values (to be extended later) and properly integrates NotesFormat and HarmonyLabelsFormat. Also, the Notes feature does not ignore arguments 'merge_ties' and 'weight_grace_notes' anymore but acts on them. ([d3daaef](https://github.com/DCMLab/dimcat/commit/d3daaef7664463a5f04189fc3374da10ee476b17))
* introduces DimcatResource.from_resource_and_dataframe() to copy properties of existing resource but detach the new resource and set a new dataframe (usually a transformation of the previous one) ([4ad7a6b](https://github.com/DCMLab/dimcat/commit/4ad7a6bb99303b67ca988fa2d0b5c996c34551c7))
* introduces font_size argument to all plotting functions for convenience [saves one to type, e.g., layout=dict(font=dict(size=30)) ] ([ecb7817](https://github.com/DCMLab/dimcat/commit/ecb78176c2d93ed44d56648bcea1dcda240629b6))
* makes all enum values specified in DimcatConfigs case-insensitive by subclassing Marshmallow's enum field and having 'by_value' default to True. Likewise, get_class() now accepts dtypes in case-insensitive manner ([45cb9d0](https://github.com/DCMLab/dimcat/commit/45cb9d00875d77d36326f15348ddeb7de0f6aef0))
* moves CADENCE_COLORS to dimcat.plotting ([cce4f38](https://github.com/DCMLab/dimcat/commit/cce4f38de7b2c0e3ecdcc05770e38ddd175aaa82))
* Result.combine_results() now returns a new Result object. The creationg of the combined dataframe has been moved to ._combine_results(), which is used by the plotting methods ([8251b7a](https://github.com/DCMLab/dimcat/commit/8251b7a960d3cc934622964d38547a8989b0e310))


### Bug Fixes

* .plot_grouped() shows bar plot when no grouper has been applied (and no grouping level has been requested) or a bubble plot in all other cases. This includes adding the arguments df, x_col, and y_col to Result.make_bubble_plot(). ([b37d8fd](https://github.com/DCMLab/dimcat/commit/b37d8fdef27de188a8bef503ead9cfe0d1b78bc7))
* adapts mwd notebook to be showing grouped plots ([b438d33](https://github.com/DCMLab/dimcat/commit/b438d336e9eb8e777c2b8e0ce31d5c812eb98d00))
* adds missing column names ([4252ad9](https://github.com/DCMLab/dimcat/commit/4252ad9aed610d2598d0a9ff513e673bfb2647cb))
* allows any FeatureSpecs for the 'features' argument of the FeatureProcessingStep (i.e., in its Marshmallow schema) ([59a5c71](https://github.com/DCMLab/dimcat/commit/59a5c710dcbcf2a9282cb7eb44c668a9fb0adfcc))
* avoids duplicating convenience columns ([d91a0d5](https://github.com/DCMLab/dimcat/commit/d91a0d5fd69efde66b3cbc32536626458828eec5))
* colorlover is not an optional dependency anymore ([f8b93a0](https://github.com/DCMLab/dimcat/commit/f8b93a0e631396cef040e2410b3cb268d17887ec))
* DcmlAnnotations was missing "chord" column ([669be4b](https://github.com/DCMLab/dimcat/commit/669be4b992177f409e80ec4d6781203ffb1ed8c9))
* DimcatResource._extract_feature() makes use of all config options ([2eda6bf](https://github.com/DCMLab/dimcat/commit/2eda6bf4bf618474d382840bf8e0dfc078686231))
* includes DimcatResource._drop_rows_with_missing_values() ([2569c97](https://github.com/DCMLab/dimcat/commit/2569c976fedc68541f6bb4dcb337abe9478b5b29))
* pass x_col and y_col to plotting.update_figure_layout() and set any axis called 'piece' to type "categorical" to avoid automatic conversion to dates ([a54b66e](https://github.com/DCMLab/dimcat/commit/a54b66e520b040b1e6d769f599fc4b3573bfadc1))
* removes column name in faceted plots ([68bd7a7](https://github.com/DCMLab/dimcat/commit/68bd7a7642bffe1de2df2a26af81b639375cbd70))
* replaces segmenting approach of CadenceLabels() (incl. label-to-label durations) with bare occurrences ([c326bf2](https://github.com/DCMLab/dimcat/commit/c326bf2d50b077afe7fb0c0821c6fc408f7162a0))
* ResourceTransformation() should not copy the resource's "descriptor_filename" because an existing descriptor might not apply anymore ([4982b4d](https://github.com/DCMLab/dimcat/commit/4982b4d37d3b12f6619950c1e1b5cd965521bd52))
* superclass FeatureProcessingStep should not reject any types if _allowed_features is None ([26a506e](https://github.com/DCMLab/dimcat/commit/26a506edf407ae3e4e79a8905d10fada86d9bd34))
* type annotations for .get_metadata() ([ce1bdfb](https://github.com/DCMLab/dimcat/commit/ce1bdfbe477397a428d4df4daa636a47ce76e596))
* when copying or transforming, get kwargs from existing DimcatResource and pass them to the respective constructor ([e1c0cda](https://github.com/DCMLab/dimcat/commit/e1c0cda366b0f17c5e81941897b40c150716dc2e))


### Documentation

* adds more text to Quickstart notebook and adds it to the manual ([682a642](https://github.com/DCMLab/dimcat/commit/682a64280f1c0147758450d79be4da5823075eab))
* moves mwe.md notebook to manual as quick.md ([4fdb284](https://github.com/DCMLab/dimcat/commit/4fdb284656b3db4f995733560a717fef59939e73))

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
