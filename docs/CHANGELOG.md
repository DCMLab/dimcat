# Changelog

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

## Changelog

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
