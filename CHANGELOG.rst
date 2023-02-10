=========
Changelog
=========

Version 0.3.0
=============

* new slicers: MeasureSlicer, ChordFeatureSlicer
* LocalKeySlicer has become a special case of ChordFeatureSlicer
* Analyzers call method ``post_process()`` on the processed data. Introduced for enabling:
* Overall consistent behaviour and naming of MultiIndex levels for all DataFrames
* Corpus objects
  * now come with the new method ``get_facet()`` that collects the results from ``iter_facet()``
  * come with a more performant slicing facility by slicing everything at once using the optimized function
    ``ms3.utils.overlapping_chunk_per_interval()`` (requires ms3 > 0.5.3)
  * consistently store the sequence of applied PipelineSteps, the first being the most recently applied
  * has a new method ``get_previous_pipeline_step`` allowing for retrieving one of them either by position or by type
* PitchClassVectors allows for new settings:
  * ``weight_grace_durations`` gives grace notes durations which by default are zero
  * ``include_empty=True`` adds empty rows for segments without note occurrence
  * all of which can be used via the commandline tool ``dimcat pcvs``
* ChordSymbolBigrams now
  * comes with new setting ``dropna`` defaulting to True and excluding transitions from and to missing values
  * checks before processing if LocalKeySlicer has been applied before
  * checks before computing (staticmethod) if all transitions actually happen within the same localkey
* All PipelineSteps how have a ``filename_factory()`` method
* TSVWriter now
  * automatically creates meaningful filenames from all applied PipelineSteps by calling their ``filename_factory()``
  * comes with the new settings ``round`` and ``fillna``
* Unittests store and check the git revision of the ``unittest_metacorpus`` repo, for which they are supposed to work

Version 0.2.0
=============

* new slicers: NoteSlicer, LocalKeySlicer
* new groupers: PieceGrouper, CorpusGrouper, YearGrouper, ModeGrouper
* new filter: IsAnnotatedFilter
* new writer: TSVWriter
* installs system-wide command ``dimcat pcvs``
* consistent naming of DataFrame MultiIndex levels
* more consistent interface, more abstractions


Version 0.1.0
=============

* new analyzers: TPCrange, PitchClassVectors, ChordSymbolUnigrams, ChordSymbolBigrams
* installs system-wide commands ``dimcat unigrams`` and ``dimcat bigrams`` for creating TSV files
