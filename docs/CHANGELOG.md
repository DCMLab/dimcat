# Changelog

## [3.0.0](https://github.com/DCMLab/dimcat/compare/v2.3.0...v3.0.0) (2023-12-13)


### ⚠ BREAKING CHANGES

* eliminates .apply_steps() in favour of a single .apply_step(*step), that is, with variadic argument. For backward compatibility, the method still accepts a single list or tuple

### Features

* adds four additional columns to HarmonyLabels and BassNotes which contain the (main) chord tones expressed as scale degrees ([396dce9](https://github.com/DCMLab/dimcat/commit/396dce9ad3aa036d13f4e623a5d2954e98dbbdba))
* adds Result.compute_entropy() and Transitions.compute_information_gain() ([c1257a8](https://github.com/DCMLab/dimcat/commit/c1257a84e7073544ec252c6ae3e5a9fe47e8c4bc))
* AdjacencyGroupSlicers now process the required_feature during .fit_to_dataset(), store it as property .slice_metadata and join it onto any processed Metadata object. In the future, there could also be a mode where this metadata is joined onto any processed feature. ([cea586e](https://github.com/DCMLab/dimcat/commit/cea586e6c318da224382a7119baedfb6be0add22))
* empowers NgramTable to make_bi/ngram_tables and NgramTuples with components made up from different columns and with individual join_str and fillna settings ([c8488cf](https://github.com/DCMLab/dimcat/commit/c8488cf9d3bb22b3480af021c24af2add0b96cb6))
* enables adding context_columns for the NgramTable's methods .get_bi/ngram_tuples() and get_bi/ngram_table(). The NgramAnalyzer therefore adds the relevant column names in post-processing. ([fe7ee3a](https://github.com/DCMLab/dimcat/commit/fe7ee3a969712528849dfbedfa28f6914a6e2e6c))
* enables applying Slicers to Metadata by joining them on the SliceIntervals (DimcatIndex) ([a4c3929](https://github.com/DCMLab/dimcat/commit/a4c39291cdfc09f5d99e50d072eaa49ebb625274))
* enables dropping ngram rows which include/correspond to terminals ([89a2552](https://github.com/DCMLab/dimcat/commit/89a25522ff1d02e94e549ccedd1935c931ea2a8c))
* enables the detailed control of terminals which may differ for different n-gram components (except the first one). ([f6a807f](https://github.com/DCMLab/dimcat/commit/f6a807f4417ccb706343b832ef5ca764ac82d709))
* HarmonyLabels and BassNotes features now come with an intervals_over_bass and (for the former) with an intervals_over_root column ([be8d06d](https://github.com/DCMLab/dimcat/commit/be8d06d461dc1cabf9802b9af39cac8d845df1a1))
* includes "root" as auxiliary column for BassNotes ([3f7bd35](https://github.com/DCMLab/dimcat/commit/3f7bd35db460011cb362c812f12dbdc3c9026703))
* makes the 'data' argument to PipelineStep.process() a variadic one, too (concordant with .apply_step()), while still accepting a single argument that can be a list or tuple ([7a37aaa](https://github.com/DCMLab/dimcat/commit/7a37aaaf1bb7607035292efcc072a649b69c3f77))
* Metadata.get_composition_years() now with 'group_cols' parameter to compute composition year means of groups (e.g. corpora) ([fef9860](https://github.com/DCMLab/dimcat/commit/fef986000a4cbd2eaf37d89a61d05442fe64ee5b))
* methods .make_ngram_table() and .make_bigram_table() of NgramTable now actually return a new NgramTable, whereas the previous functions of that name (which returned dataframes) have been renamed to .make_bigram_df() and .make_ngram_df(). ([8dbff20](https://github.com/DCMLab/dimcat/commit/8dbff202f8d71c54ecbbf2bcc32e00d0363a45f4))
* NgramTable gets the convenience method .compute_information_gain() to skip an intermediate call to .get_transitions() ([5b37414](https://github.com/DCMLab/dimcat/commit/5b374141cdb55216fa86cbb668b7867fbead2a6f))
* NgramTable._get_transitions() is cached and now complete with the terminal_symbols argument ([bd12568](https://github.com/DCMLab/dimcat/commit/bd1256825f7ddc128459519068ce6d2955f73eac))
* reduces the amount of parentheses in n-grams by not turning 'single' components (with only one column) into tuples ([02f91d4](https://github.com/DCMLab/dimcat/commit/02f91d40d9935996e0e12625801ab1e646d54fb2))
* streamlines turning n-grams into strings and allows for doing it recursively (useful when columns making up n-gram components contain tuples themselves) ([745df2e](https://github.com/DCMLab/dimcat/commit/745df2ed8471ceafc8eb3d41028e205161724f20))


### Bug Fixes

* adapts scipy.stats.entropy() to fix bug caused by pd.Float64Dtype ([4938170](https://github.com/DCMLab/dimcat/commit/4938170539e6072bcd9795797ef29c863cd63f60))
* allow DimcatResource.filter_index_level() to just drop the level without filtering rows ([5c07d97](https://github.com/DCMLab/dimcat/commit/5c07d9794ca1c5beed49aa7a282f0b9e5e9a8e04))
* applying a Grouper needs to be an inner join. Also, the index levels should come in systematic order, first the grouper levels, then the remaining ones ([8f80fc2](https://github.com/DCMLab/dimcat/commit/8f80fc2bc07582ea6584c2b494384daf5ac1199a))
* enables (de-)serialization for Filter objects ([976c179](https://github.com/DCMLab/dimcat/commit/976c1797561d1d2076fd1a1479da64815d247fa9))
* fills up missing 'quarterbeats_all_endings' column for older parts of the dataset ([390e0a5](https://github.com/DCMLab/dimcat/commit/390e0a5ce99c3b3f77ddd9d0d9be903fd854caad))
* Groupers that use metadata now should use Dataset.get_metadata(raw=True) ([5d35b20](https://github.com/DCMLab/dimcat/commit/5d35b20044a24a47cfa3848dc4d2dbd226ba4a8f))
* grouping by a single level that contains tuples resulted in several levels in the resulting MultiIndex; this fix applied for completeness before the whole function is simplified ([0ed6091](https://github.com/DCMLab/dimcat/commit/0ed6091e323074670281c7389603b4605412e494))
* NgramTable.get_default_analysis() returns Transitions ([b90f0ae](https://github.com/DCMLab/dimcat/commit/b90f0aea80240077085f2b85bafacc3124222be4))
* omit duplicate computation of 'proportions' by Transitions._sort_combined_result() ([2f09bbb](https://github.com/DCMLab/dimcat/commit/2f09bbb600919c7ccb80749306ea449dcfc2b238))
* raise NotImplementedError when trying to use convenience methods directly on Transitions object ([7cf61c4](https://github.com/DCMLab/dimcat/commit/7cf61c47b07e91e928c749836469f7e4d4b7e5f3))
* re-inserts missing import ([02bd96c](https://github.com/DCMLab/dimcat/commit/02bd96c7c26711e2fb634ac6cd34be2c42517ac6))
* singular ngram_components should also become strings (even if they are not joined on 'join_str') ([3987162](https://github.com/DCMLab/dimcat/commit/398716226ce3ca3e9b0c421a68d15583f25b909e))
* when an index level is dropped, make sure to remove it from the default_groupby ([260f8f1](https://github.com/DCMLab/dimcat/commit/260f8f1ae011a917026ea1f4d20e0c50cc8a7035))
* when applying a Filter with drop_level=True, do not turn a Dataset into a GroupedDataset (as per virtue of the respective parent Grouper) ([937002c](https://github.com/DCMLab/dimcat/commit/937002c9eed97418bbc2d75267e9626289eb1eb9))
* when Counter is used with smallest_unit=GROUP, it recurs to self.compute() ([737b6a6](https://github.com/DCMLab/dimcat/commit/737b6a6051f7256596adf8d7f0d665de13962e12))


### Reverts

* eliminates .apply_steps() in favour of a single .apply_step(*step), that is, with variadic argument. For backward compatibility, the method still accepts a single list or tuple ([fab8e13](https://github.com/DCMLab/dimcat/commit/fab8e13fafc895ee6338aad2c499dfc497ebfabc))

## [2.3.0](https://github.com/DCMLab/dimcat/compare/v2.2.0...v2.3.0) (2023-12-09)


### Features

* all schemas retrieved via the .schema or .pickled_schema property allow for loading dicts without 'dtype' key by assuming their own dtype as default ([9ff060e](https://github.com/DCMLab/dimcat/commit/9ff060ea0718242ce7a3b05cad50163bd3c5dc58))
* new category of objects: Filters. They extend any Grouper by adding the init args 'keep_values', 'drop_values', and 'drop_level' to it. They use these arguments to post-process any resource first processed by the corresponding grouper. This required renaming the relatively new HasCadenceAnnotations and HasHarmonyLabels to HasCadenceAnnotationsGrouper and HasHarmonyLabelsGrouper, to differentiate them from the new HasCadenceAnnotationsFilter and HasHarmonyLabelsFilter. The other two filters that have been implemented so far are the CorpusFilter and the PieceFilter. As an aside, Groupers do not complain anymore when they are applied to a resource that has already been grouped by a Grouper of the same type. If the grouping level exists but isn't the first one, it is systematically made the first one. This applies, by extension, to the Filters (for now) ([ec3d1f7](https://github.com/DCMLab/dimcat/commit/ec3d1f7980b5a18a01b0cdb37e93823f1549236d))


### Bug Fixes

* adapts NgramAnalyzer's init args & schema ([3e51f97](https://github.com/DCMLab/dimcat/commit/3e51f979d06ee275672ffba8b36e4ad9cff51e61))
* align_with_grouping() did not work for NgramTables because pandas prevents merge with diverging column nlevels, even if one of the sides has no columns ([e51625f](https://github.com/DCMLab/dimcat/commit/e51625fd6b7fd004ff1eb7afeb93139f5d728adc))
* allows passing a list of list (instead of a list of tuples) to DimcatIndex.from_tuples(), useful for de-serializing from JSON ([0cff3c1](https://github.com/DCMLab/dimcat/commit/0cff3c1bd0405b9dda786cb1d8c14e1f0e4c7779))
* extends app_tests.test_analyze() to the actual plotting; warns about non-Analyzer PipelineSteps applied after an Analyzer ([72ef210](https://github.com/DCMLab/dimcat/commit/72ef210eb007db3ef5a5158d7890ddf3add6ee38))
* facet titles be strings ([ed185f7](https://github.com/DCMLab/dimcat/commit/ed185f792115ecee571e0b24a903a8e2604d35cd))
* improves (de-)serialization of DimcatIndex and, by extension, the MappingGroupers' 'grouped_units' field ([f59673d](https://github.com/DCMLab/dimcat/commit/f59673da2e1a3075d8881d51cc5e3ce60c84e7db))
* parses music21.key.KeySignature the same way as usic21.key.Key ([5aa7902](https://github.com/DCMLab/dimcat/commit/5aa790269d43c18f0ebfeebe1c0a097e5b1029a6))
* the frictionless workaround for copying a resource with no path specified is now complete ([5d1426d](https://github.com/DCMLab/dimcat/commit/5d1426d33dbe81f7f97dec2edb967895fe82883b))
* the frictionless workaround for copying a resource with no path specified is now complete ([98ee01d](https://github.com/DCMLab/dimcat/commit/98ee01deda6f6e1e92a88bea416fc4b210794e37))

## [2.2.0](https://github.com/DCMLab/dimcat/compare/v2.1.0...v2.2.0) (2023-12-07)


### Features

* adds HasHarmonyLabels grouper ([4fa92de](https://github.com/DCMLab/dimcat/commit/4fa92debd399a7649ed59c59c98bc334d1228b95))
* enables .get_feature("metadata") for Dataset and DimcatPackage which, in return, enables Dataset.get_metadata(raw=False) (default), i.e. returning a processed Metadata feature (old behaviour, i.e. without processing, via Dataset.get_metadata(raw=True)) ([731c4d1](https://github.com/DCMLab/dimcat/commit/731c4d1385ba35beaad374c777274ab2331caf67))


### Bug Fixes

* align_with_grouping() makes sure to be unpacking DimcatIndex ([a279691](https://github.com/DCMLab/dimcat/commit/a279691f940c2fd220386cec835e25f768009081))
* Analyzer.Schema() adapted ([57748ca](https://github.com/DCMLab/dimcat/commit/57748caacaaf7172aaaf1c8052e68c540fc5577c))
* DimcatResource.from_resource_and_dataframe() also detaches new resource from filepath, if necessary ([6665fdd](https://github.com/DCMLab/dimcat/commit/6665fddaf60c991be505df7b9bc86731e9b8fbd8))

## [2.1.0](https://github.com/DCMLab/dimcat/compare/v2.0.0...v2.1.0) (2023-12-07)


### Features

* adds 'dimension_column' as argument for all Analyzers; enables default_analyzer for Metadata ([d192a07](https://github.com/DCMLab/dimcat/commit/d192a074af0b1b4799aadd06e222ef8fd8380a8d))
* adds convenience module `dimcat.enums` for easily importing any enum from DiMCAT. ([7cd3a3f](https://github.com/DCMLab/dimcat/commit/7cd3a3fc11d07d37d3dd3a3de5437334798e25de))
* adds PieceGrouper ([55c6d54](https://github.com/DCMLab/dimcat/commit/55c6d546635440c7d623d0db23ad75e1c990bd37))
* enable .make_ranking_table() for NgramTable (convenience for calling .make_ngram_tuples() first) ([5a788d5](https://github.com/DCMLab/dimcat/commit/5a788d5ee96b1ad66a13944c49177a309d005dc8))
* enables group_cols and group_modes for bubble_plots, too ([0d8ba17](https://github.com/DCMLab/dimcat/commit/0d8ba177542e60499f029f4110022e96559c3b0c))
* includes the UnitOfAnalysis enum as 'group_cols' argument for Result's methods ([efd7fdb](https://github.com/DCMLab/dimcat/commit/efd7fdb625b0c7219718d7ea2b397e4231c6c836))
* introduces new HarmonyLabelsFormat "ROMAN_REDUCED" ([0a08952](https://github.com/DCMLab/dimcat/commit/0a089525379b8bf80e6da75daea9fb4a3d994828))
* NgramTable.get_transitions() returns new result type Transitions ([0723fe1](https://github.com/DCMLab/dimcat/commit/0723fe15bad6013f60b0415ae0db18053a984c68))
* NgramTable.make_ngram_tuples() now actually returns tuples, not tables (which are retrieved via .make_ngram_table()). They come as a new Result type, NgramTuples, which also allows for .make_ranking_table() ([3746437](https://github.com/DCMLab/dimcat/commit/3746437c7084fcc4f7705525f8a39443980804d4))
* NgramTable() uses the new Transitions for both .plot() and .plot_grouped() ([b12b130](https://github.com/DCMLab/dimcat/commit/b12b1302a0655f415ab77eabb72b0e18bbe67d12))
* PieceGrouper and CorpusGrouper move the respective index level to level 0 ([1b7f436](https://github.com/DCMLab/dimcat/commit/1b7f436746cfd5c709bbe7c619e0de3ff558e852))
* Transitions result type plots methods return Plotly heatmaps ([3df7880](https://github.com/DCMLab/dimcat/commit/3df78807b7c1de1be8b57534d460b918820915a9))


### Bug Fixes

* base.resolve_object_spec() needs to check if config first, then if DimcatObject ([8c8c4d2](https://github.com/DCMLab/dimcat/commit/8c8c4d2955ae4eff23fac762b2d84799e5703e8a))
* do not convert "count" column to "Int64" by default (because of Plotly bug); instead convert integer columns when making ranking tables to prevent counts coming as floats ([59bd92a](https://github.com/DCMLab/dimcat/commit/59bd92a7ee96be0e6daa5f122dc4f470d66e9a97))
* Pipeline calls step.process_resource() instead of ._process_resource() because otherwise the call to .check_resource() is skipped ([024bf65](https://github.com/DCMLab/dimcat/commit/024bf65ec50587f02a9c83f28eb9855d3ccbf173))


### Documentation

* moves error to dedicated errors.md notebook. fixes [#61](https://github.com/DCMLab/dimcat/issues/61) ([390b76e](https://github.com/DCMLab/dimcat/commit/390b76ea3788206a666c147b36ebe9ff6c71b71c))

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
* both Dataset and DimcatResource now have a method .apply_step(). DimcatResources no get methods .make_bar_plot(), .make_bubble_plot(), and .make_pie_chart() which follow the same principle as .plot() and .plot_grouped(): Apply the specified PipelineSteps or the default Analyzer and call the respective method on the result. ([eef8e1b](https://github.com/DCMLab/dimcat/commit/eef8e1bd7a1437de82da6997fd682776dc93436f))
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
