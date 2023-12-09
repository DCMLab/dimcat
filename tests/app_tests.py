"""Testing the functionality required by the Music History Explorer app written in Dash."""
import marshmallow as mm
import pytest
from dimcat import Pipeline, get_class, get_schema
from dimcat.data.resources import DimcatIndex, FeatureName
from dimcat.steps.analyzers.base import AnalyzerName

EXCLUDED_ANALYZERS = [  # to be synchronized with AnalyzerConstants.analyzer_to_hide
    "Analyzer"
]  # to be synchronized with AnalyzerConstants.analyzer_to_hide
EXCLUDED_FEATURES = [  # to be synchronized with AnalyzerConstants.feature_to_hide
    # abstract features
    "Metadata",
    "Annotations",
    # untested features
    "Articulation",
    "CadenceLabels",
    "KeyAnnotations",
    "Measures",
]

# region Interface


def check_field_metadata(field_metadata, dimcat_object_name, field_name):
    if len(field_metadata) == 0:
        print(f"Field {dimcat_object_name}.Schema.{field_name} has no metadata")
        assert field_metadata
    if field_metadata.get("expose", True):
        description = field_metadata.get("description")
        if not description:
            print(f"Field {dimcat_object_name}.Schema.{field_name} has no description")
            assert description


@pytest.fixture(
    params=AnalyzerName,
)
def analyzer_name(request):
    if request.param in EXCLUDED_ANALYZERS:
        pytest.skip(f"{request.param} is excluded")
    return request.param


def test_analyzer_fields(analyzer_name):
    schema = get_schema(analyzer_name)
    for field_name, field in schema.declared_fields.items():
        check_field_metadata(field.metadata, feature_name, field_name)


@pytest.fixture(
    params=FeatureName,
)
def feature_name(request):
    if request.param in EXCLUDED_FEATURES:
        pytest.skip(f"{request.param} is excluded")
    return request.param


def test_feature_fields(feature_name):
    schema = get_schema(feature_name)
    for field_name, field in schema.declared_fields.items():
        check_field_metadata(field.metadata, feature_name, field_name)


def prettify_labels(name: str):
    """Prettify the labels
    :param name: the name to for the label
    :return: the prettified string"""
    return name.replace("_", " ").capitalize()


def test_update_feature_dropdown(analyzer_name):
    """
    Populate the features dropdown with analyzer-specific features
            :param analyzer_name: the name of the analyzer
            :return: options list for the feature dropwdown


    .. code-block:: python

        @app.callback(
            Output("features_choice", "options"),
            Input("analyzer_choice", "value"),
        )
        def update_feature_dropdown(analyzer_name):
            options = [{"label": prettify_labels(name), "value": name} for name in FeatureName if
                     name not in feature_to_hide]
            if analyzer_name != "" and analyzer_name is not None:
                analyzer = get_class(analyzer_name)
                if (analyzer._allowed_features is not None and len(analyzer._allowed_features) != 0):
                    options = [{"label": prettify_labels(f.name), "value": f.name} for f in analyzer._allowed_features]
            return options
    """
    options = [
        {"label": prettify_labels(name), "value": name}
        for name in FeatureName
        if name not in EXCLUDED_FEATURES
    ]
    analyzer = get_class(analyzer_name)
    if analyzer._allowed_features is not None and len(analyzer._allowed_features) != 0:
        options = [
            {"label": prettify_labels(f.name), "value": f.name}
            for f in analyzer._allowed_features
        ]
    assert len(options) > 0


# endregion Interface
# region Analyzing


@pytest.fixture()
def analyzer_config(analyzer_name):
    return dict(dtype=analyzer_name)


@pytest.fixture()
def feature_config(feature_name):
    return dict(dtype=feature_name)


@pytest.fixture()
def grouped_pieces(dataset_from_single_package):
    input_package = dataset_from_single_package.inputs.get_package()
    piece_index = input_package.get_piece_index()
    n_groups = 4
    grouping = {f"group_{i}": piece_index.sample(i) for i in range(n_groups)}
    grouped_pieces = DimcatIndex.from_grouping(grouping)
    print(len(grouped_pieces))
    return grouped_pieces


@pytest.fixture()
def grouper_config(grouped_pieces):
    return dict(dtype="CustomPieceGrouper", grouped_units=grouped_pieces)


def test_analyze(
    dataset_from_single_package, analyzer_config, feature_config, grouper_config
):
    """

    Analyze the groups
            :param analyzer_name: The name of the analyzer to use
            :param feature_name: The name of the feature to use
            :param analyzer_config: The configuration of the analyzer
            :param feature_config: The configuration of the feature
            :param groups_content: The store data of the groups
            :param full_dataframe: The complete dataframe used by the app
            :param dataset_from_single_package: The dataset used by the app

    .. code-block:: python

        def analyze(analyzer_name,feature_name,analyzer_config,feature_config,groups_content,full_dataframe,dataset):
            grouped_pieces = create_groups_analyzed(groups_content, full_dataframe)

            step_configs = [
                dict(dtype="FeatureExtractor", features=[feature_config]),
                analyzer_config,
                dict(dtype='CustomPieceGrouper', grouped_pieces=grouped_pieces)
            ]

            try:
                pl = dc.Pipeline.from_step_configs(step_configs)
            except mm.ValidationError as e:
                try:
                    # Remove the unnecessary parts of the error message
                    error_message = e.messages[0].split("options:")[2][3:-4]
                except IndexError:
                    error_message = str(e)
                return create_error_message(error_message_config_validation + error_message)
            result_process = pl.process(dataset)
            result = result_process.get_result()

            return get_graph_accordion_item(result.plot(), analyzer_name, feature_name, analyzer_config, feature_config)
    """
    step_configs = [
        dict(dtype="FeatureExtractor", features=[feature_config]),
        grouper_config,
        analyzer_config,
    ]
    analyzer = get_class(analyzer_config["dtype"])
    if (
        analyzer._allowed_features is not None
        and feature_config["dtype"] not in analyzer._allowed_features
    ):
        pytest.skip(
            f"{analyzer_config['dtype']} cannot process {feature_config['dtype']}"
        )

    try:
        pl = Pipeline.from_step_configs(step_configs)
    except mm.ValidationError as e:
        try:
            # Remove the unnecessary parts of the error message
            error_message = e.messages[0].split("options:")[2][3:-4]
        except IndexError:
            error_message = str(e)
        msg = "Some required fields are missing or wrong: " + error_message
        raise mm.ValidationError(msg) from e
    result_process = pl.process(dataset_from_single_package)
    result = result_process.get_result()
    result.plot_grouped()
    result.plot()


# endregion Analyzing
