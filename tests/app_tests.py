"""Testing the functionality required by the Music History Explorer app written in Dash."""
import pytest
from dimcat import get_class, get_schema
from dimcat.data.resources import FeatureName
from dimcat.steps.analyzers.base import AnalyzerName

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
    if request.param == "Analyzer":  # AnalyzerConstants.analyzer_to_hide
        pytest.skip("Analyzer is an abstract class")
    return request.param


def test_analyzer_fields(analyzer_name):
    schema = get_schema(analyzer_name)
    for field_name, field in schema.declared_fields.items():
        check_field_metadata(field.metadata, feature_name, field_name)


@pytest.fixture(
    params=FeatureName,
)
def feature_name(request):
    if request.param == "Metadata":  # AnalyzerConstants.feature_to_hide
        pytest.skip("Metadata not part of processable features (for now).")
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
        if name not in ["Metadata"]
    ]  # AnalyzerConstant.feature_to_hide
    analyzer = get_class(analyzer_name)
    if analyzer._allowed_features is not None and len(analyzer._allowed_features) != 0:
        options = [
            {"label": prettify_labels(f.name), "value": f.name}
            for f in analyzer._allowed_features
        ]
    assert len(options) > 0


# endregion Interface
