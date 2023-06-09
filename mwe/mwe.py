import os
from pprint import pformat

import marshmallow as mm
from dimcat.base import get_class, get_schema
from dimcat.data.dataset.base import Dataset, DimcatConfig
from dimcat.data.dataset.processed import AnalyzedDataset
from dimcat.data.resources.features import FeatureName
from dimcat.steps.analyzers import Counter

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # initialize the dataset
    dataset = Dataset()
    dataset.load_package("dcml_corpora.datapackage.json")

    # get the metadata
    metadata = dataset.get_metadata()

    # get feature names from the enum
    for feature_name in FeatureName:
        assert feature_name.name == feature_name.value

    # feature names correspond to names of DimcatResource classes and there are two ways of getting constructors:
    # 1. from the Enum member
    notes_constructor = FeatureName(
        "nOtEs"
    ).get_class()  # spelling of string does NOT matter
    # 2. or via the get_class() function for any DimcatObject (which the enum member uses under the hood)
    notes_constructor = get_class("Notes")  # spelling of string DOES matter

    # to get the default options of a feature we can serialize an instance
    empty_notes_feature = notes_constructor()
    print(
        f"Default options of a Notes resource:\n"
        f"{pformat(empty_notes_feature.to_dict())}"
    )

    # other serialization methods are .to_json(), .to_json_file(), and .to_config()
    # the latter is used to create a DimcatConfig object which can easily be deserialized:
    notes_config = empty_notes_feature.to_config()
    copied_notes_feature = notes_config.create()
    assert copied_notes_feature == empty_notes_feature

    # A DimcatConfig works like a dictionary with the difference that the key 'dtype' needs to be the name of the
    # DimcatObject it describes. It uses that object's schema to validate all given options.
    notes_config_from_scratch = DimcatConfig(dtype="Notes")
    notes_config_from_scratch["format"] = "NAME"
    try:
        notes_config_from_scratch["format"] = "this is not a valid format"
        raise RuntimeError("Setting an invalid option should raise an error.")
    except mm.ValidationError:
        pass

    # A DimcatConfig can be partially define a schema, creating an object from it will use the object's defaults
    notes_feature_from_scratch = notes_config_from_scratch.create()
    assert notes_feature_from_scratch == empty_notes_feature

    # to retrieve the available options, every DimcatObject exposes the class attribute .schema
    notes_schema = notes_constructor.schema
    assert (
        notes_schema == empty_notes_feature.schema == get_schema("Notes")
    )  # the latter uses get_class()

    # the types of the fields are from the marshmallow.fields module
    for name, field in notes_schema.declared_fields.items():
        if isinstance(field, mm.fields.Enum):
            info = (
                f"{name!r}: {field.enum!r} field with choices {field.choices_text!r}."
            )
        else:
            info = f"{name!r}: {type(field)} field."
        info += (
            f"\n\tload_default={field.load_default}"
            f"\n\tvalidate={field.validate}"
            f"\n\tmetadata={pformat(field.metadata, sort_dicts=False)}"
        )
        print(info)

    # two ways of creating the identically configured Counter analyzer:
    counter_config = DimcatConfig(
        dtype="Counter"
    )  # UPDATE: if the 'features' argument is not specified,
    # analyzer will be applied to all loaded features
    counter_config[
        "features"
    ] = notes_config  # notes_config created from empty_notes_feature above
    a_counter = counter_config.create()
    b_counter = get_class("Counter")(
        features=empty_notes_feature
    )  # passing the feature works as well
    assert a_counter == b_counter
    # the feature arguments are always converted to DimcaConfig objects which have the advantage
    # that they can be partially defined. These two are equivalent ways of saying "any notes regardless their format"
    c_counter = Counter(features="notes")
    d_counter = Counter(features=DimcatConfig(dtype="Notes"))
    assert c_counter == d_counter

    analyzed_dataset = d_counter.process(dataset)
    assert isinstance(analyzed_dataset, AnalyzedDataset)
    result = analyzed_dataset.get_result()
    fig = result.plot()
    print(fig)
