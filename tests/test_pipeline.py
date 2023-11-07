from dimcat.base import DimcatConfig
from dimcat.data.resources.dc import PieceIndex
from dimcat.steps.base import FeatureProcessingStep
from dimcat.steps.extractors.base import FeatureExtractor
from dimcat.steps.groupers.base import CustomPieceGrouper
from dimcat.steps.pipelines.base import Pipeline


def test_pipeline():
    pl = Pipeline([FeatureProcessingStep()])
    as_dict = pl.to_dict()
    print(as_dict)
    conf = DimcatConfig(as_dict)
    print(conf)
    pl_copy = conf.create()
    assert pl == pl_copy
    assert pl.steps == pl_copy.steps
    print(pl.steps[0])


def test_commutativity(dataset_from_single_package):
    fex = FeatureExtractor("notes")
    notes_extracted = fex.process(dataset_from_single_package)
    notes = notes_extracted.get_feature("notes")
    piece_id_tuples = PieceIndex.from_resource(notes).to_series()
    grouping = {i: piece_id_tuples.sample(3) for i in range(3)}
    grouper = CustomPieceGrouper.from_grouping(grouping)
    pl1 = Pipeline([fex, grouper])
    pl2 = Pipeline([grouper, fex])
    ds1 = pl1.process(dataset_from_single_package)
    ds2 = pl2.process(dataset_from_single_package)
    ds3 = grouper.process(notes_extracted)
    assert set(ds1.pipeline) == set(ds2.pipeline)
    assert set(ds1.pipeline) == set(ds3.pipeline)
    notes1 = ds1.get_feature("notes")
    notes2 = ds2.get_feature("notes")
    notes3 = ds3.get_feature("notes")
    assert notes1 == notes2
    assert notes1 == notes3
    print(notes1.df)
    print(
        f"notes1.status: {notes1.status!r}\nnotes2.status: {notes2.status!r}\nnotes3.status: {notes3.status!r}"
    )
