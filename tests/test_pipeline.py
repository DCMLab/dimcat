from dimcat.base import DimcatConfig, PipelineStep
from dimcat.groupers.base import CustomPieceGrouper
from dimcat.pipeline import Pipeline


def test_pipeline():
    pl = Pipeline([PipelineStep(), CustomPieceGrouper()])
    as_dict = pl.to_dict()
    print(as_dict)
    conf = DimcatConfig(as_dict)
    print(conf)
    pl_copy = conf.create()
    assert pl == pl_copy
    assert pl.steps == pl_copy.steps
    print(pl.steps[0])
