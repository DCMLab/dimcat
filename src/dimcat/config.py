import marshmallow as mm
from dimcat.utils import get_class, get_schema


class DimcatSchema(mm.Schema):
    """
    The base class of the config system of Dimcat, with which to define custom configs. This class holds the logic for
    serializing/deserializing config objects.
    """

    dtype = mm.fields.String()

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__qualname__

    @mm.post_load()
    def init_object(self, data, **kwargs):
        obj_name = data.pop("dtype")
        Constructor = get_class(obj_name)
        return Constructor(**data)

    @mm.post_dump()
    def validate_dump(self, data, **kwargs) -> None:
        if "dtype" not in data:
            raise mm.ValidationError(
                "The object to be serialized doesn't have a 'dtype' field. May it's not a "
                "DimcatObject?"
            )
        dtype_schema = get_schema(data["dtype"])
        report = dtype_schema.validate(data)
        if report:
            raise mm.ValidationError(
                f"Dump of {data['dtype']} created with a {self.name} could not be validated by "
                f"{dtype_schema.name} :\n{report}"
            )
        return data

    def __repr__(self):
        return f"{self.name}(many={self.many})"
