import logging

logger = logging.getLogger(__name__)


class OrderOfPipelineStepsWarning(UserWarning):
    """This warning is shown when the order of pipeline steps may lead to unexpected behaviour."""

    pass


class PotentiallyUnrelatedDescriptorUserWarning(UserWarning):
    """This warning is shown when, as a result of modifying a basepath, the descriptor_path
    points to a pre-existing file on disk which could potentially have nothing to do with the
    current resource.
    """

    pass


class PotentiallyMisalignedPackageUserWarning(UserWarning):
    """This warning is shown when resources are added to a package whose status is not ALLOW_MISALIGNMENT
    but which has no defined basepath."""

    pass


class ResourceWithRangeIndexUserWarning(UserWarning):
    """This warning is shown when a resource has a range index, which is typically the case
    for dataframes holding information for single piece only.
    """

    pass
