import logging

logger = logging.getLogger(__name__)


class PotentiallyUnrelatedDescriptorUserWarning(UserWarning):
    """This warning is shown when, as a result of modifying a basepath, the descriptor_path
    points to a pre-existing file on disk which could potentially have nothing to do with the
    current resource.
    """

    pass
