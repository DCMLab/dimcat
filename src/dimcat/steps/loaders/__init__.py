import logging

from .base import Loader, PackageLoader, ScoreLoader
from .m21 import Music21Loader
from .musescore import MuseScoreLoader

module_logger = logging.getLogger(__name__)
