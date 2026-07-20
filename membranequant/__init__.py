"""MembraneQuant: membrane localization quantification for EGFP + DiI images."""

import warnings

# Ignore skimage FutureWarnings (min_size, binary_opening deprecation warnings in dualcellquant/skimage)
warnings.filterwarnings("ignore", category=FutureWarning, module=".*skimage.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*dualcellquant.*")
warnings.filterwarnings("ignore", message=".*remove_small_objects.*")
warnings.filterwarnings("ignore", message=".*binary_opening.*")

__version__ = "0.1.0"

