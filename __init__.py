"""
Climate-Disease-Africa
src/__init__.py

Author:  Emmanuel Yaw Afram (Prestige)
Email:   eyafram7@gmail.com
GitHub:  github.com/eyafram7-data
"""
__version__ = "1.0.0"
__author__  = "Emmanuel Yaw Afram (Prestige)"
__email__   = "eyafram7@gmail.com"

from .data_loader   import DiseaseDataLoader
from .preprocessing import DiseasePreprocessor
from .visualization import DiseaseVisualizer
from .model         import OutbreakModelTrainer
from .prediction    import OutbreakPredictor

__all__ = [
    "DiseaseDataLoader", "DiseasePreprocessor",
    "DiseaseVisualizer", "OutbreakModelTrainer", "OutbreakPredictor"
]
