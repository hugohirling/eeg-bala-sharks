"""
Preprocessing Authors Module

Pipeline implementation following Moerel et al. (2025):
"Neural decoding of competitive decision-making in Rock-Paper-Scissors"
"""

from .master_pipeline_authors import AuthorsPreprocessingPipeline, main

__all__ = ['AuthorsPreprocessingPipeline', 'main']
