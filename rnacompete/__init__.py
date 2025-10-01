""""
RNAcompete: Processing and summarization pipeline.

This package provide pipelines for processing and summarizing RNAcompete data.
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Package version
__version__ = '1.0.0'

# Public API
from .modes.process import run_process
from .modes.summarize import run_summarize

__all__ = [
    'run_process',
    'run_summarize'
]