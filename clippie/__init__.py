"""
Clippie: A simple CPU-based, implementation of OpenAI's CLIP model's
forward-pass.

This top-level module re-exports the basic API needed to make use of Clippie
for encoding text and images.
"""

from clippie.serialiser import load

from clippie.model import encode_text, encode_image, Weights

# Bonus: Also re-export softmax which may be useful in converting
# log-similarity matrices into probabilities of the kind CLIP was trained to
# produce.
from clippie.nn import softmax
