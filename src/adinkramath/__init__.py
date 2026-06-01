"""adinkra-math: West African Adinkra symbolic encoding and SUSY adinkras for data scientists."""

from .symbol import Symbol, SymbolType, get_builtin_symbols
from .glyph import Glyph, GlyphOperation, compose, preserve_invariant
from .encoding import encode_concept, distance, nearest_concept, knn, kmeans, Concept
from .supersymmetry import create_adinkra, verify_chromotopology, boson_fermion_split, SUSYAdinkra
from .topology import connected_components, euler_characteristic, genus

__version__ = "0.1.0"
__all__ = [
    "Symbol", "SymbolType", "get_builtin_symbols",
    "Glyph", "GlyphOperation", "compose", "preserve_invariant",
    "encode_concept", "distance", "nearest_concept", "knn", "kmeans", "Concept",
    "create_adinkra", "verify_chromotopology", "boson_fermion_split", "SUSYAdinkra",
    "connected_components", "euler_characteristic", "genus",
]
