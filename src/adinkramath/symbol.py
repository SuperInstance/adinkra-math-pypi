"""Adinkra symbol primitives and built-in glyph definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SymbolType(Enum):
    """Fundamental geometric primitives found in Adinkra symbols."""
    CIRCLE = "circle"
    SPIRAL = "spiral"
    CROSS = "cross"
    KNOT = "knot"
    LINE = "line"
    ARC = "arc"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"


@dataclass
class Symbol:
    """A single Adinkra symbol composed of geometric primitives.

    Attributes:
        name: The traditional name of the symbol (e.g. "Gye Nyame").
        meaning: Short description of the symbol's proverbial meaning.
        primitives: Ordered list of geometric primitives that form the symbol.
        weight: Symbolic weight used in encoding (default 1.0).
    """
    name: str
    meaning: str
    primitives: List[SymbolType] = field(default_factory=list)
    weight: float = 1.0

    def __repr__(self) -> str:
        return f"Symbol({self.name!r}, meaning={self.meaning!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symbol):
            return NotImplemented
        return self.name == other.name and self.primitives == other.primitives

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.primitives)))

    def to_vector(self, dimensions: int = 8) -> List[float]:
        """Convert symbol to a fixed-length feature vector.

        Each dimension corresponds to a SymbolType; values are weighted counts.
        """
        vec = [0.0] * dimensions
        type_list = list(SymbolType)
        for p in self.primitives:
            idx = type_list.index(p) if p in type_list else None
            if idx is not None and idx < dimensions:
                vec[idx] += self.weight
        return vec


# ── Built-in Adinkra glyphs ──────────────────────────────────────────────

_BUILTINS: List[Symbol] = [
    Symbol("Gye Nyame", "Except God — supremacy of the divine",
           [SymbolType.SPIRAL, SymbolType.CIRCLE, SymbolType.ARC], weight=1.0),
    Symbol("Sankofa", "Go back and fetch it — learn from the past",
           [SymbolType.SPIRAL, SymbolType.LINE, SymbolType.ARC], weight=1.0),
    Symbol("Fawohodie", "Independence — freedom and emancipation",
           [SymbolType.CROSS, SymbolType.LINE, SymbolType.ARC], weight=1.0),
    Symbol("Nkyinkyim", "Initiative — dynamism and versatility",
           [SymbolType.SPIRAL, SymbolType.SPIRAL, SymbolType.LINE], weight=1.0),
    Symbol("Denkyem", "Adaptability — the crocodile lives in water but breathes air",
           [SymbolType.ARC, SymbolType.TRIANGLE, SymbolType.ARC], weight=1.0),
    Symbol("Eban", "Security — fence of protection and love",
           [SymbolType.CROSS, SymbolType.LINE, SymbolType.LINE, SymbolType.ARC], weight=1.0),
    Symbol("Adinkrahene", "Greatness — chief of all Adinkra symbols",
           [SymbolType.CIRCLE, SymbolType.CIRCLE, SymbolType.CIRCLE], weight=2.0),
    Symbol("Nsoromma", "Child of the heavens — guardianship and hope",
           [SymbolType.DIAMOND, SymbolType.LINE, SymbolType.ARC], weight=1.0),
    Symbol("Owuo Atwedee", "Mortality — the ladder of death",
           [SymbolType.LINE, SymbolType.LINE, SymbolType.LINE, SymbolType.CROSS], weight=1.0),
]


def get_builtin_symbols() -> List[Symbol]:
    """Return the nine canonical built-in Adinkra glyphs."""
    return list(_BUILTINS)
