"""Glyph composition, operations, and invariant preservation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .symbol import Symbol, SymbolType


class GlyphOperation(Enum):
    """Ways to compose two glyphs into a composite symbol."""
    STACK = "stack"
    NEST = "nest"
    INTERLEAVE = "interleave"
    MIRROR = "mirror"
    OVERLAY = "overlay"


@dataclass
class Glyph:
    """A composite Adinkra glyph built from one or more symbols.

    Attributes:
        symbols: The component symbols.
        name: Optional display name for the composite.
        meaning: Optional semantic meaning.
    """
    symbols: List[Symbol] = field(default_factory=list)
    name: str = ""
    meaning: str = ""

    def __repr__(self) -> str:
        return f"Glyph({self.name or 'unnamed'}, symbols={len(self.symbols)})"

    def primitives(self) -> List[SymbolType]:
        """Flatten all component primitives."""
        result: List[SymbolType] = []
        for s in self.symbols:
            result.extend(s.primitives)
        return result

    def total_weight(self) -> float:
        """Sum of component symbol weights."""
        return sum(s.weight for s in self.symbols)


def compose(g1: Glyph, g2: Glyph, operation: GlyphOperation) -> Glyph:
    """Compose two glyphs using the specified operation.

    Args:
        g1: First (base) glyph.
        g2: Second (modifier) glyph.
        operation: Composition strategy.

    Returns:
        A new composite Glyph.
    """
    combined: List[Symbol] = []
    name = f"{g1.name or '?'}+{g2.name or '?'}"

    if operation == GlyphOperation.STACK:
        combined = list(g1.symbols) + list(g2.symbols)

    elif operation == GlyphOperation.NEST:
        combined = list(g1.symbols)
        for s in g2.symbols:
            combined.append(Symbol(
                name=f"nested({s.name})",
                meaning=s.meaning,
                primitives=s.primitives,
                weight=s.weight * 0.8,
            ))

    elif operation == GlyphOperation.INTERLEAVE:
        it1, it2 = iter(g1.symbols), iter(g2.symbols)
        while True:
            exhausted = True
            try:
                combined.append(next(it1))
                exhausted = False
            except StopIteration:
                pass
            try:
                combined.append(next(it2))
                exhausted = False
            except StopIteration:
                pass
            if exhausted:
                break

    elif operation == GlyphOperation.MIRROR:
        combined = list(g1.symbols) + list(reversed(g2.symbols))

    elif operation == GlyphOperation.OVERLAY:
        merged_prims: List[SymbolType] = []
        for s in g1.symbols:
            merged_prims.extend(s.primitives)
        for s in g2.symbols:
            merged_prims.extend(s.primitives)
        combined = [Symbol(
            name=f"overlay({g1.name},{g2.name})",
            meaning=f"overlay of {g1.meaning} and {g2.meaning}",
            primitives=merged_prims,
            weight=g1.total_weight() + g2.total_weight(),
        )]

    return Glyph(
        symbols=combined,
        name=name,
        meaning=f"{g1.meaning or ''} ∘ {g2.meaning or ''}".strip(" ∘"),
    )


def preserve_invariant(glyph: Glyph) -> bool:
    """Check whether a glyph preserves the Adinkra topological invariant.

    The invariant requires that the glyph contain at least one closed curve
    (CIRCLE, SPIRAL, or ARC) and that the total weight be positive.

    Args:
        glyph: The glyph to check.

    Returns:
        True if the invariant is preserved.
    """
    closed_types = {SymbolType.CIRCLE, SymbolType.SPIRAL, SymbolType.ARC}
    has_closed = any(p in closed_types for p in glyph.primitives())
    positive_weight = glyph.total_weight() > 0
    return has_closed and positive_weight
