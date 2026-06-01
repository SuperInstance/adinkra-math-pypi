"""Comprehensive pytest suite for adinkra-math (45+ tests)."""

from __future__ import annotations

import math
import pytest

from adinkramath import (
    Symbol, SymbolType, get_builtin_symbols,
    Glyph, GlyphOperation, compose, preserve_invariant,
    encode_concept, distance, nearest_concept, knn, kmeans,
    Concept,
    create_adinkra, verify_chromotopology, boson_fermion_split, SUSYAdinkra,
    connected_components, euler_characteristic, genus,
)


# ── Symbol tests ──────────────────────────────────────────────────────────


class TestSymbolType:
    def test_all_types_exist(self):
        expected = {"CIRCLE", "SPIRAL", "CROSS", "KNOT", "LINE", "ARC", "TRIANGLE", "DIAMOND"}
        assert {t.name for t in SymbolType} == expected

    def test_type_values(self):
        assert SymbolType.CIRCLE.value == "circle"
        assert SymbolType.SPIRAL.value == "spiral"


class TestSymbol:
    def test_creation(self):
        s = Symbol("test", "a test", [SymbolType.CIRCLE])
        assert s.name == "test"
        assert s.meaning == "a test"
        assert s.primitives == [SymbolType.CIRCLE]

    def test_default_weight(self):
        s = Symbol("x", "y")
        assert s.weight == 1.0

    def test_to_vector_basic(self):
        s = Symbol("v", "v", [SymbolType.CIRCLE, SymbolType.CIRCLE])
        vec = s.to_vector()
        assert len(vec) == 8
        # CIRCLE is index 0
        assert vec[0] == pytest.approx(2.0)

    def test_to_vector_dimensions(self):
        s = Symbol("d", "d", [SymbolType.DIAMOND])
        vec = s.to_vector(dimensions=4)
        assert len(vec) == 4

    def test_equality(self):
        a = Symbol("a", "x", [SymbolType.LINE])
        b = Symbol("a", "y", [SymbolType.LINE])
        assert a == b

    def test_hash(self):
        a = Symbol("a", "x", [SymbolType.LINE])
        b = Symbol("a", "x", [SymbolType.LINE])
        assert hash(a) == hash(b)

    def test_repr(self):
        s = Symbol("Gye Nyame", "supreme")
        assert "Gye Nyame" in repr(s)


class TestBuiltins:
    def test_count(self):
        assert len(get_builtin_symbols()) == 9

    def test_names_unique(self):
        names = [s.name for s in get_builtin_symbols()]
        assert len(names) == len(set(names))

    def test_gye_nyame_first(self):
        syms = get_builtin_symbols()
        assert syms[0].name == "Gye Nyame"

    def test_adinkrahene_weight(self):
        syms = get_builtin_symbols()
        chief = [s for s in syms if s.name == "Adinkrahene"][0]
        assert chief.weight == 2.0

    def test_all_have_primitives(self):
        for s in get_builtin_symbols():
            assert len(s.primitives) > 0, f"{s.name} has no primitives"


# ── Glyph tests ───────────────────────────────────────────────────────────


class TestGlyph:
    def test_creation(self):
        g = Glyph(name="test")
        assert g.name == "test"
        assert g.symbols == []

    def test_primitives(self):
        s1 = Symbol("a", "a", [SymbolType.CIRCLE, SymbolType.LINE])
        s2 = Symbol("b", "b", [SymbolType.ARC])
        g = Glyph(symbols=[s1, s2])
        assert g.primitives() == [SymbolType.CIRCLE, SymbolType.LINE, SymbolType.ARC]

    def test_total_weight(self):
        s1 = Symbol("a", "a", weight=2.0)
        s2 = Symbol("b", "b", weight=3.0)
        g = Glyph(symbols=[s1, s2])
        assert g.total_weight() == 5.0


class TestCompose:
    @pytest.fixture()
    def glyphs(self):
        s1 = Symbol("left", "l", [SymbolType.CIRCLE])
        s2 = Symbol("right", "r", [SymbolType.LINE])
        return Glyph(symbols=[s1], name="L"), Glyph(symbols=[s2], name="R")

    def test_stack(self, glyphs):
        g1, g2 = glyphs
        result = compose(g1, g2, GlyphOperation.STACK)
        assert len(result.symbols) == 2

    def test_nest(self, glyphs):
        g1, g2 = glyphs
        result = compose(g1, g2, GlyphOperation.NEST)
        assert len(result.symbols) == 2
        assert "nested" in result.symbols[1].name

    def test_interleave(self, glyphs):
        g1, g2 = glyphs
        result = compose(g1, g2, GlyphOperation.INTERLEAVE)
        assert len(result.symbols) == 2

    def test_mirror(self, glyphs):
        g1, g2 = glyphs
        result = compose(g1, g2, GlyphOperation.MIRROR)
        assert len(result.symbols) == 2

    def test_overlay(self, glyphs):
        g1, g2 = glyphs
        result = compose(g1, g2, GlyphOperation.OVERLAY)
        assert len(result.symbols) == 1
        assert "overlay" in result.symbols[0].name


class TestPreserveInvariant:
    def test_valid(self):
        s = Symbol("v", "v", [SymbolType.CIRCLE])
        g = Glyph(symbols=[s])
        assert preserve_invariant(g) is True

    def test_no_closed_curve(self):
        s = Symbol("x", "x", [SymbolType.LINE, SymbolType.CROSS])
        g = Glyph(symbols=[s])
        assert preserve_invariant(g) is False

    def test_zero_weight(self):
        s = Symbol("z", "z", [SymbolType.ARC], weight=0.0)
        g = Glyph(symbols=[s])
        assert preserve_invariant(g) is False


# ── Encoding tests ────────────────────────────────────────────────────────


class TestEncodeConcept:
    def test_deterministic(self):
        v1 = encode_concept("hello world")
        v2 = encode_concept("hello world")
        assert v1 == pytest.approx(v2)

    def test_dimensions(self):
        v = encode_concept("test", dimensions=16)
        assert len(v) == 16

    def test_unit_length(self):
        v = encode_concept("unit vector test")
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_empty_string(self):
        v = encode_concept("")
        assert v == [0.0] * 8


class TestDistance:
    def test_same_vector(self):
        assert distance([1, 2], [1, 2]) == pytest.approx(0.0)

    def test_known_distance(self):
        assert distance([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            distance([1], [1, 2])


class TestNearestConcept:
    def test_basic(self):
        concepts = [
            Concept("a", [1.0, 0.0]),
            Concept("b", [0.0, 1.0]),
            Concept("c", [1.0, 1.0]),
        ]
        result = nearest_concept([0.9, 0.1], concepts)
        assert result is not None
        assert result.label == "a"

    def test_empty(self):
        assert nearest_concept([1, 2], []) is None


class TestKNN:
    def test_k2(self):
        concepts = [Concept(str(i), [float(i)]) for i in range(10)]
        results = knn([4.5], concepts, k=2)
        assert len(results) == 2
        assert results[0][0].label == "4"
        assert results[1][0].label == "5"

    def test_k0(self):
        assert knn([0], [], k=0) == []


class TestKMeans:
    def test_two_clusters(self):
        concepts = [
            Concept("a", [0.0, 0.0]),
            Concept("b", [0.1, 0.1]),
            Concept("c", [10.0, 10.0]),
            Concept("d", [9.9, 9.9]),
        ]
        clusters = kmeans(concepts, k=2, seed=42)
        assert len(clusters) == 2
        total = sum(len(c) for c in clusters)
        assert total == 4

    def test_empty(self):
        clusters = kmeans([], k=3)
        assert all(len(c) == 0 for c in clusters)


# ── Supersymmetry tests ───────────────────────────────────────────────────


class TestSUSYAdinkra:
    def test_rank1(self):
        a = create_adinkra(1)
        assert a.rank == 1
        assert len(a.bosons) == 1
        assert len(a.fermions) == 1
        assert len(a.generators) == 1

    def test_rank2(self):
        a = create_adinkra(2)
        assert len(a.bosons) == 2
        assert len(a.fermions) == 2
        assert len(a.generators) == 2

    def test_rank4(self):
        a = create_adinkra(4)
        assert a.node_count() == 16  # 8 + 8
        assert len(a.generators) == 4

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            create_adinkra(0)

    def test_adjacency(self):
        a = create_adinkra(2)
        adj = a.adjacency()
        # Every node should appear in adjacency
        for node in a.nodes:
            assert node in adj

    def test_chromotopology_rank1(self):
        a = create_adinkra(1)
        assert verify_chromotopology(a) is True

    def test_boson_fermion_split(self):
        a = create_adinkra(3)
        split = boson_fermion_split(a)
        assert len(split["bosons"]) == 4
        assert len(split["fermions"]) == 4

    def test_generators_named(self):
        a = create_adinkra(3)
        assert a.generators == ["Q1", "Q2", "Q3"]


# ── Topology tests ────────────────────────────────────────────────────────


class TestConnectedComponents:
    def test_single_node(self):
        assert connected_components({"A": []}) == 1

    def test_disconnected(self):
        graph = {"A": ["B"], "B": ["A"], "C": []}
        assert connected_components(graph) == 2

    def test_fully_connected(self):
        graph = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
        assert connected_components(graph) == 1

    def test_empty(self):
        assert connected_components({}) == 0


class TestEulerCharacteristic:
    def test_sphere(self):
        # Cube: V=8, E=12, F=6
        assert euler_characteristic(8, 12, 6) == 2

    def test_torus(self):
        # Torus: V=1, E=2, F=1 => χ=0
        assert euler_characteristic(1, 2, 1) == 0

    def test_tetrahedron(self):
        # V=4, E=6, F=4
        assert euler_characteristic(4, 6, 4) == 2


class TestGenus:
    def test_sphere(self):
        assert genus(2) == 0

    def test_torus(self):
        assert genus(0) == 1

    def test_double_torus(self):
        assert genus(-2) == 2

    def test_invalid_odd(self):
        with pytest.raises(ValueError):
            genus(3)

    def test_invalid_gt2(self):
        with pytest.raises(ValueError):
            genus(4)
