# adinkra-math

> West African Adinkra symbols as a mathematical framework — symbolic encoding, topology, supersymmetry, and machine learning.

## What This Does

`adinkra-math` implements the mathematics inspired by Adinkra symbols — the visual symbols of the Ashanti people of West Africa. It provides symbolic encoding of concepts as geometric primitives, glyph composition with invariant preservation, topological analysis (Euler characteristic, genus), supersymmetry Adinkra graphs (Boson-Fermion classification with chromotopology verification), and basic ML operations (kNN, K-means) on symbol vectors. Use it for symbolic AI, topological data analysis, physics simulations, or cultural math education.

## The Cultural Root

Adinkra symbols originated with the Ashanti (Asante) people of Ghana and Côte d'Ivoire. Each symbol encodes a proverb or concept — Sankofa ("go back and fetch it") represents learning from the past. The mathematical insight: **symbols compress complex meaning into structured geometric primitives**, which is exactly what feature vectors do in machine learning. Adinkras also appear in theoretical physics: supersymmetry Adinkras are bipartite graphs encoding the relationship between bosons and fermions.

## Install

```bash
pip install adinkra-math
```

## Quick Start

```python
from adinkramath.symbol import Symbol, SymbolType, get_builtin_symbols
from adinkramath.glyph import Glyph, GlyphOperation, compose
from adinkramath.topology import euler_characteristic, genus
from adinkramath.encoding import encode_concept, knn, kmeans
from adinkramath.supersymmetry import create_adinkra, verify_chromotopology

# Work with built-in Adinkra symbols
symbols = get_builtin_symbols()
for s in symbols:
    print(f"{s.name}: {s.to_vector()}")

# Compose glyphs (symbols combined)
g1 = Glyph(symbols=[symbols[0]])
g2 = Glyph(symbols=[symbols[1]])
composed = compose(g1, g2, GlyphOperation.SUPERIMPOSE)
print(f"Composed weight: {composed.total_weight()}")

# Check topological invariant
from adinkramath.glyph import preserve_invariant
print(f"Invariant preserved: {preserve_invariant(composed)}")

# Encode concepts as vectors
c1 = encode_concept("courage", seed=42)
c2 = encode_concept("wisdom", seed=42)
print(f"Vector length: {len(c1.vector)}")

# K-nearest neighbors
from adinkramath.encoding import Concept, nearest_concept
concepts = [encode_concept(w, seed=0) for w in ["love", "war", "peace", "fear"]]
result = nearest_concept(encode_concept("bravery", seed=0).vector, concepts)

# K-means clustering
clusters = kmeans(concepts, k=2)

# Supersymmetry Adinkras
adinkra = create_adinkra(rank=2)
print(f"Bosons: {len(adinkra.bosons)}, Fermions: {len(adinkra.fermions)}")
valid = verify_chromotopology(adinkra)
print(f"Chromotopology valid: {valid}")

# Topology
euler = euler_characteristic(vertices=8, edges=12, faces=6)
print(f"Euler characteristic: {euler}, Genus: {genus(euler)}")
```

## API Reference

### `symbol` module

#### `SymbolType` (enum)
`CIRCLE`, `SPIRAL`, `CROSS`, `LINE`, `ARC`, `DIAMOND`, `TRIANGLE`, `STAR`, `HEART` — fundamental geometric primitives.

#### `Symbol`
```python
@dataclass
class Symbol:
    name: str
    primitives: list[tuple[SymbolType, dict]]  # (type, parameters)
    meaning: str
    weight: float = 1.0

    def to_vector(self) -> list[float]
```

#### `get_builtin_symbols() → list[Symbol]`
Returns nine canonical Adinkra glyphs with their geometric primitives.

### `glyph` module

#### `GlyphOperation` (enum)
`SUPERIMPOSE`, `CONCATENATE`, `NEST` — ways to combine glyphs.

#### `Glyph`
```python
class Glyph:
    symbols: list[Symbol]

    def primitives(self) -> list[tuple]
    def total_weight(self) -> float
```

#### `compose(g1, g2, operation) → Glyph`
Combine two glyphs. Returns a new Glyph with merged primitives.

#### `preserve_invariant(glyph) → bool`
Check if the glyph preserves the Adinkra topological invariant (must contain both closed and open primitives).

### `topology` module

#### `connected_components(graph) → int`
Count connected components in an adjacency-list graph.

#### `euler_characteristic(vertices, edges, faces) → int`
Compute χ = V − E + F. For a sphere, χ = 2.

#### `genus(euler_char) → int`
Compute genus g = (2 − χ) / 2. Sphere → 0, torus → 1.

### `encoding` module

#### `Concept`
```python
@dataclass
class Concept:
    label: str
    vector: list[float]
```

#### `encode_concept(text, dimensions=16, seed=None) → Concept`
Deterministically hash text to a fixed-length float vector.

#### `distance(v1, v2) → float`
Euclidean distance between vectors.

#### `nearest_concept(vector, concepts) → Concept`
Find the closest concept by Euclidean distance.

#### `knn(query, concepts, k=5) → list[Concept]`
K-nearest neighbors search.

#### `kmeans(concepts, k=3, max_iter=100) → list[list[Concept]]`
Simple K-means clustering on concept vectors.

### `supersymmetry` module

#### `SUSYAdinkra`
```python
@dataclass
class SUSYAdinkra:
    rank: int
    bosons: list[int]
    fermions: list[int]
    edges: list[tuple[int, int, int]]  # (boson, fermion, generator_index)
```

#### `create_adinkra(rank) → SUSYAdinkra`
Create a rank-N supersymmetry Adinkra with 2^(N-1) bosons and 2^(N-1) fermions.

#### `verify_chromotopology(adinkra) → bool`
Verify: (1) every node has exactly N edges, (2) each generator index appears once per node, (3) two-colorability holds.

#### `boson_fermion_split(adinkra) → tuple[list, list]`
Split the Adinkra into bosonic and fermionic sectors.

## How It Works

**Symbol Vectors:** Each `SymbolType` maps to a dimension in a fixed-length feature vector. A symbol's vector is the weighted sum of its primitives' contributions.

**Concept Encoding:** Uses deterministic hashing (SHA-family) to map arbitrary text to a float vector of configurable dimension. Same input always produces the same vector.

**K-means:** Standard Lloyd's algorithm — random initialization, assign to nearest centroid, recompute centroids, repeat until convergence or max iterations.

**SUSY Adinkras:** For rank N, creates 2^(N-1) boson and 2^(N-1) fermion nodes connected by N generators. The chromotopology conditions ensure the graph properly represents supersymmetry algebra.

## The Math

**Euler Characteristic:** χ = V − E + F. For orientable closed surfaces: genus g = (2 − χ)/2.

**Supersymmetry Adinkras:** For N supersymmetry generators, the Adinkra has 2^(N-1) bosons and 2^(N-1) fermions. Each node has degree N. The chromotopology ensures the "dashing" (chromatic structure) is consistent with the Clifford algebra underlying supersymmetry.

**K-means Objective:** Minimize J = Σₖ Σ_{x∈Cₖ} ||x − μₖ||² where μₖ is the centroid of cluster Cₖ.

## License

MIT
