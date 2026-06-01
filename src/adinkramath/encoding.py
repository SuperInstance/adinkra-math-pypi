"""Concept encoding, distance metrics, KNN, and K-means clustering."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Concept:
    """A concept with an associated encoding vector.

    Attributes:
        label: Human-readable name.
        vector: Fixed-length float vector encoding.
    """
    label: str
    vector: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Concept({self.label!r}, dim={len(self.vector)})"


def _hash_float(text: str, seed: int) -> float:
    """Deterministic hash of text + seed into [0, 1)."""
    h = hashlib.sha256(f"{text}|{seed}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def encode_concept(text: str, dimensions: int = 8) -> List[float]:
    """Encode a text concept into a fixed-length float vector.

    Uses deterministic hashing so the same text always produces the same
    vector. Values are in the range [-1, 1].

    Args:
        text: The concept string to encode.
        dimensions: Desired vector length.

    Returns:
        A list of *dimensions* floats.
    """
    words = text.lower().split()
    n = len(words)
    vec: List[float] = [0.0] * dimensions
    if n == 0:
        return vec

    for i in range(dimensions):
        total = 0.0
        for j, word in enumerate(words):
            total += _hash_float(word, i * 1000 + j)
        vec[i] = (total / n) * 2.0 - 1.0  # shift to [-1, 1]

    # Normalize to unit length
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-12:
        vec = [v / norm for v in vec]
    return vec


def distance(v1: List[float], v2: List[float]) -> float:
    """Euclidean distance between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        The L2 distance (always >= 0).

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector lengths differ: {len(v1)} vs {len(v2)}")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def nearest_concept(
    vector: List[float],
    concepts: List[Concept],
) -> Optional[Concept]:
    """Find the nearest concept to a given vector by Euclidean distance.

    Args:
        vector: Query vector.
        concepts: Candidate concepts with pre-computed vectors.

    Returns:
        The nearest Concept, or None if the list is empty.
    """
    if not concepts:
        return None
    return min(concepts, key=lambda c: distance(vector, c.vector))


def knn(
    query: List[float],
    concepts: List[Concept],
    k: int = 5,
) -> List[Tuple[Concept, float]]:
    """K-nearest neighbours search.

    Args:
        query: Query vector.
        concepts: Search space.
        k: Number of neighbours to return.

    Returns:
        List of (concept, distance) tuples sorted by ascending distance.
    """
    if k <= 0:
        return []
    scored = [(c, distance(query, c.vector)) for c in concepts]
    scored.sort(key=lambda t: t[1])
    return scored[:k]


def _centroid(points: List[List[float]]) -> List[float]:
    """Compute the centroid of a list of vectors."""
    if not points:
        return []
    dim = len(points[0])
    return [sum(p[d] for p in points) / len(points) for d in range(dim)]


def kmeans(
    concepts: List[Concept],
    k: int = 3,
    max_iterations: int = 100,
    seed: Optional[int] = None,
) -> List[List[Concept]]:
    """Simple K-means clustering on concept vectors.

    Args:
        concepts: Data points to cluster.
        k: Number of clusters.
        max_iterations: Maximum number of Lloyd iterations.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A list of *k* clusters, each a list of Concepts.
    """
    n = len(concepts)
    if n == 0 or k <= 0:
        return [[] for _ in range(k)]
    k = min(k, n)

    rng = random.Random(seed)
    # Initialise centroids by picking k random concepts
    indices = rng.sample(range(n), k)
    centroids = [list(concepts[i].vector) for i in indices]

    clusters: List[List[Concept]] = []
    for _ in range(max_iterations):
        # Assign each concept to nearest centroid
        clusters = [[] for _ in range(k)]
        for c in concepts:
            best = min(range(k), key=lambda j: distance(c.vector, centroids[j]))
            clusters[best].append(c)

        # Recompute centroids
        new_centroids: List[List[float]] = []
        for j in range(k):
            if clusters[j]:
                new_centroids.append(_centroid([c.vector for c in clusters[j]]))
            else:
                new_centroids.append(centroids[j])

        if all(
            distance(a, b) < 1e-12
            for a, b in zip(centroids, new_centroids)
        ):
            break
        centroids = new_centroids

    return clusters
