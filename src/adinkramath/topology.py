"""Graph topology utilities: connected components, Euler characteristic, genus."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple


def connected_components(graph: Dict[str, List[str]]) -> int:
    """Count connected components in an undirected adjacency-list graph.

    Args:
        graph: Mapping of node → list of neighbours.

    Returns:
        Integer count of connected components.
    """
    visited: Set[str] = set()
    components = 0

    for node in graph:
        if node in visited:
            continue
        components += 1
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbour in graph.get(current, []):
                if neighbour not in visited:
                    stack.append(neighbour)

    return components


def euler_characteristic(v: int, e: int, f: int) -> int:
    """Compute the Euler characteristic χ = V − E + F.

    For a polyhedron homeomorphic to a sphere, χ = 2.
    For a torus, χ = 0.

    Args:
        v: Number of vertices.
        e: Number of edges.
        f: Number of faces.

    Returns:
        The Euler characteristic.
    """
    return v - e + f


def genus(euler: int) -> int:
    """Compute the genus (number of handles) from the Euler characteristic.

    For an orientable closed surface: g = (2 − χ) / 2.

    Args:
        euler: Euler characteristic χ.

    Returns:
        The genus g (non-negative integer for valid surfaces).

    Raises:
        ValueError: If χ is odd or greater than 2 (impossible for closed
            orientable surfaces).
    """
    if euler > 2:
        raise ValueError(f"Euler characteristic {euler} > 2 is impossible for a closed surface")
    if (2 - euler) % 2 != 0:
        raise ValueError(f"Euler characteristic {euler} gives non-integer genus")
    return (2 - euler) // 2
