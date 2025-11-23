# EVOLVE-BLOCK-START
"""Adaptive circle packing for n=26 circles"""

import numpy as np

def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Initial placement of circles
    centers[0] = [0.5, 0.5]  # Center circle
    radii[0] = 0.1  # Initial radius for the center circle

    # Place circles in a spiral pattern
    for i in range(1, n):
        angle = 2 * np.pi * (i - 1) / (n - 1)
        radius = 0.1 + 0.02 * (i % 5)  # Varying radius for diversity
        centers[i] = [0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)]
        radii[i] = radius

    # Adjust positions and radii to avoid overlap and stay within bounds
    for _ in range(100):  # Iterative refinement
        for i in range(n):
            # Clip positions to stay within the unit square
            centers[i] = np.clip(centers[i], radii[i], 1 - radii[i])

            # Adjust radii based on distance to borders
            radii[i] = min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])

            # Check for overlaps and adjust radii
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j]:  # Overlap detected
                    # Scale down radii proportionally
                    overlap = radii[i] + radii[j] - dist
                    scale = (dist) / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale

    # Calculate the final sum of radii
    sum_of_radii = np.sum(radii)
    return centers, radii, sum_of_radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii