# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles"""

import numpy as np


def initialize_centers(n):
    """Initialize the centers of the circles."""
    centers = np.zeros((n, 2))
    centers[0] = [0.5, 0.5]  # Center circle

    # Place circles in a structured pattern
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    return np.clip(centers, 0.01, 0.99)  # Ensure within bounds


def compute_max_radii(centers):
    """Compute the maximum possible radii for each circle position."""
    n = centers.shape[0]
    radii = np.ones(n)

    # Limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Limit by distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


def adjust_circle_positions(centers, radii):
    """Adjust circle positions based on computed radii."""
    for i in range(len(centers)):
        x, y = centers[i]
        # Adjust position if radius is too large
        if radii[i] > min(x, y, 1 - x, 1 - y):
            centers[i] = [np.clip(x, radii[i], 1 - radii[i]), np.clip(y, radii[i], 1 - radii[i])]


def construct_packing():
    """Construct a specific arrangement of 26 circles in a unit square."""
    n = 26
    centers = initialize_centers(n)
    radii = compute_max_radii(centers)
    adjust_circle_positions(centers, radii)
    return centers, radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
