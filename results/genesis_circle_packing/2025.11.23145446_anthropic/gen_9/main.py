# EVOLVE-BLOCK-START
"""Adaptive circle packing for n=26 circles"""

import numpy as np

def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    centers = np.zeros((n, 2))

    # Initial placements for inner and outer circles
    radius_in = 0.20
    radius_out = 0.35

    # Place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around the center
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + radius_in * np.cos(angle), 0.5 + radius_in * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        radius = radius_out * (1 - (i % 2) * 0.05)  # Slightly smaller for alternate circles
        centers[i + 9] = [0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)]

    # Optimize positions iteratively
    for _ in range(100):  # Number of optimization iterations
        radii = compute_max_radii(centers)
        centers = optimize_positions(centers, radii)

    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Final radius computation
    radii = compute_max_radii(centers)
    return centers, radii


def optimize_positions(centers, radii):
    """
    Adjust the positions of circles based on their radii to prevent overlaps
    and maintain proximity to the center.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radii of circles

    Returns:
        Updated centers after optimization
    """
    n = centers.shape[0]
    
    # Move circles towards the centroid of their neighboring circles
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(centers[i] - centers[j])
                # If too close, adjust position away based on radius
                if dist < (radii[i] + radii[j]):
                    direction = (centers[i] - centers[j]) / dist
                    adjustment = (radii[i] + radii[j] - dist) / 2
                    centers[i] += adjustment * direction  # Push circle i away from j

    return centers


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii