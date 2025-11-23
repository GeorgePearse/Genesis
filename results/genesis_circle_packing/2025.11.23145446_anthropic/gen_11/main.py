# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

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
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Use a dynamic and adaptive placement strategy for circles
    for i in range(n):
        centers[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]  # Initial random placement

    # Optimize positions iteratively
    for _ in range(100):  # Number of optimization iterations
        radii = compute_max_radii(centers)
        centers = optimize_positions(centers, radii)

    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Final radius computation
    radii = compute_max_radii(centers)
    return centers, radii


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
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
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