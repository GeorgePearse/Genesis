# EVOLVE-BLOCK-START
"""Hybrid optimization-based circle packing for n=26 circles"""

import numpy as np
from scipy.optimize import minimize

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
    initial_centers = np.zeros((n, 2))
    initial_radii = np.zeros(n)

    # Initial placement of circles
    initial_centers[0] = [0.5, 0.5]  # Center circle
    initial_radii[0] = 0.1  # Initial radius for center circle

    # Place circles in a structured pattern
    for i in range(1, n):
        angle = 2 * np.pi * (i - 1) / (n - 1)
        distance = 0.4 + 0.1 * (i % 2)  # Vary distance for outer circles
        initial_centers[i] = [0.5 + distance * np.cos(angle), 0.5 + distance * np.sin(angle)]
        initial_radii[i] = 0.05  # Initial radius for outer circles

    # Optimize positions and radii using a hybrid optimization approach
    optimized_result = minimize(objective_function, np.concatenate((initial_centers.flatten(), initial_radii)), 
                                 method='L-BFGS-B', bounds=bounds(n))

    # Extract optimized centers and radii
    optimized_centers = optimized_result.x[:2*n].reshape((n, 2))
    optimized_radii = optimized_result.x[2*n:]

    return optimized_centers, optimized_radii

def objective_function(x):
    """
    Objective function to maximize the sum of radii while ensuring no overlaps
    and that circles stay within the unit square.

    Args:
        x: Flattened array containing centers and radii

    Returns:
        Negative sum of radii (to minimize)
    """
    n = len(x) // 3
    centers = x[:2*n].reshape((n, 2))
    radii = x[2*n:]

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    # Penalize overlaps and out-of-bounds
    penalty = 0.0
    for i in range(n):
        # Check for out-of-bounds
        if np.any(centers[i] < radii[i]) or np.any(centers[i] > 1 - radii[i]):
            penalty += 1e6  # Large penalty for out-of-bounds

        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j]:  # Overlap detected
                penalty += 1e6  # Large penalty for overlap

    return -(sum_radii - penalty)  # Return negative for minimization

def bounds(n):
    """
    Create bounds for the optimization problem.

    Args:
        n: Number of circles

    Returns:
        List of bounds for centers and radii
    """
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x-coordinate
        bounds.append((0, 1))  # y-coordinate
    for i in range(n):
        bounds.append((0, 0.5))  # radius (max radius is 0.5)
    return bounds

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii