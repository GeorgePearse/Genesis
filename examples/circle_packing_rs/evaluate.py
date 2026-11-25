"""Evaluator for the Rust circle-packing task. Standard library only.

Compiles the candidate with `rustc -O`, runs it, validates the layout it
prints and scores the sum of radii. The score is computed here from the
printed circles; nothing the program says about itself is trusted.
"""

import argparse
import json
import math
import os
import subprocess

N = 26
TOLERANCE = 1e-9
COMPILE_TIMEOUT = 120
RUN_TIMEOUT = 30


def write_results(results_dir, score, correct, error=None, public=None, feedback=""):
    metrics = {"combined_score": score, "public": public or {}, "private": {}, "text_feedback": feedback}
    with open(os.path.join(results_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(results_dir, "correct.json"), "w") as handle:
        json.dump({"correct": correct, "error": error}, handle, indent=2)


def parse_circles(output):
    circles = []
    for number, line in enumerate(output.strip().splitlines(), start=1):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"line {number} has {len(parts)} fields, expected `x y r`")
        x, y, r = (float(part) for part in parts)
        if not all(math.isfinite(value) for value in (x, y, r)):
            raise ValueError(f"line {number} has a non-finite value")
        circles.append((x, y, r))
    if len(circles) != N:
        raise ValueError(f"expected {N} circles, got {len(circles)}")
    return circles


def validate(circles):
    for index, (x, y, r) in enumerate(circles):
        if r < 0:
            return f"circle {index} has a negative radius {r}"
        if x - r < -TOLERANCE or x + r > 1 + TOLERANCE or y - r < -TOLERANCE or y + r > 1 + TOLERANCE:
            return f"circle {index} at ({x:.6f}, {y:.6f}) with radius {r:.6f} leaves the unit square"
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            (xi, yi, ri), (xj, yj, rj) = circles[i], circles[j]
            distance = math.hypot(xi - xj, yi - yj)
            if distance < ri + rj - TOLERANCE:
                return f"circles {i} and {j} overlap: centre distance {distance:.9f} < {ri + rj:.9f}"
    return None


def main(program_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    binary = os.path.join(results_dir, "candidate")
    try:
        compiled = subprocess.run(
            ["rustc", "-O", "--edition", "2021", "-o", binary, program_path],
            capture_output=True, text=True, timeout=COMPILE_TIMEOUT,
        )
    except FileNotFoundError:
        return write_results(results_dir, 0.0, False, "rustc is not installed")
    except subprocess.TimeoutExpired:
        return write_results(results_dir, 0.0, False, "compilation timed out")
    if compiled.returncode != 0:
        errors = compiled.stderr.strip()[-3000:]
        return write_results(results_dir, 0.0, False, "the program does not compile", feedback=errors)

    try:
        run = subprocess.run([binary], capture_output=True, text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return write_results(results_dir, 0.0, False, f"the program ran longer than {RUN_TIMEOUT} seconds")
    finally:
        if os.path.exists(binary):
            os.remove(binary)
    if run.returncode != 0:
        return write_results(results_dir, 0.0, False, f"the program exited with status {run.returncode}",
                             feedback=run.stderr.strip()[-2000:])

    try:
        circles = parse_circles(run.stdout)
    except ValueError as error:
        return write_results(results_dir, 0.0, False, f"unreadable output: {error}")
    problem = validate(circles)
    if problem:
        return write_results(results_dir, 0.0, False, problem, feedback=problem)

    total = sum(r for _, _, r in circles)
    radii = sorted(r for _, _, r in circles)
    public = {
        "sum_of_radii": total,
        "num_circles": len(circles),
        "min_radius": radii[0],
        "max_radius": radii[-1],
    }
    write_results(results_dir, total, True, public=public,
                  feedback=f"valid packing; sum of radii {total:.6f} (best known is about 2.635)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    arguments = parser.parse_args()
    main(arguments.program_path, arguments.results_dir)
