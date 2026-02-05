"""

import numpy as np

# Fuzzy Set Operations

def fuzzy_union(A, B):
    return np.maximum(A, B)

def fuzzy_intersection(A, B):
    return np.minimum(A, B)

def fuzzy_complement(A):
    return 1 - A

def fuzzy_difference(A, B):
    return np.minimum(A, fuzzy_complement(B))



# Fuzzy Relation (Cartesian Product)

def fuzzy_relation(A, B):
    # Cartesian product: min(A(x), B(y)) for each pair (x, y)
    relation = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            relation[i][j] = min(A[i], B[j])
    return relation


# Max–Min Composition

def maxmin_composition(R1, R2):
    # R1: m x n, R2: n x p
    m, n = R1.shape
    n2, p = R2.shape
    if n != n2:
        raise ValueError("Incompatible relation sizes for composition")

    R = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            R[i][j] = np.max(np.minimum(R1[i, :], R2[:, j]))
    return R


# Example Usage

# Define two fuzzy sets A and B
A = np.array([0.2, 0.5, 0.7, 1.0])   # fuzzy set A
B = np.array([0.3, 0.6, 0.8, 0.4])   # fuzzy set B

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)

# Perform basic fuzzy set operations
print("\n--- Fuzzy Set Operations ---")
print("Union:", fuzzy_union(A, B))
print("Intersection:", fuzzy_intersection(A, B))
print("Complement of A:", fuzzy_complement(A))
print("Difference (A - B):", fuzzy_difference(A, B))

# Create fuzzy relation using Cartesian Product
print("\n--- Fuzzy Relation ---")
R1 = fuzzy_relation(A, B)
print("Relation R1 (A x B):\n", R1)

# Another fuzzy relation (B x A for example)
R2 = fuzzy_relation(B, A)
print("Relation R2 (B x A):\n", R2)

# Max-Min Composition of R1 and R2
print("\n--- Max-Min Composition ---")
R_comp = maxmin_composition(R1, R2)
print("R1 o R2:\n", R_comp)

"""
import argparse
import sys
import unittest
from datetime import datetime
from io import StringIO
import contextlib

def run_unittest(start_dir=".", pattern="test*.py", verbosity=2, failfast=False):
    loader = unittest.TestLoader()
    try:
        suite = loader.discover(start_dir=start_dir, pattern=pattern)
    except Exception as e:
        return None, f"Error during discovery: {e}\n"

    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    output = stream.getvalue()
    return result, output

def run_pytest(args_list):
    try:
        import pytest  # local import so module not required when not used
    except Exception as e:
        print("pytest is not available:", e, file=sys.stderr)
        return 1, ""
    stream = StringIO()
    # Capture pytest's stdout/stderr into our stream
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        rc = pytest.main(list(args_list))
    return rc, stream.getvalue()

def save_report(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def parse_args():
    p = argparse.ArgumentParser(description="Simple test runner (unittest + optional pytest).")
    p.add_argument("--runner", choices=("unittest", "pytest"), default="unittest",
                   help="Which test runner to use.")
    p.add_argument("--start-dir", default=".", help="Directory to discover tests (unittest).")
    p.add_argument("--pattern", default="test*.py", help="Filename pattern for unittest discovery.")
    p.add_argument("--verbosity", type=int, default=2, help="Verbosity for unittest runner.")
    p.add_argument("--failfast", action="store_true", help="Stop on first unittest failure.")
    p.add_argument("--output", default=None, help="Path to save textual test report.")
    p.add_argument("--save-raw", default=None, help="Save raw runner output to given path.")
    p.add_argument("pytest_args", nargs="*", help="Extra args forwarded to pytest (when using pytest).")
    return p.parse_args()

def main():
    args = parse_args()
    start_time = datetime.now()

    header = f"Test run at {start_time.isoformat(timespec='seconds')}\nRunner: {args.runner}\n\n"

    if args.runner == "pytest":
        pytest_args = args.pytest_args or ["-q"]
        rc, raw = run_pytest(pytest_args)
        duration = (datetime.now() - start_time).total_seconds()
        report = header + f"pytest exit code: {rc}\nDuration: {duration:.2f}s\n\n"
        report += raw
        print(report)
        if args.save_raw:
            save_report(args.save_raw, raw)
        if args.output:
            save_report(args.output, report)
        sys.exit(rc if rc == 0 else 2)

    # default: unittest
    result, output = run_unittest(start_dir=args.start_dir, pattern=args.pattern,
                                  verbosity=args.verbosity, failfast=args.failfast)

    duration = (datetime.now() - start_time).total_seconds()
    summary_lines = [header]
    if result is None:
        # discovery or runner error; output contains message
        summary_lines.append(output)
        rc = 2
        tests_run = 0
        failures = errors = skipped = 0
    else:
        summary_lines.append(output)
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, "skipped") else 0
        rc = 0 if (failures == 0 and errors == 0) else 2

    summary_lines.append("\nSummary:\n")
    summary_lines.append(f"  Ran: {tests_run}\n")
    summary_lines.append(f"  Failures: {failures}\n")
    summary_lines.append(f"  Errors: {errors}\n")
    summary_lines.append(f"  Skipped: {skipped}\n")
    summary_lines.append(f"  Duration: {duration:.2f}s\n")

    final_report = "".join(summary_lines)
    print(final_report)

    if args.save_raw and output:
        save_report(args.save_raw, output)
    if args.output:
        save_report(args.output, final_report)
        print(f"Saved report to: {args.output}")

    sys.exit(rc)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Test run interrupted by user.", file=sys.stderr)
        sys.exit(1)