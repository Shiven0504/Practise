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