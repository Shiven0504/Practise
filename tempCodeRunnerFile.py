import argparse
import sys
import unittest
from datetime import datetime
from io import StringIO

def run_unittest(start_dir=".", pattern="test*.py", verbosity=2, failfast=False):
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=start_dir, pattern=pattern)
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    output = stream.getvalue()
    return result, output

def run_pytest(args_list):
    try:
        import pytest
    except Exception as e:
        print("pytest is not available:", e, file=sys.stderr)
        return 1, ""
    # call pytest main; it will run and return an exit code
    rc = pytest.main(args_list)
    return rc, ""

def save_report(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def parse_args():
    p = argparse.ArgumentParser(description="Simple test runner (unittest + optional pytest fallback).")
    p.add_argument("--runner", choices=("unittest", "pytest"), default="unittest",
                   help="Which test runner to use.")
    p.add_argument("--start-dir", default=".", help="Directory to discover tests (unittest).")
    p.add_argument("--pattern", default="test*.py", help="Filename pattern for unittest discovery.")
    p.add_argument("--verbosity", type=int, default=2, help="Verbosity for unittest runner.")
    p.add_argument("--failfast", action="store_true", help="Stop on first unittest failure.")
    p.add_argument("--output", default=None, help="Path to save textual test report.")
    p.add_argument("pytest_args", nargs="*", help="Extra args forwarded to pytest (when using pytest).")
    return p.parse_args()

def main():
    args = parse_args()

    timestamp = datetime.now().isoformat(timespec="seconds")
    header = f"Test run at {timestamp}\nRunner: {args.runner}\n\n"

    if args.runner == "pytest":
        rc, _ = run_pytest(args.pytest_args or ["-q"])
        if args.output:
            save_report(args.output, header + f"pytest exit code: {rc}\n")
        sys.exit(rc)

    # default: unittest
    result, output = run_unittest(start_dir=args.start_dir, pattern=args.pattern,
                                  verbosity=args.verbosity, failfast=args.failfast)

    summary_lines = []
    summary_lines.append(header)
    summary_lines.append(output)
    summary_lines.append("\nSummary:\n")
    summary_lines.append(f"  Ran: {result.testsRun}\n")
    summary_lines.append(f"  Failures: {len(result.failures)}\n")
    summary_lines.append(f"  Errors: {len(result.errors)}\n")
    summary_lines.append(f"  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}\n")
    rc = 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 2

    final_report = "".join(summary_lines)
    print(final_report)

    if args.output:
        save_report(args.output, final_report)
        print(f"Saved report to: {args.output}")

    sys.exit(rc)

if __name__ == "__main__":
    main()