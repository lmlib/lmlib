"""
run_local_skipped_examples.py
==============================
Finds all examples excluded from the automated gallery build (``gen-gallery:skip-run``
blocks in ``README.md`` files) and runs them locally, saving the produced matplotlib
figures next to each ``.py`` source file as ``<stem>.png``, ``<stem>_2.png``, etc.

These committed PNGs are then picked up by the gallery build without re-running the
scripts (which may need a GPU or other hardware unavailable on CI).

Usage
-----
    python scripts/run_local_skipped_examples.py [options]

Options
-------
    --root DIR      Root directory to search for README.md files (default: coding/)
    --timeout N     Kill a script after N seconds (default: 300)
    --dry-run       Print which scripts would be executed without running them
    --verbose       Print full stdout/stderr for every script, not just failures

Examples
--------
    python scripts/run_local_skipped_examples.py
    python scripts/run_local_skipped_examples.py --root coding/91-backend
    python scripts/run_local_skipped_examples.py --dry-run --verbose
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import yaml

# ── ANSI colours ──────────────────────────────────────────────────────────────
USE_COLOUR = sys.stdout.isatty() and os.name != "nt"
GREEN  = "\033[32m" if USE_COLOUR else ""
RED    = "\033[31m" if USE_COLOUR else ""
YELLOW = "\033[33m" if USE_COLOUR else ""
RESET  = "\033[0m"  if USE_COLOUR else ""
BOLD   = "\033[1m"  if USE_COLOUR else ""


def find_skipped_examples(root: Path) -> list[tuple[Path, str]]:
    """Walk *root* and collect (py_path, readme_path) pairs for every skip-run entry."""
    entries = []
    for readme in sorted(root.rglob("README.md")):
        folder = readme.parent
        text = readme.read_text(encoding="utf-8")
        m = re.search(r'<!--\s*gen-gallery:skip-run\s*(.*?)-->', text, re.DOTALL)
        if not m:
            continue
        data = yaml.safe_load(m.group(1)) or {}
        for filename in data.get("skip-run", []) or []:
            py_path = folder / filename
            if py_path.exists():
                entries.append((py_path, readme))
            else:
                print(f"{YELLOW}WARNING:{RESET} {py_path} listed in skip-run but not found")
    return entries


def run_and_save(py_path: Path, timeout: int, verbose: bool) -> tuple[bool, str]:
    """Execute *py_path*, saving each plt.show() call as a PNG next to the source.

    Returns (success, summary_message).
    """
    stem = py_path.stem
    source_dir = str(py_path.parent)

    wrapper_code = textwrap.dedent(f"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        import os as _os

        _source_dir = {source_dir!r}
        _stem       = {stem!r}
        _fig_index  = [0]

        def _save_open_figs():
            for _fn in _plt.get_fignums():
                _fig_index[0] += 1
                _idx = _fig_index[0]
                _name = _os.path.join(_source_dir,
                                      _stem + ('.png' if _idx == 1 else f'_{{_idx}}.png'))
                _plt.figure(_fn).savefig(_name, dpi=150, bbox_inches='tight')
                print(f"  [saved: {{_name}}]")
            _plt.close('all')

        # patch plt.show so every show() call triggers a save
        _plt.show = _save_open_figs

        import runpy
        runpy.run_path({str(py_path)!r}, run_name='__main__')

        # catch any figures that weren't followed by plt.show()
        if _plt.get_fignums():
            _save_open_figs()
    """).strip()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                     prefix='_skip_wrap_') as tf:
        tf.write(wrapper_code)
        wrapper_path = tf.name

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    start = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, wrapper_path],
            capture_output=True, text=True,
            timeout=timeout, env=env,
            cwd=str(py_path.parent),
        )
        elapsed = time.perf_counter() - start
        ok = result.returncode == 0

        output = ""
        if result.stdout:
            output += result.stdout.strip()
        if result.stderr:
            output = (output + "\n\n" + result.stderr.strip()) if output else result.stderr.strip()

        if verbose or not ok:
            for line in output.splitlines():
                print("    " + line)

        msg = f"{'PASS' if ok else 'FAIL'}  ({elapsed:.1f}s)"
        return ok, msg

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return False, f"TIMEOUT after {timeout}s"
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return False, f"ERROR: {exc}"
    finally:
        try:
            os.unlink(wrapper_path)
        except OSError:
            pass


def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    default_root = repo_root / "coding"

    parser = argparse.ArgumentParser(
        description="Run locally all examples excluded from the automated gallery build."
    )
    parser.add_argument("--root", type=Path, default=default_root,
                        help=f"Root directory to search (default: {default_root})")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Seconds before a script is killed (default: 300)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print which scripts would be run without executing them")
    parser.add_argument("--verbose", action="store_true",
                        help="Print stdout/stderr for every script, not just failures")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"{RED}Error:{RESET} '{root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    entries = find_skipped_examples(root)
    if not entries:
        print(f"{YELLOW}No skip-run entries found under '{root}'.{RESET}")
        sys.exit(0)

    total = len(entries)
    print(f"{BOLD}Found {total} skip-run script(s) under '{root}'{RESET}")
    if args.dry_run:
        print(f"  {YELLOW}dry-run mode — no scripts will be executed{RESET}")
    print(f"  timeout per script: {args.timeout}s")
    print()

    width = max(len(str(p.relative_to(root))) for p, _ in entries)
    passed = failed = 0
    results = []

    for py_path, readme in entries:
        rel = str(py_path.relative_to(root))
        print(f"  {rel:<{width}}  ", end="", flush=True)

        if args.dry_run:
            print(f"{YELLOW}(skipped){RESET}")
            continue

        ok, msg = run_and_save(py_path, args.timeout, args.verbose)
        if ok:
            passed += 1
            print(f"{GREEN}{msg}{RESET}")
        else:
            failed += 1
            print(f"{RED}{msg}{RESET}")
        results.append((rel, ok))

    if args.dry_run:
        return

    print()
    print("─" * (width + 30))
    print(f"{BOLD}Results:{RESET}  "
          f"{GREEN}{passed} passed{RESET}  |  "
          f"{RED}{failed} failed{RESET}  |  "
          f"{total} total")

    if failed:
        print(f"\n{BOLD}Failed scripts:{RESET}")
        for rel, ok in results:
            if not ok:
                print(f"  {RED}✗{RESET}  {rel}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All skip-run scripts passed.{RESET}")


if __name__ == "__main__":
    main()
