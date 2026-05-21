"""
run_all_examples.py
===================
Recursively finds and executes every Python file under a given root folder,
collects pass/fail results, and prints a summary.

Usage
-----
    python run_all_examples.py <root_folder> [options]

Options
-------
    --timeout N     Kill a script after N seconds (default: 60)
    --ignore GLOB   Glob pattern to skip (can be repeated), e.g. --ignore "*test*"
    --no-display    Set MPLBACKEND=Agg so matplotlib never opens windows
                    (enabled by default when no display is detected)
    --verbose       Print full stdout/stderr for every script, not just failures

Examples
--------
    python run_all_examples.py ./examples
    python run_all_examples.py ./examples --timeout 120 --ignore "*test*"
    python run_all_examples.py ./examples --verbose
"""

import argparse
import fnmatch
import os
import subprocess
import sys
import time


# ── ANSI colours (disabled on Windows or when output is not a terminal) ──────
USE_COLOUR = sys.stdout.isatty() and os.name != "nt"
GREEN  = "\033[32m" if USE_COLOUR else ""
RED    = "\033[31m" if USE_COLOUR else ""
YELLOW = "\033[33m" if USE_COLOUR else ""
RESET  = "\033[0m"  if USE_COLOUR else ""
BOLD   = "\033[1m"  if USE_COLOUR else ""


def collect_scripts(root: str, ignore_patterns: list[str]) -> list[str]:
    """Return all .py files under *root*, sorted, excluding ignored patterns."""
    scripts = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Sort so execution order is deterministic
        dirnames.sort()
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path  = os.path.relpath(full_path, root)
            if any(fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(fname, pat)
                   for pat in ignore_patterns):
                continue
            scripts.append(full_path)
    return scripts


def run_script(path: str, timeout: int, extra_env: dict, save_figures: bool = False, plots_dir: str = "") -> tuple[bool, float, str, str]:
    """
    Run *path* in a subprocess.

    Returns (passed, elapsed_seconds, stdout, stderr).
    """
    env = os.environ.copy()
    env.update(extra_env)

    start = time.perf_counter()
    try:
        env = env.copy()
        if save_figures:
            # Write a wrapper that patches plt.show() then runs the target
            # script via runpy so that __file__ and the working directory are
            # set correctly, just as if the script had been run directly.
            import tempfile, textwrap
            script_stem = os.path.splitext(os.path.basename(path))[0]
            wrapper_code = textwrap.dedent(f"""
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as _plt
                _fig_counter = [0]
                _plots_dir   = {plots_dir!r}
                _script_stem = {script_stem!r}
                def _patched_show(*args, **kwargs):
                    _fig_counter[0] += 1
                    for i, fig_num in enumerate(_plt.get_fignums()):
                        out = f"{{_plots_dir}}/{{_script_stem}}_fig{{_fig_counter[0]:02d}}_{{i+1:02d}}.png"
                        _plt.figure(fig_num).savefig(out, bbox_inches='tight', dpi=150)
                        print(f"  [fig saved: {{out}}]")
                    _plt.close('all')
                _plt.show = _patched_show
                import runpy
                runpy.run_path({path!r}, run_name='__main__')
            """).strip()
            wrapper = tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                                  delete=False, prefix='_mpl_wrap_')
            wrapper.write(wrapper_code)
            wrapper.flush()
            run_path = wrapper.name
        else:
            wrapper = None
            run_path = path

        result = subprocess.run(
            [sys.executable, run_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.path.dirname(path),
        )
        elapsed = time.perf_counter() - start
        passed  = result.returncode == 0
        if wrapper is not None:
            try:
                os.unlink(wrapper.name)
            except OSError:
                pass
        return passed, elapsed, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return False, elapsed, "", f"TIMEOUT after {timeout}s"
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return False, elapsed, "", f"RUNNER ERROR: {exc}"


def has_display() -> bool:
    """Heuristic: is there a graphical display available?"""
    if sys.platform == "darwin":
        return True   # macOS always has a display session
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main():
    parser = argparse.ArgumentParser(
        description="Recursively run all Python scripts in a folder and report results."
    )
    parser.add_argument("root", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"), help="Root folder to search (default: ../examples relative to this script)")
    parser.add_argument("--no-save-figures", action="store_true",
                        help="Disable saving matplotlib figures as PNG files (saving is on by default)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Seconds before a script is killed (default: 120)")
    parser.add_argument("--ignore", action="append", default=[],
                        metavar="GLOB",
                        help="Filename/path glob to skip (repeatable)")
    parser.add_argument("--no-display", action="store_true",
                        help="Force MPLBACKEND=Agg (suppress matplotlib windows)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print stdout/stderr for every script, not just failures")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"{RED}Error:{RESET} '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Suppress GUI windows when there is no display, or when requested.
    # When saving figures we use a small sitecustomize-style snippet injected
    # via PYTHONSTARTUP to patch plt.show() → save-to-PNG before each script.
    extra_env: dict[str, str] = {}
    save_figures = not args.no_save_figures
    plots_dir = os.path.join(os.getcwd(), "_plots")
    if save_figures:
        os.makedirs(plots_dir, exist_ok=True)
    if args.no_display or not has_display():
        extra_env["MPLBACKEND"] = "Agg"

    scripts = collect_scripts(root, args.ignore)
    if not scripts:
        print(f"{YELLOW}No Python scripts found under '{root}'.{RESET}")
        sys.exit(0)

    total   = len(scripts)
    passed  = 0
    failed  = 0
    results = []   # list of (path, ok, elapsed, stdout, stderr)

    print(f"{BOLD}Found {total} script(s) under '{root}'{RESET}")
    if save_figures:
        print(f"  figures: saved to '{plots_dir}'")
    elif extra_env.get("MPLBACKEND"):
        print(f"  matplotlib backend: {extra_env['MPLBACKEND']}")
    if args.ignore:
        print(f"  ignore patterns: {args.ignore}")
    print(f"  timeout per script: {args.timeout}s")
    print()

    width = max(len(os.path.relpath(s, root)) for s in scripts)

    for idx, script in enumerate(scripts, 1):
        rel = os.path.relpath(script, root)
        prefix = f"[{idx:>{len(str(total))}}/{total}]  {rel:<{width}}"
        print(prefix, end="  ", flush=True)

        ok, elapsed, stdout, stderr = run_script(script, args.timeout, extra_env, save_figures, plots_dir)

        if ok:
            passed += 1
            print(f"{GREEN}PASS{RESET}  ({elapsed:.1f}s)")
        else:
            failed += 1
            print(f"{RED}FAIL{RESET}  ({elapsed:.1f}s)")

        results.append((rel, ok, elapsed, stdout, stderr))

        # Print output immediately so long runs show progress
        if args.verbose or not ok:
            if stdout.strip():
                print("    ── stdout " + "─" * 50)
                for line in stdout.rstrip().splitlines():
                    print("    " + line)
            if stderr.strip():
                print("    ── stderr " + "─" * 50)
                for line in stderr.rstrip().splitlines():
                    print("    " + line)
            print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("─" * (width + 30))
    print(f"{BOLD}Results:{RESET}  "
          f"{GREEN}{passed} passed{RESET}  |  "
          f"{RED}{failed} failed{RESET}  |  "
          f"{total} total")

    if failed:
        print(f"\n{BOLD}Failed scripts:{RESET}")
        for rel, ok, elapsed, stdout, stderr in results:
            if not ok:
                print(f"  {RED}✗{RESET}  {rel}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All scripts passed.{RESET}")


if __name__ == "__main__":
    main()
