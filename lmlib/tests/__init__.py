"""
lmlib.tests
===========
Test suite for lmlib.  Entry point: ``lmlib.test()``.

"""
import os


def test(verbosity=1, failfast=False):
    """
    Run the lmlib test suite.

    Uses pytest if available (so both pytest-style and unittest-style test
    modules are collected), and falls back to unittest discovery otherwise.

    Note: some test modules (e.g. ``test_alssm_poly_meixner``) are written in
    pytest style (parametrized classes, fixtures) and are NOT collected by
    plain unittest discovery — run with pytest to include them.

    Parameters
    ----------
    verbosity : int, optional
        0 = silent, 1 = dots (default), 2 = verbose test names.
    failfast : bool, optional
        Stop on the first failure if True.

    Returns
    -------
    bool
        True if all tests passed (or were skipped), False if any failed.

    Examples
    --------
    >>> import lmlib
    >>> lmlib.test()
    >>> lmlib.test(verbosity=2)
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        import pytest
    except ImportError:
        pytest = None

    if pytest is not None:
        args = [test_dir]
        if verbosity <= 0:
            args.append('-q')
        elif verbosity >= 2:
            args.append('-v')
        if failfast:
            args.append('-x')
        # pytest exit code 0 = all passed, 5 = no tests collected
        return pytest.main(args) == 0

    # ---- fallback: unittest discovery (skips pytest-only test modules) ----
    import unittest
    top_level = os.path.dirname(test_dir)  # the lmlib/ package dir (contains tests/)
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=test_dir,
        pattern='test_*.py',
        top_level_dir=top_level,
    )
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    return runner.run(suite).wasSuccessful()
