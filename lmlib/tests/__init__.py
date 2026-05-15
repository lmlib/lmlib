"""
lmlib.tests
===========
Test suite for lmlib.  Entry point: ``lmlib.test()``.

Individual test modules
-----------------------
test_RLSAlssm    — RLSAlssm.filter / minimize_x / eval_errors
test_CostSegment — CostSegment construction and properties
test_Trajectory  — Trajectory.eval / eval_y
test_rls         — end-to-end regression + backend comparison
test_nd_split    — per-ALSSM splitting correctness for NDCompositeCost
"""
import unittest
import os


def test(verbosity=1, failfast=False):
    """
    Run the lmlib test suite.

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
    test_dir  = os.path.dirname(os.path.abspath(__file__))
    top_level = os.path.dirname(test_dir)  # the lmlib/ package dir (contains tests/)
    loader = unittest.TestLoader()
    suite  = loader.discover(
        start_dir=test_dir,
        pattern='test_*.py',
        top_level_dir=top_level,
    )
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    return result.wasSuccessful()