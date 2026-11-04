"""The full page's glance block is as hot as the generator (2026-11-04).

``_generator.py`` is re-compiled from disk on every in-run rebuild, but the
glance block lives in ``_short.py``, which the saver process imported once at
job start. A pulled change to it (the 200/50-row toggle, f1c75f00) therefore
never reached the running 12mo job's page. ``_fresh_short_module`` reloads
``_short`` from disk on each build.
"""
import sys
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.monitor import _short as sh
from lisatools.globalfit.monitor._short import _generator_functions

NAME = "lisatools.globalfit.monitor._short"


def compile_ns():
    ns = {"np": np, "MISSING": []}
    _generator_functions(ns, names=("_fresh_short_module",), consts=())
    return ns


class FreshShortModuleTest(unittest.TestCase):
    def tearDown(self):
        # leave a clean, real _short behind for the rest of the suite
        import importlib
        importlib.reload(sys.modules[NAME])

    def test_reload_replaces_a_stale_in_memory_copy(self):
        ns = compile_ns()
        mod = sys.modules[NAME]
        real = mod.build_glance
        mod.build_glance = "stale copy from job start"
        mod.history_windows = None
        got = ns["_fresh_short_module"]()
        self.assertIs(got, sys.modules[NAME])
        self.assertTrue(callable(got.build_glance))
        self.assertEqual(got.build_glance.__qualname__, real.__qualname__)
        self.assertTrue(callable(got.history_windows))
        self.assertEqual(ns["MISSING"], [])

    def test_a_broken_reload_keeps_the_loaded_copy_and_says_so(self):
        ns = compile_ns()
        import importlib
        with mock.patch.object(importlib, "reload", side_effect=SyntaxError("bad edit")):
            got = ns["_fresh_short_module"]()
        self.assertIs(got, sh)
        self.assertTrue(callable(got.build_glance))
        self.assertEqual(len(ns["MISSING"]), 1)
        self.assertIn("did not reload", ns["MISSING"][0])

    def test_generator_takes_the_glance_names_from_the_fresh_module(self):
        from lisatools.globalfit.monitor import generator_path
        src = open(generator_path(), encoding="utf-8").read()
        self.assertNotIn("from lisatools.globalfit.monitor._short import GLANCE_CSS", src)
        self.assertIn("_short_mod = _fresh_short_module()", src)


if __name__ == "__main__":
    unittest.main()
