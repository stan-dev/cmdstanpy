"""Testing utilities for CmdStanPy."""

import contextlib
import os
import unittest
from importlib import reload


class CustomTestCase(unittest.TestCase):
    # pylint: disable=invalid-name
    @contextlib.contextmanager
    def assertRaisesRegexNested(self, exc, msg):
        """A version of assertRaisesRegex that checks the full traceback.

        Useful for when an exception is raised from another and you wish to
        inspect the inner exception.
        """
        with self.assertRaises(exc) as ctx:
            yield
        exception = ctx.exception
        exn_string = str(ctx.exception)
        while exception.__cause__ is not None:
            exception = exception.__cause__
            exn_string += "\n" + str(exception)
        self.assertRegex(exn_string, msg)

    # pylint: disable=no-self-use
    @contextlib.contextmanager
    def without_import(self, library, module):
        with unittest.mock.patch.dict('sys.modules', {library: None}):
            reload(module)
            yield
        reload(module)

    # pylint: disable=invalid-name
    def assertPathsEqual(self, path1, path2):
        """Assert paths are equal after normalization"""
        self.assertTrue(os.path.samefile(path1, path2))
