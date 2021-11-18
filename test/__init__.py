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

    # recipe from https://stackoverflow.com/a/34333710
    # pylint: disable=no-self-use
    @contextlib.contextmanager
    def modified_environ(self, *remove, **update):
        """
        Temporarily updates the ``os.environ`` dictionary in-place.

        The ``os.environ`` dictionary is updated in-place so that
        the modification is sure to work in all situations.

        :param remove: Environment variables to remove.
        :param update: Dictionary of environment variables and values to
             add/update.
        """
        env = os.environ
        update = update or {}
        remove = remove or []

        # List of environment variables being updated or removed.
        stomped = (set(update.keys()) | set(remove)) & set(env.keys())
        # Environment variables and values to restore on exit.
        update_after = {k: env[k] for k in stomped}
        # Environment variables and values to remove on exit.
        remove_after = frozenset(k for k in update if k not in env)

        try:
            env.update(update)
            for k in remove:
                env.pop(k, None)
            yield
        finally:
            env.update(update_after)
            for k in remove_after:
                env.pop(k)

    # pylint: disable=invalid-name
    def assertPathsEqual(self, path1, path2):
        """Assert paths are equal after normalization"""
        self.assertTrue(os.path.samefile(path1, path2))
