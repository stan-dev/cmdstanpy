"""Compiler options tests"""

import os
import unittest

from cmdstanpy.compiler_opts import CompilerOptions

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class CompilerOptsTest(unittest.TestCase):
    def test_opts_empty(self):
        opts = CompilerOptions()
        opts.validate()
        self.assertTrue(len(opts.compose()) == 0)
        self.assertEqual(
            opts.__repr__(),
            'CompilerOptions(stanc_options=None, cpp_options=None)',
        )

        stanc_opts = {}
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertTrue(len(opts.compose()) == 0)

        cpp_opts = {}
        opts = CompilerOptions(cpp_options=cpp_opts)
        opts.validate()
        self.assertTrue(len(opts.compose()) == 0)

        opts = CompilerOptions(stanc_options=stanc_opts, cpp_options=cpp_opts)
        opts.validate()
        self.assertTrue(len(opts.compose()) == 0)
        self.assertEqual(
            opts.__repr__(), 'CompilerOptions(stanc_options={}, cpp_options={})'
        )

    def test_opts_stanc(self):
        stanc_opts = {'warn-uninitialized': True}
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertTrue(len(opts.compose()) == 1)
        self.assertEqual(opts.compose(), ['STANCFLAGS+=--warn-uninitialized'])

        stanc_opts['name'] = 'foo'
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertTrue(len(opts.compose()) == 2)
        self.assertEqual(
            opts.compose(),
            ['STANCFLAGS+=--warn-uninitialized', 'STANCFLAGS+=--name=foo'],
        )


if __name__ == '__main__':
    unittest.main()
