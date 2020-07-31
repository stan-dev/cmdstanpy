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
        self.assertEqual(opts.compose(), [])
        self.assertEqual(
            opts.__repr__(), 'stanc_options=None, cpp_options=None'
        )

        stanc_opts = {}
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertEqual(opts.compose(), [])

        cpp_opts = {}
        opts = CompilerOptions(cpp_options=cpp_opts)
        opts.validate()
        self.assertEqual(opts.compose(), [])

        opts = CompilerOptions(stanc_options=stanc_opts, cpp_options=cpp_opts)
        opts.validate()
        self.assertEqual(opts.compose(), [])
        self.assertEqual(opts.__repr__(), 'stanc_options={}, cpp_options={}')

    def test_opts_stanc(self):
        stanc_opts = {}
        opts = CompilerOptions()
        opts.validate()
        self.assertEqual(opts.compose(), [])

        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertEqual(opts.compose(), [])

        stanc_opts['warn-uninitialized'] = True
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertEqual(opts.compose(), ['STANCFLAGS+=--warn-uninitialized'])

        stanc_opts['name'] = 'foo'
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertEqual(
            opts.compose(),
            ['STANCFLAGS+=--warn-uninitialized', 'STANCFLAGS+=--name=foo'],
        )

    def test_opts_stanc_opencl(self):
        stanc_opts = {}
        stanc_opts['use-opencl'] = 'foo'
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertEqual(
            opts.compose(), ['STANCFLAGS+=--use-opencl', 'STAN_OPENCL=TRUE']
        )

    def test_opts_stanc_ignore(self):
        stanc_opts = {}
        stanc_opts['auto-format'] = True
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        self.assertEqual(opts.compose(), [])

    def test_opts_stanc_includes(self):
        path2 = os.path.join(HERE, 'data', 'optimize')
        paths_str = ','.join([DATAFILES_PATH, path2]).replace('\\', '/')
        expect = 'STANCFLAGS+=--include_paths=' + paths_str

        stanc_opts = {'include_paths': paths_str}
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        opts_list = opts.compose()
        self.assertTrue(expect in opts_list)

        stanc_opts = {'include_paths': [DATAFILES_PATH, path2]}
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        opts_list = opts.compose()
        self.assertTrue(expect in opts_list)

    def test_opts_add_include_paths(self):
        expect = 'STANCFLAGS+=--include_paths=' + DATAFILES_PATH.replace(
            '\\', '/'
        )
        stanc_opts = {'warn-uninitialized': True}
        opts = CompilerOptions(stanc_options=stanc_opts)
        opts.validate()
        opts_list = opts.compose()
        self.assertTrue(expect not in opts_list)

        opts.add_include_path(DATAFILES_PATH)
        opts.validate()
        opts_list = opts.compose()
        self.assertTrue(expect in opts_list)

        path2 = os.path.join(HERE, 'data', 'optimize')
        paths_str = ','.join([DATAFILES_PATH, path2]).replace('\\', '/')
        expect = 'STANCFLAGS+=--include_paths=' + paths_str
        opts.add_include_path(path2)
        opts.validate()
        opts_list = opts.compose()
        self.assertTrue(expect in opts_list)

    def test_opts_cpp(self):
        cpp_opts = {}
        opts = CompilerOptions(cpp_options=cpp_opts)
        opts.validate()
        self.assertEqual(opts.compose(), [])

        cpp_opts['STAN_MPI'] = 'TRUE'
        opts = CompilerOptions(cpp_options=cpp_opts)
        opts.validate()
        self.assertEqual(opts.compose(), ['STAN_MPI=TRUE'])

        cpp_opts['STAN_BAD'] = 'TRUE'
        opts = CompilerOptions(cpp_options=cpp_opts)
        with self.assertRaises(ValueError):
            opts.validate()

    def test_opts_cpp_opencl(self):
        cpp_opts = {'OPENCL_DEVICE_ID': 1}
        opts = CompilerOptions(cpp_options=cpp_opts)
        opts.validate()
        opts_list = opts.compose()
        self.assertTrue('STAN_OPENCL=TRUE' in opts_list)
        self.assertTrue('OPENCL_DEVICE_ID=1' in opts_list)

        cpp_opts = {'OPENCL_DEVICE_ID': 'BAD'}
        opts = CompilerOptions(cpp_options=cpp_opts)
        with self.assertRaises(ValueError):
            opts.validate()

        cpp_opts = {'OPENCL_DEVICE_ID': -1}
        opts = CompilerOptions(cpp_options=cpp_opts)
        with self.assertRaises(ValueError):
            opts.validate()

        cpp_opts = {'OPENCL_PLATFORM_ID': 'BAD'}
        opts = CompilerOptions(cpp_options=cpp_opts)
        with self.assertRaises(ValueError):
            opts.validate()

        cpp_opts = {'OPENCL_PLATFORM_ID': -1}
        opts = CompilerOptions(cpp_options=cpp_opts)
        with self.assertRaises(ValueError):
            opts.validate()

    def test_opts_add(self):
        stanc_opts = {'warn-uninitialized': True}
        cpp_opts = {'STAN_OPENCL': 'TRUE', 'OPENCL_DEVICE_ID': 1}
        opts = CompilerOptions(stanc_options=stanc_opts, cpp_options=cpp_opts)
        opts.validate()
        opts_list = opts.compose()
        self.assertTrue('STAN_OPENCL=TRUE' in opts_list)
        self.assertTrue('OPENCL_DEVICE_ID=1' in opts_list)
        new_opts = CompilerOptions(
            cpp_options={'STAN_OPENCL': 'FALSE', 'OPENCL_DEVICE_ID': 2}
        )
        opts.add(new_opts)
        opts_list = opts.compose()
        self.assertTrue('STAN_OPENCL=FALSE' in opts_list)
        self.assertTrue('OPENCL_DEVICE_ID=2' in opts_list)

        expect = 'STANCFLAGS+=--include_paths=' + DATAFILES_PATH.replace(
            '\\', '/'
        )
        stanc_opts2 = {'include_paths': DATAFILES_PATH}
        new_opts2 = CompilerOptions(stanc_options=stanc_opts2)
        opts.add(new_opts2)
        opts_list = opts.compose()
        self.assertTrue(expect in opts_list)

        path2 = os.path.join(HERE, 'data', 'optimize')
        expect = 'STANCFLAGS+=--include_paths=' + ','.join(
            [DATAFILES_PATH, path2]
        ).replace('\\', '/')
        stanc_opts3 = {'include_paths': path2}
        new_opts3 = CompilerOptions(stanc_options=stanc_opts3)
        opts.add(new_opts3)
        opts_list = opts.compose()
        self.assertTrue(expect in opts_list)


if __name__ == '__main__':
    unittest.main()
