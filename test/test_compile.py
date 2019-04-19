import os
import os.path
import unittest

from cmdstanpy.cmds import *

examples_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "test", "files"))

class CompileTest(unittest.TestCase):

    def test_compile_good(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = compile_model(stan)
        self.assertEqual("bernoulli", model.name)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_compile_bad(self):
        stan = os.path.join(examples_path, "bbad.stan")
        with self.assertRaises(Exception):
            model = compile_model(stan)

if __name__ == '__main__':
    unittest.main()
