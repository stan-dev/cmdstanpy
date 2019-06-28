import os
import os.path
import unittest

from cmdstanpy.lib import Model, RunSet, SamplerArgs

datafiles_path = os.path.join('test', 'data')
goodfiles_path = os.path.join(datafiles_path, 'runset-good')
badfiles_path = os.path.join(datafiles_path, 'runset-bad')


class RunSetTest(unittest.TestCase):
    def test_check_retcodes(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(goodfiles_path, 'bern')
        args = SamplerArgs(
            model,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_file=output,
            sampling_iters=100,
            max_treedepth=11,
            adapt_delta=0.95,
        )
        runset = RunSet(chains=4, args=args)
        retcodes = runset.retcodes
        self.assertEqual(4, len(retcodes))
        for i in range(len(retcodes)):
            self.assertEqual(-1, runset.retcode(i))
        runset.set_retcode(0, 0)
        self.assertEqual(0, runset.retcode(0))
        for i in range(1, len(retcodes)):
            self.assertEqual(-1, runset.retcode(i))
        self.assertFalse(runset.check_retcodes())
        for i in range(1, len(retcodes)):
            runset.set_retcode(i, 0)
        self.assertTrue(runset.check_retcodes())

    def test_validate_outputs(self):
        # construct runset using existing sampler output
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(goodfiles_path, 'bern')
        args = SamplerArgs(
            model,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_file=output,
            sampling_iters=100,
            max_treedepth=11,
            adapt_delta=0.95,
        )
        runset = RunSet(chains=4, args=args)
        retcodes = runset.retcodes
        for i in range(len(retcodes)):
            runset.set_retcode(i, 0)
        self.assertTrue(runset.check_retcodes())
        runset.check_console_msgs()
        runset.validate_csv_files()
        self.assertEqual(4, runset.chains)
        self.assertEqual(100, runset.draws)
        self.assertEqual(8, len(runset.column_names))
        self.assertEqual('lp__', runset.column_names[0])

    def test_validate_bad_transcript(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(badfiles_path, 'bad-transcript-bern')
        args = SamplerArgs(
            model,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_file=output,
            sampling_iters=100,
            max_treedepth=11,
            adapt_delta=0.95,
        )
        runset = RunSet(chains=4, args=args)
        with self.assertRaisesRegex(Exception, 'Exception'):
            runset.check_console_msgs()

    def test_validate_bad_hdr(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(badfiles_path, 'bad-hdr-bern')
        args = SamplerArgs(
            model,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_file=output,
            sampling_iters=100,
            max_treedepth=11,
            adapt_delta=0.95,
        )
        runset = RunSet(chains=4, args=args)
        retcodes = runset.retcodes
        for i in range(len(retcodes)):
            runset.set_retcode(i, 0)
        self.assertTrue(runset.check_retcodes())
        with self.assertRaisesRegex(ValueError, 'header mismatch'):
            runset.validate_csv_files()

    def test_validate_bad_draws(self):
        # construct runset using existing sampler output
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(badfiles_path, 'bad-draws-bern')
        args = SamplerArgs(
            model,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_file=output,
            sampling_iters=100,
            max_treedepth=11,
            adapt_delta=0.95,
        )
        runset = RunSet(chains=4, args=args)
        retcodes = runset.retcodes
        for i in range(len(retcodes)):
            runset.set_retcode(i, 0)
        self.assertTrue(runset.check_retcodes())
        with self.assertRaisesRegex(ValueError, 'draws'):
            runset.validate_csv_files()

    def test_validate_extra_param(self):
        # construct runset using existing sampler output
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(badfiles_path, 'bad-cols-bern')
        args = SamplerArgs(
            model,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_file=output,
            sampling_iters=100,
            max_treedepth=11,
            adapt_delta=0.95,
        )
        runset = RunSet(chains=4, args=args)
        retcodes = runset.retcodes
        for i in range(len(retcodes)):
            runset.set_retcode(i, 0)
        self.assertTrue(runset.check_retcodes())
        with self.assertRaisesRegex(ValueError, 'bad draw'):
            runset.validate_csv_files()


if __name__ == '__main__':
    unittest.main()
