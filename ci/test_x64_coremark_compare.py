import unittest
from ci.x64_coremark_compare import score_from_output, throughput_from_output

class CoreMarkScoreTest(unittest.TestCase):
    def test_execution_uses_elapsed_work_and_rejects_missing_or_duplicate_results(self):
        self.assertEqual(throughput_from_output('measurement runs=10 elapsed_ns=2000000000\n'), 5)
        for output in ['', 'measurement runs=0 elapsed_ns=2',
                       'measurement runs=2 elapsed_ns=0',
                       'measurement runs=2 elapsed_ns=3\nmeasurement runs=2 elapsed_ns=3',
                       'measurement runs=NaN elapsed_ns=3']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                throughput_from_output(output)

    def test_one_result(self):
        self.assertEqual(score_from_output('runs=1 elapsed=20s result=[F32(25147.742)]'), 25147.742)

    def test_missing_duplicate_nonfinite_and_failed_validation(self):
        for output in ['', 'result=[F32(1)] result=[F32(2)]', 'result=[F32(NaN)]',
                       'result=[F32(inf)]', 'result=[F32(-1)]', 'result=[F32(0)]']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                score_from_output(output)
