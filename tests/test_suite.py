# Original Code
# https://github.com/PyTables/PyTables/blob/master/tables/tests/test_suite.py
# See https://github.com/graykode/matorage/blob/0.1.0/NOTICE
# modified by TaeHwan Jung(@graykode)

import sys
import unittest

from matorage.utils import is_torch_available, is_tf_available


def suite():
    test_modules = [
        "tests.test_datasaver",
    ]
    if is_torch_available():
        test_modules.extend([
            "tests.test_torch_data",
            "tests.test_torch_model",
            "tests.test_torch_optimizer",
        ])
    if is_tf_available():
        test_modules.extend([
            "tests.test_tf_data",
            "tests.test_tf_model",
            "tests.test_tf_optimizer",
        ])

    alltests = unittest.TestSuite()
    for name in test_modules:
        # Unexpectedly, the following code doesn't seem to work anymore
        # in python 3
        # exec('from %s import suite as test_suite' % name)
        __import__(name)
        test_suite = sys.modules[name].suite
        alltests.addTest(test_suite())
    return alltests


def test(verbose=False):
    result = unittest.TextTestRunner(verbosity=1 + int(verbose)).run(suite())
    if result.wasSuccessful():
        return 0
    else:
        return 1

if __name__ == '__main__':
    test()
