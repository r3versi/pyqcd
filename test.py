import unittest

source_dir = './tests'

loader = unittest.TestLoader()
suite = loader.discover(source_dir)
runner = unittest.TextTestRunner()
runner.run(suite)
