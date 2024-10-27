import unittest
from keyqueries import all_expanded_queries

class TestIterateAllVariants(unittest.TestCase):
    def test_example_01(self):
        rm3_expansion = {'query': 'applypipeline:off appoint^0.056280680 total^0.300000012 correspond^0.048301216 energi^0.300000012'}
        actual = all_expanded_queries(rm3_expansion, 1)
        
        self.assertEqual(15, len(actual))
        self.assertTrue('applypipeline:off appoint^0.056280680' in actual)
        self.assertTrue('applypipeline:off total^0.300000012' in actual)

    def test_example_02(self):
        rm3_expansion = {'query': 'applypipeline:off a^1 b^2'}
        actual = all_expanded_queries(rm3_expansion, 1)
        
        self.assertEqual(3, len(actual))
        self.assertTrue('applypipeline:off a^1' in actual)
        self.assertTrue('applypipeline:off b^2' in actual)

    def test_example_03(self):
        rm3_expansion = {'query': 'applypipeline:off a^1 b^2'}
        actual = all_expanded_queries(rm3_expansion, 2)
        
        self.assertEqual(1, len(actual))
        self.assertEqual(['applypipeline:off a^1 b^2'], actual)
