import unittest
from rdd_detector.scores import image_score, combine_scores, classify


class ScoreTests(unittest.TestCase):
    def test_background_cancellation(self):
        self.assertAlmostEqual(image_score(0.4, 0.3, 0.2, 0.1), 0.05)
        self.assertAlmostEqual(image_score(0.7, 0.6, 0.5, 0.4), 0.05)

    def test_notebook_examples_with_rounded_inputs(self):
        self.assertAlmostEqual(combine_scores(0.01452, 0.15851), 0.0003648, delta=1e-7)
        self.assertAlmostEqual(combine_scores(-0.01480, 0.16756), -0.0004156, delta=1e-7)

    def test_threshold_boundary(self):
        self.assertEqual(classify(0.0001, 0.0001), "ai_generated")
        self.assertEqual(classify(0.0001001, 0.0001), "real")


if __name__ == "__main__":
    unittest.main()
