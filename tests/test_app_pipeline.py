import unittest
from pathlib import Path

from app.pipeline import PipelineResult, run_pipeline


class TestAppPipeline(unittest.TestCase):
    def test_run_pipeline_with_project_images(self):
        image1 = Path("data/raw/lunar_reference/LRO/quickmap-lroc.png")
        image2 = Path("data/prepared/lunar_reference/LRO_gray.png")

        result = run_pipeline(image1, image2)

        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(hasattr(result, "success"))
        self.assertIn("stage_results", result.to_dict())
        self.assertTrue(result.stage_results["m1"]["executed"] or result.stage_results["m2"]["executed"])


if __name__ == "__main__":
    unittest.main()
