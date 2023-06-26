import unittest
import os
import cv2

class TestImageLoading(unittest.TestCase):
    """
    This is just a simple test template..
    """
    def setUp(self):
        self.root_dir=  os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
        self.test_data_dir = os.path.join(self.root_dir, "data")

    def test_load_test_images(self):
        test_image1_path = os.path.join(self.test_data_dir, "i0201a.png")
        test_image2_path = os.path.join(self.test_data_dir, "i0201b.png")

        test_image1 = cv2.imread(test_image1_path)
        test_image2 = cv2.imread(test_image2_path)

        self.assertIsNotNone(test_image1, "Could not load i0201a.png")
        self.assertIsNotNone(test_image2, "Could not load i0201b.png")

if __name__ == "__main__":
    unittest.main()
