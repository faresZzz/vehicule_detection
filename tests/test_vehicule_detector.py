import unittest
import numpy as np
import cv2
from vehicule_detection.vehicule_detector import VehiculeDetector
import matplotlib.pyplot as plt

class TestVehiculeDetector(unittest.TestCase):
    
    def setUp(self):
        # Create simulated images for testing
        self.image1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Empty image
        self.image2 = cv2.rectangle(self.image1.copy(), (20, 20), (50, 50), (255, 255, 255), -1)  # White square
        self.frames = [self.image1, self.image2]
        self.parameters = {
        "thresh" : 30,
        "maxval" : 255,
        "kernel" : np.ones((4,4),np.uint8),
        "iteration" : 1,
        "y_min" : 0,
        "min_area" : 50
        }
        self.detector = VehiculeDetector([self.image1, self.image2], parameters=self.parameters)

    def test_set_images(self):
        new_images = [self.image2]
        self.detector.set_images(new_images)
        self.assertEqual(len(self.detector.images), 1)
        self.assertTrue(np.array_equal(self.detector.images[0], self.image2))

    def test_get_images(self):
        self.assertTrue(np.array_equal(self.detector.get_images()[0], self.image1))
    
    def test_get_param(self):
        self.detector.set_param(thresh=30, maxval=255)
        self.assertEqual(self.detector.get_param("thresh"), 30)
        self.assertEqual(self.detector.get_param("maxval"), 255)
        self.assertIsNone(self.detector.get_param("non_existent_param"))

    def test_set_param(self):
        self.detector.set_param(thresh=40, maxval=245)
        self.assertEqual(self.detector.parameters["thresh"], 40)
        self.assertEqual(self.detector.parameters["maxval"], 245)
        self.detector.set_param(thresh=50)
        self.assertEqual(self.detector.parameters["thresh"], 50)  # Check that the parameter is updated
    
    def test_check_param_existing(self):
        self.detector.set_param(thresh=30)
        result = self.detector._check_param(None, "thresh", "test_check_param_existing")
        self.assertEqual(result, 30)  # Check that the parameter is correctly retrieved from the class parameters

    def test_check_param_passed(self):
        self.detector.set_param(thresh=30)
        result = self.detector._check_param(40, "thresh", "test_check_param_passed")
        self.assertEqual(result, 30)  # Check priority of class parameter over passed parameter

    def test_check_param_missing(self):
        with self.assertRaises(ValueError) as context:
            self.detector._check_param(None, "non_existent_param", "test_check_param_missing")
        self.assertTrue("Parameter NON_EXISTENT_PARAM needed for TEST_CHECK_PARAM_MISSING function!" in str(context.exception))  # Check that the correct error message is raised

    def test_convert_grey(self):
        grey_image = self.detector.convert_grey(self.image2.copy())
        self.assertEqual(len(grey_image.shape), 2)  # Check that the image is grayscale
    
    def test_threshold(self):
        grey_image = self.detector.convert_grey(self.image2.copy())
        thresholded_image = self.detector.threshold(grey_image, thresh=30, maxval=255)
        self.assertEqual(np.max(thresholded_image), 255)  # Check that the maximum value is 255 (binary)

    def test_dilate_close(self):
        grey_image = self.detector.convert_grey(self.image2.copy())
        thresholded_image = self.detector.threshold(grey_image, thresh=30, maxval=255)
        dilated_image = self.detector.dilate_close(thresholded_image, kernel=np.ones((4,4),np.uint8), iteration=1)
        self.assertGreater(np.sum(dilated_image), np.sum(thresholded_image))  # Check that the image is dilated

    def test_find_contours(self):
        grey_image = self.detector.convert_grey(self.image2.copy())
        thresholded_image = self.detector.threshold(grey_image, thresh=30, maxval=255)
        dilated_image = self.detector.dilate_close(thresholded_image, kernel=np.ones((4,4),np.uint8), iteration=1)
        contours = self.detector.find_contours(dilated_image, y_min=0, min_area=50)
        self.assertEqual(len(contours), 1)  # Check that one contour is found

    def test_convert_to_boxes(self):
        grey_image = self.detector.convert_grey(self.image2.copy())
        thresholded_image = self.detector.threshold(grey_image, thresh=30, maxval=255)
        dilated_image = self.detector.dilate_close(thresholded_image, kernel=np.ones((4,4),np.uint8), iteration=1)
        contours = self.detector.find_contours(dilated_image, y_min=0, min_area=50)
        boxes = self.detector.convert_to_boxes(contours)
        self.assertEqual(len(boxes), 1)  # Check that one box is created
        self.assertEqual(len(boxes[0]), 4)  # Check that the box has 4 coordinates

    def test_draw_countours(self):
        grey_image = self.detector.convert_grey(self.image2.copy())
        thresholded_image = self.detector.threshold(grey_image, thresh=30, maxval=255)
        dilated_image = self.detector.dilate_close(thresholded_image, kernel=np.ones((4,4),np.uint8), iteration=1)
        contours = self.detector.find_contours(dilated_image, y_min=0, min_area=50)
        boxes = self.detector.convert_to_boxes(contours)
        image_with_contours = self.detector.draw_countours(self.image2.copy(), boxes)
        self.assertTrue(np.any(image_with_contours != self.image2))  # Check that modifications have been made

    def test_count(self):
        grey_image = self.detector.convert_grey(self.image2.copy())
        thresholded_image = self.detector.threshold(grey_image, thresh=30, maxval=255)
        dilated_image = self.detector.dilate_close(thresholded_image, kernel=np.ones((4,4),np.uint8), iteration=1)
        contours = self.detector.find_contours(dilated_image, y_min=0, min_area=50)
        self.detector.count(self.image2, contours)
        self.assertEqual(self.detector.vehicule_counter, 1)  # Check that the number of detected vehicles is correct

    def test_detect(self):
        total_vehicule, _ = self.detector.detect()
        self.assertEqual(total_vehicule, 1)  # Check that the total number of detected vehicles is correct

        
if __name__ == "__main__":
    unittest.main()
