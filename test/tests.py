import unittest
from .test_utils import TestFrameVideoHandler
from .test_vehicule_detector import TestVehiculeDetector

class Tests(unittest.TestCase):
    def setUp(self):
        self.tester_handler = TestFrameVideoHandler()
        self.tester_vehicle_detector = TestVehiculeDetector()


if __name__ == '__main__':
    unittest.main()