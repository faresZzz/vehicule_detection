import unittest
import os
import shutil
import numpy as np
import cv2
from vehicule_detection.utils import FrameVideoHandler, Color

class TestFrameVideoHandler(unittest.TestCase):

    def setUp(self):
        self.working_path = "test_dir"
        self.handler = FrameVideoHandler(working_path=self.working_path)
        
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image_name = "test_image.png"
        
        self.test_dir = "test_dir/"

        os.makedirs(self.working_path, exist_ok=True)

        cv2.imwrite(self.working_path + "/" + self.test_image_name, self.test_image)
        self.create_video()

    
    def create_video(self): 
        self.test_video_name = "test_video.mp4"
        self.video_length = 0
        images = [self.test_image for _ in range(10)]
        height, width, chanel = images[0].shape
        size = (width,height)

        # create the video writer
        out = cv2.VideoWriter(self.working_path + "/" + self.test_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

        for frame in images:
            # writing to a image array
            out.write(frame)
            self.video_length += 1
        # release the video writer
        out.release()

    def tearDown(self):
        if os.path.exists(self.working_path):
            shutil.rmtree(self.working_path)

    def test_path_with_provided_path(self):
        path = "provided_path"
        result = self.handler._path(path)
        self.assertEqual(result, path + "/")

    def test_path_without_provided_path(self):
        with self.assertWarns(UserWarning):
            result = self.handler._path()
        self.assertEqual(result, self.working_path + "/")

    def test_check_path(self):
        self.assertTrue(self.handler.check_path(self.test_dir))
        self.assertFalse(self.handler.check_path("non_existent_dir"))

    def test_create_path(self):
        new_dir = self.working_path +"/new_test_dir"
        self.handler.create_path(new_dir)
        self.assertTrue(os.path.exists(new_dir))
        os.rmdir(new_dir)

    def test_load_frame(self):
        img = self.handler.load_frame(self.test_dir, "test_image.png")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (100, 100, 3))

    def test_load_frames(self):
        imgs = self.handler.load_frames(self.test_dir)
        self.assertIsInstance(imgs, list)
        self.assertEqual(len(imgs), 2)
        self.assertEqual(imgs[0].shape, (100, 100, 3))

    def test_save_frame(self):
        save_path = "test_dir/saved_image.png"
        self.handler.save_frame(self.test_image, self.test_dir, "saved_image.png")
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_save_frames(self):
        save_path = "test_dir/saved_image_0.png"
        self.handler.save_frames([self.test_image], self.test_dir, "saved_image.png")
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_load_video(self):
        # # Create a dummy video file
        # test_video_file = "video.mp4"
        # out = cv2.VideoWriter(self.test_dir + test_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1, (100, 100))
        # for _ in range(10):
        #     out.write(self.test_image)
        # out.release()
        video = self.handler.load_video(self.test_dir, self.test_video_name)
        self.assertTrue(video.isOpened())
        video.release()

    def test_image2video(self):
        self.handler.image2video([self.test_image for _ in range(10)], self.test_dir, "video.mp4", 10)
        self.assertTrue(os.path.exists(self.test_dir + "video.mp4"))

    def test_video2image(self):
        video = self.handler.load_video(self.test_dir, self.test_video_name)
        images = self.handler.video2image(video)
        self.assertIsInstance(images, list)
        self.assertEqual(len(images), self.video_length)
        self.assertEqual(images[0].shape, (100, 100, 3))
        video.release()

    def test_show_image(self):
        # This test is visual and should be run manually to check the output
        self.handler.show_image(self.test_image, Color.COLOR)

if __name__ == '__main__':
    unittest.main()