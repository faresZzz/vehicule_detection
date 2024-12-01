import math
import cv2
import numpy as np
from typing import Any
from vehicule_detection.utils import FrameVideoHandler, Color


class VehiculeDetector:

    def __init__(self, images : list[np.ndarray] = None, parameters : dict = None):

        self.images = images
        self.parameters = parameters
        # initialize the vehicle counter
        self.vehicule_counter = 0
        self.previous_vehicle_count = 0

    def set_images(self, images : list[np.ndarray] = None) -> None: 
        if images:
            assert type(images) ==  list and type(images[0]) == np.ndarray
            self.images = images

    def get_images(self) -> list[np.ndarray]: 
        return self.images

    def get_param(self, parameter_name: str) -> None | Any: 
        if not self.parameters: 
            return None

        if parameter_name in self.parameters.keys():
            return self.parameters[parameter_name]
        
        return None
    
    def set_param(self, **kwargs) -> None: 
        if not self.parameters: 
            self.parameters = {}
        
        for key, value in kwargs.items(): 
            self.parameters[key] = value
    
    def _check_param(self, parameter : Any , parameter_name : str, caller_func_name : str) -> None | Any:
        # check for a function parameter. With priority to the parameter passed to the class parameter
        # check if parameter in the class parameters
        param = self.get_param(parameter_name)

        if param is not None:
            # if parameter is not None return it
            return param
        
        if parameter is None:
            # if param is None and parameter is None raise an error
            raise ValueError(f"Parameter {parameter_name.upper()} needed for {caller_func_name.upper()} function! Please specify it using {self.set_param.__name__.upper()} method")
        return parameter
    
    def detect(self, images : list[np.ndarray] = None) -> list[int, list[np.ndarray]]: 
        # function to detect vehicules in a list of images 

        if not images: 
            # if images is not passed to the function use the images passed to the class
            images = self.images

        i = 0
        out = []
        
        # loop through all images
        while i < len(images) - 1: 

            # get 2 consecutive images
            imageA = images[i].copy()
            imageB = images[i + 1].copy()

            # convert Image in GRAY
            greyA = self.convert_grey(imageA)
            greyB = self.convert_grey(imageB)

            # Calculate the Difference between 2 images
            diff = cv2.absdiff(greyB, greyA)

            # # image thesholding
            thresh = self.threshold(diff , thresh=30, maxval=255)

            # apply dilatation to merges regions 
            kernel =  np.ones((4,4),np.uint8)
            dilated = self.dilate_close(thresh, kernel=kernel, iteration=1)

            # find contours
            # set location of the line to count vehicles from the bottom of the image avoiding unwanted contours
            y_min = imageA.shape[0] - imageA.shape[0]//3
            # set the minimum area of a contour to be considered as a vehicle
            image_area = imageA.shape[0] * imageA.shape[1]
            min_area = image_area // 100
            contours = self.find_contours(dilated, y_min=y_min, min_area=min_area)
            
            # convert contours to boxes
            boxes = self.convert_to_boxes(contours)

            # draw contours into image
            img = self.draw_countours(imageB, boxes)

            # draw line and count vehicles
            img = self.count(img, contours)

            # append image to the output list
            out.append(img)


            i += 1
        
        return self.vehicule_counter, out 

    def convert_grey(self, image : np.ndarray) -> np.ndarray: 
        # convert image to grey
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def differenciate(self, image1 : np.ndarray, image2 : np.ndarray) -> np.ndarray:
        # calculate the difference between 2 images
        diff = cv2.absdiff(image2, image1)
        return diff
        
    def threshold(self, image: np.ndarray, thresh : float = None , maxval : float = None) -> np.ndarray:
        
        # check if the parameters are passed to the function and if not check if they are in the class parameters
        thresh = self._check_param(thresh, "thresh", str(self.threshold.__name__))
        maxval = self._check_param(maxval, "maxval", str(self.threshold.__name__))
            
        # apply thresholding
        _ , th = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)

        return th

    def dilate_close(self, image : np.ndarray, kernel : np.ndarray = None, iteration : int = None) -> np.ndarray:

        # check if the parameters are passed to the function and if not check if they are in the class parameters
        kernel = self._check_param(kernel, "kernel", str(self.dilate_close.__name__))
        iteration = self._check_param(iteration, "iteration", str(self.dilate_close.__name__))

        # apply dilatation
        result = cv2.dilate(image,kernel,iterations = iteration)
        # apply closing
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)


        return result

    def find_contours(self, image : np.ndarray, y_min : float = None, min_area : float = None) -> list[np.ndarray]:
        
        # check if the parameters are passed to the function and if not check if they are in the class parameters
        y_min = self._check_param(y_min, "y_min", str(self.find_contours.__name__))
        min_area = self._check_param(min_area, "min_area", str(self.find_contours.__name__))

        # find contours
        contours, _ = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        valid = []
        for countour in contours:
            # get the bounding box of the contour
            cntr = countour
            x, y, w, h = cv2.boundingRect(countour)
            # check if the contour is valid
            contour_valid = cv2.contourArea(cntr) > min_area
            if contour_valid and y > y_min:
                valid.append(countour)
            
        return valid

    def convert_to_boxes(self, contours : list[np.ndarray]) -> list[list[float, float, float, float]]:
        # convert contours to boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, w, h])

        return boxes
    
    def draw_countours(self, image:np.ndarray, contours:list[np.ndarray]):
        # draw contours on the image
        for box in contours:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image

    def draw_line(self, image, coordinate:list[float, float]): 
        # draw a line on the image
        coordinate = self._check_param(coordinate, "coordinate", str(self.draw_line.__name__))
        cv2.line(image, (0, coordinate[0]),(coordinate[1],coordinate[0]),(100, 255, 255))
        return image
    
    def count(self, image : np.ndarray, contours:list[np.ndarray]) -> np.ndarray: 
        # count the number of vehicles in the image
        count = len(contours)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # write the number of vehicles detected on the image
        cv2.putText(image, "vehicles detected: " + str(count), (55, 15), font, 0.6, (0, 180, 0), 2)
        # update the vehicle counter
        if self.previous_vehicle_count < count:
            self.vehicule_counter += count - self.previous_vehicle_count
        
        # update variables
        self.previous_vehicle_count = count
        return image



if __name__ == "__main__":
    # test the vehicule detector
    detector_parameters = {
        "thresh" : 30,
        "maxval" : 255,
        "kernel" : np.ones((4,4),np.uint8),
        "iteration" : 1,
        "y_min" : 0,
        "min_area" : 50
    }
    # create a frame handler and VehiculeDetector
    handler = FrameVideoHandler()
    detector = VehiculeDetector(parameters=detector_parameters)
    
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Empty image
    image2 = cv2.rectangle(image1.copy(), (20, 20), (50, 50), (255, 255, 255), -1)  # White square

    handler.show_image(image1, Color.COLOR)
    handler.show_image(image2, Color.COLOR)

    frames = [image1, image2]
    detector.set_images(frames)
    detector.set_param(y_min=0)
    vehicules, out = detector.detect()

    handler.show_image(out[0], Color.COLOR)
    print(f"Total vehicules detected: {vehicules}")
    handler.show_image(out[1], Color.COLOR)


    # # Test detection on images low quality
    # frames_path = "data/raw/frames"
    # frames = handler.load_frames(frames_path)
    # frame_size = frames[0].shape[:2]
    # detector.set_param(y_min=frame_size[0] - (frame_size[0]//3 + 10))
    # image_area = frame_size[0] * frame_size[1]
    # min_area = image_area // 100
    # detector.set_param(min_area=min_area)
    # total_vehicule, out = detector.detect(frames)
    # handler.show_image(out[10], Color.COLOR)
    # handler.image2video(filename="detected_images.mp4", images=out, path_out="data/processed/videos")
    # print(f"Total vehicules detected: {total_vehicule}")


    # # Test detection on video from high quality
    # video_path = "data/raw/highway.mp4"
    # video = handler.load_video("data/raw/highway.mp4")
    # frames = handler.video2image(video=video)
    # video.release()
    # frame_size = frames[0].shape[:2]
    # detector.set_param(y_min=frame_size[0] - (frame_size[0]//3 + 50))
    # image_area = frame_size[0] * frame_size[1]
    # min_area = image_area // 100
    # detector.set_param(min_area=min_area)
    # total_vehicule, out = detector.detect(frames)
    # handler.show_image(out[10], Color.COLOR)
    # handler.image2video(filename="detected_highway.mp4", images=out, path_out="data/processed/videos")
    # print(f"Total vehicules detected: {total_vehicule}")


    # cv2.destroyAllWindows()



    

