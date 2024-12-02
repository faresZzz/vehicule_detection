
import os
import numpy as np
import cv2
import warnings
from enum import Enum
import matplotlib.pyplot as plt

class Color(Enum):
    COLOR = cv2.COLOR_BGR2RGB
    GRAY = cv2.COLOR_BGR2GRAY

class FrameVideoHandler:
    def __init__(self, working_path: str = "."):
        # path where the images will be saved
        self.working_path = working_path

    def _path(self, path: str = None) -> str: 
        # if no path is provided, use the working path
        if not path: 
            warnings.warn(f"No path provided !! Using {self.working_path}")
            path = self.working_path

         # check if the path ends with a "/"
        path += "/" if path[-1] != "/" else ""

        return path

    def check_path(self, path:str = None) -> bool: 
        # check if path provided
        path = self._path(path)
        # check if the path exists
        if not os.path.exists(path):
            # if not, raise a warning
            warnings.warn(f"This path: {path} does not exist")
            return False 
        
        # check if the path is a directory
        assert os.path.isdir(path) == True
    
        return True
           
    def create_path(self, path:str = None) -> bool: 
        # check if path provided
        path = self._path(path)

        # if the path does not exist, create it
        os.makedirs(path, exist_ok=True)
        # check if the path was created
        if created := self.check_path(path=path): 
            print(f"Path created: {path}")

        return created
    
    def load_frame(self, path_in : str = None, name : str = None) -> np.ndarray: 
        # get the path
        path_in = self._path(path_in)

        # check if the path exists else raise an error 
        if not self.check_path(path_in):
            raise FileNotFoundError(f"This file {path_in} does not exist")
        
        # load the image
        img = cv2.imread(path_in + name)
        
        # check if the image was loaded
        if img is None: 
            raise ValueError(f"Error image cannot be oppened. Image {path_in + name} ")
        
        print(" Image loaded")
        return img
    
    def load_frames(self, path_in : str = None, names : list[str] = None) -> list[np.ndarray]:
        # get the path
        path_in = self._path(path_in)

        # check if the path exists else raise an error
        if not self.check_path(path_in):
            raise FileNotFoundError(f"This dir {path_in} does not exist")

        # get the name of all files in the frames dir
        frames_name =  os.listdir(path_in) if not names else names
        # sort files names 

        frames_name = sorted(frames_name)
        # load all images 
        imgs = [cv2.imread(path_in + frame_name) for frame_name in frames_name]

        print(f"Loaded {len(imgs)} images")

        return imgs

    def save_frame(self, image:np.ndarray, path_out: str = None, name : str = None): 
        # check if name for the image is provided else default to "img"
        if not name: 
            name = "img"
        
        # get the path
        path_out = self._path(path_out)
        
        # check if the path exists else create it
        if not self.check_path(path_out):
            self.create_path(path_out)
            print(f"Directory created at {path_out}")


        # save the image
        saved = cv2.imwrite(path_out + name, image) 

        # check if the image was saved
        if not saved: 
            raise IOError(f"Failed to write image to {path_out}. Check if the directory exists and you have write permissions.")

        print(f" Image saved at: {path_out}, with name: {name}")

    def save_frames(self, images:list[np.ndarray], path_out : str = None, name : str = None) -> None:

        # check if name for the images is provided else default to "img"
        if not name:
            name = "img.png"

        name, extension = name.split(".") if "." in name else (name, "png")
        # save all images using the save_frame method
        for i, image in enumerate(images):
            self.save_frame(image, path_out, f"{name}_{i}.{extension}" )

        print(f"Saved {len(images)} under the name {name}*.{extension} in dir {path_out}")
    
    def load_video(self, path_in : str = None, name : str = None) -> cv2.VideoCapture: 

        # get path
        path_in = self._path(path_in)

        # check if the path exists else raise an error
        if not self.check_path(path_in):
            raise FileNotFoundError(f"This dir {path_in} does not exist")


        # load the video
        video = cv2.VideoCapture(path_in + name)

        # check if the video was loaded
        if not video.isOpened():
            raise FileNotFoundError(f"Impossible de charger la vidéo : '{path_in + name}'")

        print("Video loaded")
        return video
    
    def image2video(self, images : list[np.ndarray], path_out : str = None, name : str = None, fps : float = 14) -> None: 
        
        # check if name for the video is provided else default to "video.mp4"
        if not name:
            name = "video.mp4"

        
        # get the path
        path_out = self._path(path_out)
        # check if the path exists else create it
        if not self.check_path(path_out):
            self.create_path(path_out)
            print(f"Directory created at {path_out}")

        # get the height, width and chanel of the first image
        height, width, chanel = images[0].shape
        size = (width,height)

        # create the video writer
        out = cv2.VideoWriter(path_out + name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for frame in images:
            # check if the frame size is the same as the first frame
            if frame.shape != (height, width, chanel): 
                warnings.warn("Frame size not mathching: {frame.shape}. Passing frame")
                continue

            # writing to a image array
            out.write(frame)
        # release the video writer
        out.release()
        print(f"Video created and saved at {path_out + name}")

    def video2image(self, video : cv2.VideoCapture = None, path_out : str = None, skip_frames : int = 1 ):
        
        # check if video object is provided and opened
        if not video.isOpened(): 
            raise ValueError("Video can not be oppened")
        
        # Used as counter variable 
        i = 0
        # checks whether frames were extracted 
        success = True
        
        images = []
        while success: 
    

            if i % skip_frames == 0: 
                # Read the video file
                success, image = video.read() 

                # check if the frame was read
                if not success: 
                    warnings.warn(" Error reading video passing a frame")
                    continue
                # check if the path exists meaning the frames will be saved
                if path_out:
                    # Saves the frames with frame-count 
                    self.save_frames(path_out, [image])
                images.append(image)
        
            i += 1
        
        return images

    def show_image(self, image : np.ndarray, color : Color ) -> None: 
        # plot the image
        plt.imshow(cv2.cvtColor(image, color.value))
        plt.title(f"Frame: {str(image)}")
        plt.show(block=False)  # plot image without blocking the code
        plt.pause(3)  # wait 3 seconds
        plt.close()  # close the plot
