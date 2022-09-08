import cv2
import numpy as np
import time
import json


class ImageRecognition():
    def __init__(self, game_transition: str, label: str, correct_rate: float = 0.8):
        self.game_transition = game_transition
        self.label = label
        self.threshold, self.range = self.__read_json(
            game_transition, label)
        self.correct_rate = correct_rate

    def __read_json(self, game_transition: str, label: str):
        open_json = open(f"config/json/{game_transition}.json")
        read_json = json.load(open_json)
        # print(read_json[game_transition][label]["threshold"])
        # print(read_json[game_transition][label]["range"])
        return read_json[game_transition][label]["threshold"], read_json[game_transition][label]["range"]

    def is_equal(self, image):
        binary_image = self._binary(image)
        # print(len(self.range))
        for i in range(len(self.range)):
            clipped_binary_image = self._clip_image(
                binary_image, self.range[i]["target"][0], self.range[i]["target"][1])
            if not self._image_is_equal(clipped_binary_image, self.range[i]["color"]):
                return False
        self._draw_rectangle(image)
        return True

    def _draw_rectangle(self, image):
        for i in range(len(self.range)):
            cv2.rectangle(image, (self.range[i]["target"][0]),
                          (self.range[i]["target"][1]), (255, 0, 0), 1)

    def _binary(self, image):
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray_image, self.threshold[0], self.threshold[1], cv2.THRESH_BINARY)
        # cv2.imshow('frame', binary)
        # if cv2.waitKey(0) == 27:
        #     return
        return binary

    def _clip_image(self, binary_image, min, max):
        # print(min, max)
        clipped_binary_image = binary_image[min[1]:max[1], min[0]:max[0]]
        # cv2.imshow('frame', clipped_binary_image)
        # if cv2.waitKey(1) == 27:
        #     return
        return clipped_binary_image

    def _image_is_equal(self, clipped_binary_image, prepared_binary_image):
        size = self.__check_size(prepared_binary_image)
        # print("correct-rate", end=" ")
        # print(np.count_nonzero(clipped_binary_image == prepared_binary_image)/self.__check_size(clipped_binary_image))
        # print("pre-image: ", np.array(prepared_binary_image).shape, "now-image-size: ", clipped_binary_image.shape)
        # print(clipped_binary_image)
        if np.count_nonzero(clipped_binary_image == prepared_binary_image) > self.correct_rate*size:
            return True

    def __check_size(self, image):
        size = len(image[0])*len(image)
        return size


if __name__ == "__main__":
    # movie = cv2.VideoCapture(1)
    movie = cv2.VideoCapture('./data/video/area_trim1.mp4')
    game_start = ImageRecognition("game", "starts", 0.9)

    while movie.isOpened:
        ret, frame = movie.read()
        frame = cv2.resize(frame, (768, 432))

        print(game_start.is_equal(frame))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break
        time.sleep(0.02)

    movie.release()
