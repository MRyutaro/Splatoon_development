import cv2
import time
from image_recognition_v1 import ImageRecognition


class CheckPinch(ImageRecognition):
    def is_equal(self, image):
        # print("True or False")
        # print(len(self.range)) -> 1
        binary_image = self._binary(image)
        clipped_binary_image = self._clip_image(
            binary_image, self.range[0]["target"][0], self.range[0]["target"][1])
        if self._image_is_equal(clipped_binary_image, self.range[0]["color"][0]):
            self._draw_rectangle(image)
            return True
        else:
            return False


if __name__ == "__main__":
    # movie = cv2.VideoCapture(1)
    movie = cv2.VideoCapture('./data/video/area_trim1.mp4')
    friend_is_in_pinch = CheckPinch("in_pinch", "is friend")
    enemy_is_in_pinch = CheckPinch("in_pinch", "is enemy")

    while movie.isOpened:
        ret, frame = movie.read()
        frame = cv2.resize(frame, (768, 432))

        print(friend_is_in_pinch.is_equal(frame))
        print(enemy_is_in_pinch.is_equal(frame))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break
        time.sleep(0.02)

    movie.release()
