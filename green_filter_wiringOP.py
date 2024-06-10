import cv2 as cv
import numpy as np
import wiringpi as wp

# Заглушки для WiringOP
# class WiringOPStub:
#     OUTPUT = 1
#     HIGH = 1
#     LOW = 0
#
#     @staticmethod
#     def wiringPiSetupGpio():
#         print("WiringOP: wiringPiSetupGpio() called")
#
#     @staticmethod
#     def pinMode(pin, mode):
#         print(f"WiringOP: pinMode(pin={pin}, mode={mode}) called")
#
#     @staticmethod
#     def digitalWrite(pin, value):
#         print(f"WiringOP: digitalWrite(pin={pin}, value={value}) called")


# Подменяем WiringOP на заглушку
# wp = WiringOPStub

relay_ch_left = 4
relay_ch_right = 17

# Инициализация заглушки WiringOP
wp.wiringPiSetupGpio()


def createPath(img):
    h, w = img.shape[:2]
    return (np.zeros((h, w, 3), np.uint8), h, w)


cv.namedWindow("Result")

# Используем захват видео с камеры
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

hsv_min = np.array((46, 139, 47), np.uint8)
hsv_max = np.array((85, 205, 87), np.uint8)
lastx = 0
lasty = 0
path_color = (0, 0, 255)

flag, img = cap.read()
path, h, w = createPath(img)

wp.pinMode(relay_ch_left, wp.OUTPUT)
wp.pinMode(relay_ch_right, wp.OUTPUT)
end_left = int(0)
end_right = int(0)

while True:
    flag, img = cap.read()
    if not flag:
        print("Failed to capture image")
        break

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, hsv_min, hsv_max)

    moments = cv.moments(thresh, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']

    # print(f"dArea: {dArea}")

    if dArea > 100:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea)
        cv.circle(img, (x, y), 10, (0, 0, 255), -1)
        if x < w // 2:  # Объект в левой половине
            if end_left == 0:
                wp.digitalWrite(relay_ch_left, wp.HIGH)
                end_left = 1
            if end_right == 1:
                wp.digitalWrite(relay_ch_right, wp.LOW)
                end_right = 0
        else:  # Объект в правой половине
            if end_right == 0:
                wp.digitalWrite(relay_ch_right, wp.HIGH)
                end_right = 1
            if end_left == 1:
                wp.digitalWrite(relay_ch_left, wp.LOW)
                end_left = 0
    else:
        if end_left == 1:
            wp.digitalWrite(relay_ch_left, wp.LOW)
            end_left = 0
        if end_right == 1:
            wp.digitalWrite(relay_ch_right, wp.LOW)
            end_right = 0

    img = cv.add(img, path)
    cv.imshow('result', img)
    ch = cv.waitKey(5)
    if ch == 27:
        break

cap.release()
cv.destroyAllWindows()
