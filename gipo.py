import cv2 as cv
import numpy as np
import OPi.GPIO as GPIO

relay_ch_left = 4
relay_ch_right = 17

# Инициализация GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

def createPath(img):
    h, w = img.shape[:2]
    return (np.zeros((h, w, 3), np.uint8), h, w)

if __name__ == '__main__':
    def callback(*arg):
        print(arg)

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
if not flag:
    raise IOError("Failed to capture initial frame")

path, h, w = createPath(img)

GPIO.setup(relay_ch_left, GPIO.OUT)
GPIO.setup(relay_ch_right, GPIO.OUT)
end_left = int(0)
end_right = int(0)

while True:
    flag, img = cap.read()
    if not flag:
        print("Failed to capture image")
        break

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, hsv_min, hsv_max)

    # Выводим отладочную информацию
    print("Threshold sum:", np.sum(thresh))
    if np.sum(thresh) == 0:
        print("No pixels found in the given HSV range")

    # Находим контуры
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        moments = cv.moments(contour)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea > 100:
            x = int(dM10 / dArea)
            y = int(dM01 / dArea)
            cv.circle(img, (x, y), 10, (0, 0, 255), -1)
            if x < w // 2:  # Объект в левой половине
                if end_left == 0:
                    GPIO.output(relay_ch_left, GPIO.HIGH)
                    end_left = 1
            else:  # Объект в правой половине
                if end_right == 0:
                    GPIO.output(relay_ch_right, GPIO.HIGH)
                    end_right = 1

    # Сбрасываем реле, если объектов нет
    if end_left == 1 and all(cv.moments(cnt)['m00'] <= 100 for cnt in contours if cv.boundingRect(cnt)[0] < w // 2):
        GPIO.output(relay_ch_left, GPIO.LOW)
        end_left = 0

    if end_right == 1 and all(cv.moments(cnt)['m00'] <= 100 for cnt in contours if cv.boundingRect(cnt)[0] >= w // 2):
        GPIO.output(relay_ch_right, GPIO.LOW)
        end_right = 0

    img = cv.add(img, path)
    cv.imshow('result', img)
    ch = cv.waitKey(5)
    if ch == 27:
        break

GPIO.cleanup()
cap.release()
cv.destroyAllWindows()
