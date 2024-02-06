import cv2

def show_webcam():
    # Відкрити вебкамеру
    cap = cv2.VideoCapture(0)

    while True:
        # Зчитати зображення з вебкамери
        ret, frame = cap.read()

        # Показати зчитане зображення у вікні
        cv2.imshow('Webcam', frame)

        # Додати кнопку для виходу з циклу
        if cv2.waitKey(1) == 27:  # 27 - ASCII код для клавіші Esc
            break

    # Закрити вебкамеру та вікно
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_webcam()
