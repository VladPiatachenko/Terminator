import cv2

def detect_objects():
    # Завантажити класифікатор об'єктів
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Відкрити вебкамеру
    cap = cv2.VideoCapture(0)

    while True:
        # Зчитати кадр з вебкамери
        ret, frame = cap.read()

        # Конвертувати зображення у відтінки сірого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Знайти об'єкти на зображенні
        objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Намалювати прямокутники навколо знайдених об'єктів
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Показати кадр з об'єктами у вікні
        cv2.imshow('Object Detection', frame)

        # Вийти з циклу при натисканні esc
        if cv2.waitKey(1) == 27:
            break

    # Закрити вебкамеру та вікно
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_objects()