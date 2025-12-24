import cv2

# 1. Yüz tanıma modelini yüklüyoruz
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Kamerayı TEKRAR başlatıyoruz (Her dosya kendi kamerasını açar)
cap = cv2.VideoCapture(0)

print("Yüz tespiti başlatıldı... Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Aynalama
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Griye çevirme

    # 3. Yüzleri Tespit Et
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    # 4. Yüzleri çerçevele
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Yuz", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('MoodSense - Yuz Tespiti', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()