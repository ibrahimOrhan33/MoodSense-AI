import cv2

# 1. Kameraya erişim sağlıyoruz (0 genellikle varsayılan webcam'dir)
cap = cv2.VideoCapture(0)

# Kamera açıldı mı kontrol et
if not cap.isOpened():
    print("Hata: Kamera açılamadı!")
    exit()

print("MoodSense başlatıldı... Çıkmak için 'q' tuşuna basın.")

while True:
    # 2. Kameradan kare kare görüntü okuyoruz
    # ret: Okuma başarılı mı? (True/False)
    # frame: O anki görüntü karesi
    ret, frame = cap.read()

    if not ret:
        print("Görüntü alınamıyor...")
        break

    # (Opsiyonel) Aynalama etkisi için görüntüyü çevirebiliriz
    frame = cv2.flip(frame, 1)

    # 3. Görüntüyü bir pencerede göster
    cv2.imshow('MoodSense - Kamera Testi', frame)

    # 4. Klavyeden giriş bekle (1 milisaniye)
    # Eğer 'q' tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Temizlik işlemleri
cap.release() # Kamerayı serbest bırak
cv2.destroyAllWindows() # Pencereleri kapat
print("MoodSense kapatıldı.")