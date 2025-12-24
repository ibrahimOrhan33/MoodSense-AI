import cv2
import os
import time

def yeni_kayit_olustur():
    # --- AYARLAR ---
    veritabani_yolu = "veritabani"
    max_fotograf = 20  # Kişi başı kaç fotoğraf çekilsin?

    # 1. Kullanıcı ismini alalım
    # GUI'den çağırırken input sorun olabilir ama terminal modu için bu uygun.
    # GUI için main.py içinde özel input alacağız.
    try:
        isim = input("Kaydedilecek kişinin adını girin (Türkçe karakter kullanmadan): ").lower()
    except:
        # Eğer GUI ortamındaysak ve input çalışmazsa diye önlem (GUI'de parametre ile çözeceğiz ama şimdilik basit tutalım)
        return

    if not isim: return

    # 2. Kişiye özel klasör oluşturma
    kisi_yolu = os.path.join(veritabani_yolu, isim)

    if not os.path.exists(kisi_yolu):
        os.makedirs(kisi_yolu)
        print(f"'{isim}' için klasör oluşturuldu: {kisi_yolu}")
    else:
        print(f"Dikkat: '{isim}' kullanıcısı zaten var. Üzerine yeni fotoğraflar eklenecek.")

    # Yüz algılama modeli
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    count = 0
    print("Kamera açılıyor... Lütfen kameraya bakın ve hafifçe başınızı hareket ettirin.")
    time.sleep(2) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        cv2.putText(frame, f"Kaydedilen: {count}/{max_fotograf}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            resim_adi = os.path.join(kisi_yolu, f"{isim}_{count}.jpg")
            cv2.imwrite(resim_adi, frame)
            count += 1
            time.sleep(0.5) # Biraz daha seri çeksin diye süreyi kısalttım (1 -> 0.5)
            
        cv2.imshow('MoodSense - Kayit Modu', frame)
        
        if count >= max_fotograf:
            print(f"Başarılı! {max_fotograf} adet fotoğraf kaydedildi.")
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Derin öğrenme modelinin veritabanını yenilemesi için (pkl dosyasını silmek gerekebilir)
    # DeepFace otomatik hallediyor genelde ama bir dahaki girişte biraz bekletebilir.
    print(f"Kayıt tamamlandı: {isim}")

# Eğer dosyayı doğrudan çalıştırırsan yine eskisi gibi çalışsın:
if __name__ == "__main__":
    yeni_kayit_olustur()