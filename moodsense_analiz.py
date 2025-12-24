import cv2
from deepface import DeepFace
import csv
import time
from datetime import datetime
import os

# --- AYARLAR ---
dosya_adi = "duygu_verileri.csv"
kayit_sikligi = 2  # Kaç saniyede bir kayıt yapılsın? (Performans için önemli)

# 1. Dosya Hazırlığı: Eğer dosya yoksa başlıkları yazalım
if not os.path.exists(dosya_adi):
    with open(dosya_adi, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Tarih", "Saat", "Duygu"]) # Excel sütun başlıkları

# Yüz tespiti modeli
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Zaman takibi için değişken
son_kayit_zamani = time.time()

print(f"MoodSense Analiz ve Kayıt Modu Başlatıldı...")
print(f"Veriler '{dosya_adi}' dosyasına kaydedilecek.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        try:
            # Duygu Analizi
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(analysis, list):
                result = analysis[0]
            else:
                result = analysis

            dominant_emotion = result['dominant_emotion']
            
            # Türkçeleştirme
            emotion_tr = {
                "happy": "Mutlu", "sad": "Uzgun", "angry": "Kizgin",
                "surprise": "Saskin", "fear": "Korkmus", 
                "disgust": "Igrenmis", "neutral": "Notr"
            }
            text = emotion_tr.get(dominant_emotion, dominant_emotion)
            
            # Ekrana Yaz
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- VERİ KAYIT BÖLÜMÜ ---
            # Şu anki zaman ile son kayıt zamanı arasındaki fark belirlediğimiz süreden büyükse kaydet
            if time.time() - son_kayit_zamani > kayit_sikligi:
                
                simdi = datetime.now()
                tarih = simdi.strftime("%Y-%m-%d")
                saat = simdi.strftime("%H:%M:%S")
                
                # Dosyayı "append" (ekleme) modunda açıyoruz
                with open(dosya_adi, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([tarih, saat, text])
                
                print(f"Kayıt Eklendi: {text}") # Terminalden takip etmen için
                son_kayit_zamani = time.time() # Sayacı sıfırla

        except Exception as e:
            pass
        
    
    cv2.imshow('MoodSense - Duygu Analizi', frame)

    # 1. Yöntem: 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 2. Yöntem: Pencerenin 'X' tuşuna basılırsa çık
    # getWindowProperty: Pencerenin durumunu kontrol eder. 
    # Pencere kapanmışsa (VISIBLE < 1) döngüyü kırar.
    try:
        if cv2.getWindowProperty('MoodSense - Duygu Analizi', cv2.WND_PROP_VISIBLE) < 1:
            break
    except:
        break

cap.release()
cv2.destroyAllWindows()