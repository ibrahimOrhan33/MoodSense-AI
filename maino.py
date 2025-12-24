import sys
import cv2
import os
import csv
import numpy as np
import random
import time
import glob 
from datetime import datetime
import ollama

# --- PyQt Kütüphaneleri ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLineEdit, QPushButton, 
                             QTextEdit, QInputDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# --- AI ve Yüz Tanıma ---
from deepface import DeepFace
try:
    import yfinance as yf
    from googlesearch import search
except ImportError:
    pass

# ==========================================
# 1. AYARLAR
# ==========================================
VERITABANI_YOLU = "veritabani"
LOG_DOSYASI = "giris_kayitlari.csv"
SOHBET_KLASORU = "sohbet_gecmisi"

# Model Ayarı (Türkçe başarısı için gemma2:2b de deneyebilirsin)
AI_MODEL = "llama3.2" 

if not os.path.exists(VERITABANI_YOLU): os.makedirs(VERITABANI_YOLU)
if not os.path.exists(SOHBET_KLASORU): os.makedirs(SOHBET_KLASORU)

AI_AKTIF = True

# ==========================================
# 2. LOGLAMA
# ==========================================
def giris_logla(kullanici, duygu, durum):
    yoktu = not os.path.exists(LOG_DOSYASI)
    simdi = datetime.now()
    try:
        with open(LOG_DOSYASI, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if yoktu: writer.writerow(["Tarih", "Saat", "Kullanici", "Duygu", "Durum"])
            writer.writerow([simdi.strftime("%Y-%m-%d"), simdi.strftime("%H:%M:%S"), kullanici, duygu, durum])
    except: pass

# ==========================================
# 3. YARDIMCI: CACHE TEMİZLEME
# ==========================================
def deepface_cache_temizle():
    try:
        pkl_files = glob.glob(os.path.join(VERITABANI_YOLU, "*.pkl"))
        for f in pkl_files:
            try: os.remove(f)
            except: pass
    except Exception: pass

# ==========================================
# 4. CHAT THREAD (PROFESYONEL VE TAM CÜMLE)
# ==========================================

class ChatThread(QThread):
    response_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self, user_message, kullanici_adi, is_first_message): # is_first_message eklendi
        super().__init__()
        self.user_message = user_message
        self.kullanici_adi = kullanici_adi 
        self.is_first_message = is_first_message # Değişkeni kaydet
class ChatThread(QThread):

    response_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, user_message, kullanici_adi, is_first_message): 
        super().__init__()
        self.user_message = user_message
        self.kullanici_adi = kullanici_adi 
        self.is_first_message = is_first_message # Durumu sınıfa kaydettik

    def run(self):
        try:
            self.status_signal.emit("MoodSense araştırıyor...")
            # --- 1. VERİ TOPLAMA (Finans & Web) ---
            toplanan_bilgiler = []
            
            # Finans Verisi
            try:
                finans_kelimeleri = ["dolar", "euro", "sterlin", "borsa", "altın", "piyasa", "kur"]
                if any(x in self.user_message.lower() for x in finans_kelimeleri):
                    dolar = yf.Ticker("USDTRY=X").history(period="1d")['Close'].iloc[-1]
                    euro = yf.Ticker("EURTRY=X").history(period="1d")['Close'].iloc[-1]
                    veri = f"PİYASA BİLGİSİ: Dolar: {dolar:.2f} TL, Euro: {euro:.2f} TL"
                    toplanan_bilgiler.append(veri)
            except: pass
            
            # Google Arama Verisi
            try:
                # Bilgi gerektiren soruları algıla
                bilgi_kelimeleri = ["kim", "nedir", "neresi", "nere", "zaman", "kaç", "hangi", "başkent", "hava", "haber", "sonuç", "puan", "listele", "say"]
                if any(x in self.user_message.lower() for x in bilgi_kelimeleri):
                    # Detaylı bilgi için 2 sonuç çekelim
                    search_results = search(self.user_message, num_results=2, advanced=True)
                    for r in search_results:
                        if r.description:
                            temiz_bilgi = r.description.replace("\n", " ")
                            toplanan_bilgiler.append(f"WEB BİLGİSİ: {temiz_bilgi}")
            except: pass

            context_data = "\n".join(toplanan_bilgiler)
            tarih = datetime.now().strftime("%d %B %Y")
            
            # --- 2. SİSTEM PROMPTU (PROFESYONEL AYAR) ---
            # Burada modele "Tam Cümle" kurma ve kurumsal olma talimatı veriyoruz.
            # --- ENTEGRE EDİLMİŞ GÜÇLÜ SİSTEM PROMPTU ---
            # Selamlama talimatını dinamik yapıyoruz
        
            
            system_prompt = (
                f"Senin adın MoodSense. Tarih: {tarih}. Kullanıcı: {self.kullanici_adi}.\n"
                "Görevin: Kullanıcıya profesyonel, bilgili, ciddi ve aynı zamanda yardımsever bir asistan olarak hizmet vermektir.\n"
                "KESİN KURALLAR:\n"
                "1. CEVAP TARZI VE DİL: Cevapların tamamen Türkçe olmalı. 'current', 'location' gibi İngilizce kelimeleri asla kullanma. BAŞLIK YASAĞI: Cevabına asla 'Cevap:', 'MoodSense:', 'EK BİLGİLER:' veya 'WEB BİLGİSİ:' gibi başlıklarla başlama. Doğrudan cümleye gir.\n "
                "Asla robotik ve tek kelimelik cevaplar verme; mutlaka dil bilgisi kurallarına uygun, tam cümleler kur. "
                "(Örn: Yanlış: 'Ankara.', Doğru: 'Türkiye'nin başkenti Ankara'dır.').\n"
                "2. ÜSLUP VE CİDDİYET: 'Hocam', 'Kardeş' gibi ifadeler kullanma; samimi ama profesyonel mesafeyi koruyan bir dost gibi konuş. SELAMLAMA KONTROLÜ: Kullanıcıyla zaten sohbettesin. Her mesajda 'Merhaba', 'İyi günler' veya 'Sana nasıl yardımcı olabilirim' gibi girişler yapma. Sadece soruya odaklan.\n "
                "Eğer bir şeyin maddeleri soruluyorsa, önce kısa bir giriş cümlesi yaz, sonra maddeleri sırala.\n"
                "3. VERİ KULLANIMI: Aşağıdaki kaynak verileri oku ama çıktıya asla 'EK BİLGİLER', 'WEB BİLGİSİ' veya 'KAYNAK' gibi başlıklar yazma. DİL VE AKICILIK: Tamamen Türkçe konuş. 'current', 'location', 'price' gibi İngilizce kelimeleri asla kullanma. Bilgileri doğal bir sohbet akışı içinde sun.\n "
                "Bu bilgileri doğal bir sohbet akışı içinde kendi cümlelerinle harmanlayarak anlat.\n"
                "4. HESAPLAMA: Eğer kullanıcı miktar soruyorsa (Örn: 2 Euro kaç TL?), sana verilen kur bilgisini kullanarak matematiksel hesaplamayı yap ve sonucu söyle.\n"
                "5. DÜRÜSTLÜK: Cevabı bilmiyorsan veya kaynak veriler yetersizse uydurma; nazikçe bilmediğini belirt.\n\n"
                f"--- KAYNAK VERİLER ---\n{context_data}"
            )

            messages = [{'role': 'system', 'content': system_prompt}]
            
            # --- 3. HAFIZA YÜKLEME ---
            if self.kullanici_adi:
                try:
                    dosya_yolu = os.path.join(SOHBET_KLASORU, f"{self.kullanici_adi}_sohbet.txt")
                    if os.path.exists(dosya_yolu):
                        with open(dosya_yolu, "r", encoding="utf-8") as f:
                            # Son 6 mesajı alalım ki bağlam kopmasın ama eski hatalar da gelmesin
                            satirlar = f.readlines()[-6:] 
                            for satir in satirlar:
                                if "Ben:" in satir:
                                    icerik = satir.split("Ben:", 1)[1].strip()
                                    messages.append({'role': 'user', 'content': icerik})
                                elif "MoodSense:" in satir:
                                    icerik = satir.split("MoodSense:", 1)[1].strip()
                                    messages.append({'role': 'assistant', 'content': icerik})
                except: pass

            # --- 4. YENİ MESAJ ---
            messages.append({'role': 'user', 'content': self.user_message})

            # --- 5. OLLAMA ÇAĞRISI ---
            self.status_signal.emit("MoodSense düşünüyor...")

            response = ollama.chat(
                model=AI_MODEL, 
                messages=messages,
                options={'temperature': 0.1} # Ciddiyet ve tutarlılık için düşük sıcaklık
            )
            
            ai_text = response['message']['content']
            self.response_signal.emit(ai_text)

        except Exception as e:
            self.response_signal.emit(f"Hata: {str(e)}")

# ==========================================
# 5. GÜVENLİK THREAD (FACENET512)
# ==========================================
class GuvenlikThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)
    access_signal = pyqtSignal(str, str)
    timeout_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.wait()

    def run(self):
        self._is_running = True
        deepface_cache_temizle() 
        
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        baslangic = time.time()
        süre_siniri = 15 
        
        frame_counter = 0
        basari_saglandi = False
        
        KULLANILAN_MODEL = "Facenet512" 
        GUVENLIK_ESIGI = 0.25 
        
        while self._is_running:
            ret, cv_img = cap.read()
            if not ret: break
            cv_img = cv2.flip(cv_img, 1)
            frame_counter += 1
            
            gecen_sure = time.time() - baslangic
            kalan_sure = int(süre_siniri - gecen_sure)

            if kalan_sure <= 0 and not basari_saglandi:
                self.timeout_signal.emit()
                break 

            if not basari_saglandi:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if frame_counter % 30 == 0: 
                    self.status_signal.emit(f"Hassas Tarama... ({kalan_sure}s)")
                    try:
                        dfs = DeepFace.find(img_path=cv_img, 
                                            db_path=VERITABANI_YOLU, 
                                            model_name=KULLANILAN_MODEL,
                                            distance_metric="cosine",
                                            enforce_detection=False, 
                                            silent=True)
                        
                        if len(dfs) > 0 and not dfs[0].empty:
                            en_yakin_sonuc = dfs[0].iloc[0]
                            path = en_yakin_sonuc['identity']
                            mesafe = en_yakin_sonuc['distance'] 
                            
                            if mesafe < GUVENLIK_ESIGI:
                                isim = os.path.basename(os.path.dirname(path))
                                analiz = DeepFace.analyze(cv_img, actions=['emotion'], enforce_detection=False, silent=True)
                                if isinstance(analiz, list): analiz = analiz[0]
                                raw = analiz['dominant_emotion']
                                tr_map = {"happy": "Mutlu", "sad": "Üzgün", "angry": "Kızgın", "neutral": "Nötr", "fear": "Korkmuş", "surprise": "Şaşkın"}
                                duygu = tr_map.get(raw, raw)

                                self.access_signal.emit(isim, duygu)
                                basari_saglandi = True 
                            else:
                                self.status_signal.emit(f"EŞLEŞME YETERSİZ ({kalan_sure}s)")
                        else:
                            self.status_signal.emit(f"BULUNAMADI ({kalan_sure}s)")
                    except Exception as e: 
                        print(f"Tarama Hatası: {e}")
            
            self.change_pixmap_signal.emit(cv_img)
        cap.release()

# ==========================================
# 6. ANA PENCERE
# ==========================================
class MoodSenseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoodSense AI - Güvenlik Paneli (Ollama Powered)")
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        self.kullanici_adi = None 
        self.anlik_duygu = None
        self.chat_worker = None
        self.thread = None
        
        self.init_ui()
        self.sistem_bosta_modu()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # --- SOL PANEL ---
        self.camera_container = QWidget()
        self.left_layout = QVBoxLayout()
        self.camera_container.setLayout(self.left_layout)
        self.left_layout.addStretch() 
        
        self.lbl_baslik = QLabel("MoodSense Güvenlik")
        self.lbl_baslik.setFont(QFont("Arial", 22, QFont.Bold))
        self.lbl_baslik.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.lbl_baslik, alignment=Qt.AlignCenter)

        self.lbl_camera = QLabel()
        self.lbl_camera.setFixedSize(480, 360)
        self.lbl_camera.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.left_layout.addWidget(self.lbl_camera, alignment=Qt.AlignCenter)

        self.lbl_status = QLabel("Sistem Beklemede...")
        self.lbl_status.setFont(QFont("Arial", 14))
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #00d2ff; margin-top: 10px;")
        self.left_layout.addWidget(self.lbl_status, alignment=Qt.AlignCenter)
        
        self.button_layout = QVBoxLayout()
        self.button_layout.setContentsMargins(50, 10, 50, 10)
        
        self.btn_baslat = QPushButton("SİSTEMİ BAŞLAT / GİRİŞ YAP")
        self.btn_baslat.setFixedHeight(45)
        self.btn_baslat.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 15px; border-radius: 8px;")
        self.btn_baslat.clicked.connect(self.tarama_baslat)
        self.button_layout.addWidget(self.btn_baslat)

        self.btn_kayit = QPushButton("YENİ KAYIT OLUŞTUR")
        self.btn_kayit.setFixedHeight(45)
        self.btn_kayit.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; font-size: 15px; border-radius: 8px;")
        self.btn_kayit.clicked.connect(self.yeni_kayit_baslat)
        self.button_layout.addWidget(self.btn_kayit)

        self.btn_tekrar = QPushButton("TEKRAR DENE")
        self.btn_tekrar.setFixedHeight(45)
        self.btn_tekrar.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; font-size: 15px; border-radius: 8px;")
        self.btn_tekrar.clicked.connect(self.tarama_baslat)
        self.btn_tekrar.hide() 
        self.button_layout.addWidget(self.btn_tekrar)

        self.left_layout.addLayout(self.button_layout)
        self.left_layout.addStretch() 
        self.main_layout.addWidget(self.camera_container)

        # --- SAĞ PANEL (SOHBET) ---
        self.chat_container = QWidget()
        self.right_layout = QVBoxLayout()
        self.chat_container.setLayout(self.right_layout)
        
        self.chat_header = QLabel("AI Asistan")
        self.chat_header.setFont(QFont("Arial", 16, QFont.Bold))
        self.chat_header.setAlignment(Qt.AlignCenter)
        self.chat_header.setStyleSheet("color: #666;")
        self.right_layout.addWidget(self.chat_header)

        # AI Asistan Başlığının Altına Butonu Ekle
        self.btn_logout = QPushButton("Ana Sayfaya Dön")
        self.btn_logout.setFixedHeight(35)
        # Butona kırmızımsı/uyarıcı bir renk veriyoruz
        self.btn_logout.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; border-radius: 5px; margin: 5px;")
        self.right_layout.addWidget(self.btn_logout)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; padding: 15px; font-size: 14px; border: 1px solid #444;")
        self.right_layout.addWidget(self.chat_area)

        self.lbl_typing = QLabel("")
        self.lbl_typing.setFont(QFont("Arial", 9, QFont.StyleItalic))
        self.lbl_typing.setStyleSheet("color: #00d2ff; margin-left: 5px; margin-bottom: 5px;")
        self.right_layout.addWidget(self.lbl_typing)

        input_layout = QHBoxLayout()
        self.msg_input = QLineEdit()
        self.msg_input.setPlaceholderText("Giriş bekleniyor...")
        self.msg_input.setEnabled(False)
        self.msg_input.setStyleSheet("padding: 12px; background-color: #3d3d3d; color: white; border: 1px solid #555; border-radius: 5px;")
        self.msg_input.returnPressed.connect(self.mesaj_gonder_baslat)
        input_layout.addWidget(self.msg_input)
        
        self.btn_send = QPushButton("Gönder")
        self.btn_send.setEnabled(False)
        self.btn_send.clicked.connect(self.mesaj_gonder_baslat)
        self.btn_send.setStyleSheet("padding: 12px; background-color: #00d2ff; color: black; font-weight: bold; border-radius: 5px;")
        input_layout.addWidget(self.btn_send)
        
        self.btn_logout.clicked.connect(self.oturum_kapat)
        self.right_layout.addLayout(input_layout)
        self.main_layout.addWidget(self.chat_container)
        self.chat_container.hide() 

    def sistem_bosta_modu(self):
        if self.thread: self.thread.stop()
        self.chat_container.hide()
        self.camera_container.show()
        self.lbl_camera.clear()
        self.lbl_camera.setText("MOODSENSE GÜVENLİK SİSTEMİ\n\nBaşlatmak için aşağıdaki butona tıklayın.")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: #111; color: #aaa; font-size: 16px; border: 2px solid #555;")
        self.lbl_status.setText("Sistem Hazır")
        self.lbl_status.setStyleSheet("color: #888;")
        self.btn_baslat.show()
        self.btn_kayit.show()
        self.btn_tekrar.hide()

    def tarama_baslat(self):
        if self.thread: self.thread.stop()
        self.lbl_camera.clear()
        self.lbl_camera.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.btn_baslat.hide()
        self.btn_kayit.hide()
        self.btn_tekrar.hide()
        self.thread = GuvenlikThread()
        self.thread.change_pixmap_signal.connect(self.guncelle_kamera)
        self.thread.status_signal.connect(self.guncelle_durum)
        self.thread.access_signal.connect(self.giris_basarili)
        self.thread.timeout_signal.connect(self.tanima_basarisiz)
        self.thread.start()

    def guncelle_kamera(self, cv_img):
        if not self.kullanici_adi:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.lbl_camera.setPixmap(QPixmap.fromImage(qt_img.scaled(480, 360, Qt.KeepAspectRatio)))

    def guncelle_durum(self, text):
        if not self.kullanici_adi:
            self.lbl_status.setText(text)
            color = "red" if "YETERSİZ" in text or "BULUNAMADI" in text else "#00d2ff"
            self.lbl_status.setStyleSheet(f"color: {color};")

    def tanima_basarisiz(self):
        self.thread.stop()
        self.lbl_status.setText("SÜRE DOLDU - TANINAMADI")
        self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
        self.btn_baslat.show()
        self.btn_kayit.show()
        self.btn_tekrar.show() 
        QMessageBox.warning(self, "Erişim Yok", "Sizi tanıyamadım.")

    def yeni_kayit_baslat(self):
        if self.thread: self.thread.stop()
        isim, ok = QInputDialog.getText(self, 'Yeni Kayıt', 'Adınız (Türkçe karakter kullanmayın):')
        if ok and isim:
            self.lbl_status.setText("Kayıt Başlıyor...")
            kisi_yolu = os.path.join(VERITABANI_YOLU, isim.lower())
            if not os.path.exists(kisi_yolu): os.makedirs(kisi_yolu)
            deepface_cache_temizle()
            cap_kayit = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            count = 0
            son_cekilen_zaman = 0 
            while True:
                ret, frame = cap_kayit.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                clean_frame = frame.copy() 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                su_an = time.time()
                if len(faces) > 0 and (su_an - son_cekilen_zaman > 1.0):
                    path = os.path.join(kisi_yolu, f"{isim}_{count}.jpg")
                    cv2.imwrite(path, clean_frame)
                    count += 1
                    son_cekilen_zaman = su_an
                    self.lbl_status.setText(f"Fotoğraf: {count}/15")
                elif len(faces) == 0:
                     self.lbl_status.setText("Yüz Aranıyor...")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.lbl_camera.setPixmap(QPixmap.fromImage(qt_img.scaled(480, 360, Qt.KeepAspectRatio)))
                QApplication.processEvents()
                if count >= 15: break
            cap_kayit.release()
            deepface_cache_temizle()
            QMessageBox.information(self, "Kayıt Tamamlandı", f"Kaydınız başarıyla oluşturuldu {isim.capitalize()}. Giriş yapılıyor...")
            self.giris_basarili(isim, "mutlu")

    def giris_basarili(self, isim, duygu):
        if self.kullanici_adi: return 
        if self.thread: self.thread.stop()
        self.kullanici_adi = isim
        self.anlik_duygu = duygu
        self.camera_container.hide() 
        self.chat_container.show()
        self.chat_header.setText(f"MoodSense - {isim.upper()}")
        self.chat_header.setStyleSheet("color: #00ff00;")
        self.msg_input.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.msg_input.setPlaceholderText("Yerel AI hizmetinizde...")
        self.msg_input.setFocus()
        giris_logla(isim, duygu, "Giris Basarili")
        self.gecmisi_yukle()
        
        if AI_AKTIF:
            duygu_lower = duygu.lower()
            if "mutlu" in duygu_lower:
                mesaj = f"Ooo {isim.capitalize()} hoş geldin! Yüzünde güller açıyor, enerjin harika!"
            elif "üzgün" in duygu_lower:
                mesaj = f"Merhaba {isim.capitalize()}. Seni biraz durgun gördüm, canını sıkan bir şey mi var?"
            elif "kızgın" in duygu_lower:
                mesaj = f"Hoş geldin {isim.capitalize()}. Biraz gergin gibisin, sakin ol şampiyon."
            elif "korkmuş" in duygu_lower:
                mesaj = f"{isim.capitalize()}, endişeli görünüyorsun. Her şey yolunda mı?"
            elif "şaşkın" in duygu_lower:
                mesaj = f"Selam {isim.capitalize()}, seni bir şey şaşırtmış gibi! Ne oldu?"
            else:
                mesaj = f"Selam {isim.capitalize()}. Bugün gayet ciddi ve odaklanmış görünüyorsun. Senin için ne yapabilirim?"
            self.chat_ekle("MoodSense", mesaj)

    def gecmisi_yukle(self):
        try:
            dosya = f"{SOHBET_KLASORU}/{self.kullanici_adi}_sohbet.txt"
            if os.path.exists(dosya):
                with open(dosya, "r", encoding="utf-8") as f:
                    satirlar = f.readlines()[-5:]
                    self.chat_area.append(f"<div style='color:#666; text-align:center; margin:10px;'>--- Geçmiş ---</div>")
                    for s in satirlar:
                        self.chat_area.append(f"<span style='color:#aaa; font-size:12px;'>{s.strip()}</span>")
        except: pass

    def mesaj_gonder_baslat(self):
        text = self.msg_input.text().strip()
        if not text: return
        if text.lower() in ["çıkış", "cikis", "exit", "kapat", "q", "oturumu kapat"]:
            self.chat_ekle("Sistem", f"Güle güle {self.kullanici_adi.capitalize()}! Oturum kapatılıyor...")
            self.msg_input.clear()
            self.msg_input.setEnabled(False)
            QApplication.processEvents()
            time.sleep(2.0)
            self.oturum_kapat() 
            return
        self.chat_ekle("Ben", text)
        self.msg_input.clear()
        self.msg_input.setEnabled(False) 
        self.chat_worker = ChatThread(text, self.kullanici_adi)
        self.chat_worker.response_signal.connect(self.ai_cevap_geldi)
        self.chat_worker.status_signal.connect(self.lbl_typing.setText)
        self.chat_worker.start()

    
    def oturum_kapat(self):
        """Oturumu kapatır, log kaydı tutar ve arayüzü ana sayfaya döndürür."""
        if self.kullanici_adi:
            # 1. Mevcut loglama mantığını korur
            giris_logla(self.kullanici_adi, self.anlik_duygu, "Cikis Yapildi")
            deepface_cache_temizle()
        
        # 2. Değişkenleri sıfırla
        self.kullanici_adi = None
        self.anlik_duygu = None
        self.chat_area.clear()
        
        # 3. Başlığı ve stilleri varsayılana döndür
        self.chat_header.setText("AI Asistan")
        self.chat_header.setStyleSheet("color: #666;")
        
        # 4. Arayüzü eski haline getir
        self.chat_container.hide()
        self.camera_container.show()
        self.sistem_bosta_modu()

#    def oturum_kapat(self):
#        if self.kullanici_adi:
#            giris_logla(self.kullanici_adi, self.anlik_duygu, "Cikis Yapildi")
#            deepface_cache_temizle()
#        self.kullanici_adi = None
#        self.anlik_duygu = None
#        self.chat_area.clear()
#        self.chat_header.setText("AI Asistan")
#        self.chat_header.setStyleSheet("color: #666;")
#        self.chat_container.hide()
#        self.camera_container.show()
#        self.sistem_bosta_modu()

    def ai_cevap_geldi(self, cevap):
        self.lbl_typing.setText("")
        self.chat_ekle("MoodSense", cevap)
        self.msg_input.setEnabled(True)
        self.msg_input.setFocus()

    def chat_ekle(self, sender, msg):
        # 1. Renk, hizalama ve arka plan belirle
        if sender == "Ben":
            color = "#00ff00"
            align = "right"
            bg_color = "#1b331b" # Kullanıcı mesajı yeşilimsi
        elif sender == "MoodSense":
            color = "#00d2ff"
            align = "left"
            bg_color = "#1b2733" # AI mesajı mavimsi
        else:
            color = "white"
            align = "center"
            bg_color = "transparent"

        msg_html = msg.replace("\n", "<br>")
        
        # 2. %100 genişlikte bir tablo ile kesin sağ/sol hizalaması yap
        full_html = f"""
        <table width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-bottom: 10px;">
            <tr>
                <td align="{align}">
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 12px; display: inline-block; max-width: 80%;">
                        <b style="color: {color}; font-size: 11px;">{sender}</b><br>
                        <span style="color: white; font-size: 14px;">{msg_html}</span>
                    </div>
                </td>
            </tr>
        </table>
        """
        self.chat_area.append(full_html)
        
        # 3. Otomatik olarak en aşağı kaydır
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

        # 4. KULLANICIYA ÖZEL GEÇMİŞ DOSYASINA KAYDET (Kritik kısım)
        if self.kullanici_adi and sender != "Sistem":
            try:
                clean_msg = msg.replace("\n", " ")
                dosya = f"{SOHBET_KLASORU}/{self.kullanici_adi}_sohbet.txt"
                with open(dosya, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().strftime('%d-%m %H:%M')}] {sender}: {clean_msg}\n")
            except: 
                pass    

#    def chat_ekle(self, sender, msg):
 #       color = "#00d2ff" if sender == "MoodSense" else "#00ff00"
 #       align = "left" if sender == "MoodSense" else "right"
 #       msg = msg.replace("\n", "<br>")
 #       html_msg = f"<div style='margin-bottom:10px; text-align:{align};'><span style='color:{color}; font-weight:bold;'>{sender}</span><br><span style='font-size:15px;'>{msg}</span></div>"
 #       self.chat_area.append(html_msg)
 #       if self.kullanici_adi:
 #           try:
  #              clean_msg = msg.replace("<br>", " ")
   #             dosya = f"{SOHBET_KLASORU}/{self.kullanici_adi}_sohbet.txt"
    #            with open(dosya, "a", encoding="utf-8") as f:
     #               f.write(f"[{datetime.now().strftime('%d-%m %H:%M')}] {sender}: {clean_msg}\n")
      #      except: pass

    def closeEvent(self, event):
        if self.thread: self.thread.stop()
        if self.kullanici_adi: giris_logla(self.kullanici_adi, self.anlik_duygu, "Cikis Yapildi")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MoodSenseWindow()
    window.show()
    sys.exit(app.exec_())