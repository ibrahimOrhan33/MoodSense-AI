import sys
import cv2
import os
import csv
import numpy as np
import random
import time
import glob 
from datetime import datetime
import google.generativeai as genai

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
API_KEY = "AIzaSyAQticX1_cSBAa_LTfJkIBKGxLb_nyVr8U" 
VERITABANI_YOLU = "veritabani"
LOG_DOSYASI = "giris_kayitlari.csv"
SOHBET_KLASORU = "sohbet_gecmisi"

if not os.path.exists(VERITABANI_YOLU): os.makedirs(VERITABANI_YOLU)
if not os.path.exists(SOHBET_KLASORU): os.makedirs(SOHBET_KLASORU)

AI_AKTIF = False
try:
    if "BURAYA_KENDI" not in API_KEY:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash') 
        chat_session = model.start_chat(history=[])
        AI_AKTIF = True
except Exception: pass

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
# 4. CHAT THREAD
# ==========================================
class ChatThread(QThread):
    response_signal = pyqtSignal(str)
    def __init__(self, user_message, chat_session):
        super().__init__()
        self.user_message = user_message
        self.chat_session = chat_session
    def run(self):
        try:
            toplanan_bilgiler = []
            try:
                dolar = yf.Ticker("USDTRY=X").history(period="1d")['Close'].iloc[-1]
                euro = yf.Ticker("EURTRY=X").history(period="1d")['Close'].iloc[-1]
                toplanan_bilgiler.append(f"[FİNANS]: 1 Dolar = {dolar:.2f} TL, 1 Euro = {euro:.2f} TL")
            except: pass
            
            try:
                search_results = search(self.user_message, num_results=2, advanced=True)
                for r in search_results:
                    if r.description: toplanan_bilgiler.append(f"[WEB]: {r.description}")
            except: pass

            context_data = "\n".join(toplanan_bilgiler)
            tarih = datetime.now().strftime("%d %B %Y")
            prompt = (
                f"Tarih: {tarih}\n"
                f"Soru: {self.user_message}\n"
                f"Bulunan Veriler:\n{context_data}\n"
                "Sen MoodSense asistanısın. Bu verileri kullanarak veya genel kültürünle samimi cevap ver."
            )
            response = self.chat_session.send_message(prompt)
            self.response_signal.emit(response.text)
        except Exception as e:
            self.response_signal.emit(f"Hata: {str(e)}")

# ==========================================
# 5. GÜVENLİK THREAD (FACENET512 - DENGELİ AYAR)
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
        
        # --- GÜVENLİK AYARLARI (GÜNCELLENDİ) ---
        KULLANILAN_MODEL = "Facenet512" 
        
        # 0.25: Hem seni tanır hem yabancıyı reddeder (İdeal Denge)
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
                            
                            # Terminale yazdıralım (Kontrol için)
                            print(f"Aday: {path} - Fark: {mesafe}") 
                            
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
        self.setWindowTitle("MoodSense AI - Güvenlik Paneli")
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

        # --- SOL PANEL (KAMERA & BAŞLANGIÇ) ---
        self.camera_container = QWidget()
        self.left_layout = QVBoxLayout()
        self.camera_container.setLayout(self.left_layout)
        
        # HİZALAMA: Üste boşluk
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
        
        # BUTONLAR (DİKEY)
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
        
        # HİZALAMA: Alta boşluk
        self.left_layout.addStretch() 
        
        self.main_layout.addWidget(self.camera_container)

        # --- SAĞ PANEL (SOHBET - BAŞLANGIÇTA GİZLİ) ---
        self.chat_container = QWidget()
        self.right_layout = QVBoxLayout()
        self.chat_container.setLayout(self.right_layout)
        
        self.chat_header = QLabel("AI Asistan")
        self.chat_header.setFont(QFont("Arial", 16, QFont.Bold))
        self.chat_header.setAlignment(Qt.AlignCenter)
        self.chat_header.setStyleSheet("color: #666;")
        self.right_layout.addWidget(self.chat_header)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; padding: 15px; font-size: 14px; border: 1px solid #444;")
        self.right_layout.addWidget(self.chat_area)

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
        self.msg_input.setPlaceholderText("İnternetten her şeyi sorabilirsin...")
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
        if AI_AKTIF:
            self.chat_worker = ChatThread(text, chat_session)
            self.chat_worker.response_signal.connect(self.ai_cevap_geldi)
            self.chat_worker.start()
        else:
            self.chat_ekle("Sistem", "AI Bağlantısı Yok.")
            self.msg_input.setEnabled(True)

    def oturum_kapat(self):
        if self.kullanici_adi:
            giris_logla(self.kullanici_adi, self.anlik_duygu, "Cikis Yapildi")
            deepface_cache_temizle()
        
        self.kullanici_adi = None
        self.anlik_duygu = None
        self.chat_area.clear()
        self.chat_header.setText("AI Asistan")
        self.chat_header.setStyleSheet("color: #666;")
        
        self.chat_container.hide()
        self.camera_container.show()
        
        self.sistem_bosta_modu()

    def ai_cevap_geldi(self, cevap):
        self.chat_ekle("MoodSense", cevap)
        self.msg_input.setEnabled(True)
        self.msg_input.setFocus()

    def chat_ekle(self, sender, msg):
        color = "#00d2ff" if sender == "MoodSense" else "#00ff00"
        align = "left" if sender == "MoodSense" else "right"
        html_msg = f"<div style='margin-bottom:10px; text-align:{align};'><span style='color:{color}; font-weight:bold;'>{sender}</span><br><span style='font-size:15px;'>{msg}</span></div>"
        self.chat_area.append(html_msg)
        if self.kullanici_adi:
            try:
                dosya = f"{SOHBET_KLASORU}/{self.kullanici_adi}_sohbet.txt"
                with open(dosya, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().strftime('%d-%m %H:%M')}] {sender}: {msg}\n")
            except: pass

    def closeEvent(self, event):
        if self.thread: self.thread.stop()
        if self.kullanici_adi: giris_logla(self.kullanici_adi, self.anlik_duygu, "Cikis Yapildi")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MoodSenseWindow()
    window.show()
    sys.exit(app.exec_())