import os
import csv
from datetime import datetime
import time
import google.generativeai as genai

# ==========================================
# 🔑 API ANAHTARINI BURAYA YAPIŞTIR (Kendi anahtarınla değiştir)
# ==========================================
API_KEY = "AIzaSyCZlAUVSLTHtnRiRHDu9AEU-YxSsYvBH5A" 

# Model Yapılandırması
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"API Hatası: {e}")
    model = None

def sohbet_baslat(kullanici_adi, anlik_duygu="belirsiz"):
    print("\n" * 50)
    print(f"==========================================")
    print(f"🧠 MOODSENSE AI - Hoş Geldin {kullanici_adi.capitalize()}")
    print(f"📊 Algılanan Duygu Durumu: {anlik_duygu.upper()}")
    print(f"==========================================")
    
    # Dosya işlemleri (Aynı)
    klasor = "sohbet_gecmisi"
    if not os.path.exists(klasor):
        os.makedirs(klasor)
    dosya_yolu = os.path.join(klasor, f"{kullanici_adi}.txt")
    
    # Geçmişi Oku
    if os.path.exists(dosya_yolu):
        with open(dosya_yolu, "r", encoding="utf-8") as f:
            lines = f.readlines()
            context_text = "".join(lines[-20:])
    else:
        context_text = ""

    # Chat Başlat
    chat_session = model.start_chat(history=[])
    
    # --- GÜNCELLENMİŞ VE GÜÇLENDİRİLMİŞ TALİMAT (PROMPT) ---
    prompt = (
        f"Senin adın 'MoodSense'. Sen sıradan bir bot değil, kullanıcının yakın bir dostu gibi davranan, "
        f"esprili, zeki ve empati yeteneği yüksek bir asistansın. "
        f"Şu an karşındaki kullanıcının adı: {kullanici_adi}. "
        f"Sisteme kameradan giriş yaptı ve yüz ifadesi şu an: '{anlik_duygu}'.\n\n"
        
        f"GÖREVLERİN:\n"
        f"1. Söze girerken ASLA 'Merhaba ben bir yapay zekayım' deme. Doğrudan bir insan gibi konuş.\n"
        f"2. Açılış cümlende MUTLAKA kullanıcının bu '{anlik_duygu}' haline değin.\n"
        f"   - Eğer 'mutlu' ise: Enerjik ol, 'Ooo yüzünde güller açıyor!' gibi gir.\n"
        f"   - Eğer 'üzgün' ise: Şefkatli ol, 'Canın bir şeye mi sıkkın?' diye sor.\n"
        f"   - Eğer 'nötr' veya 'ciddi' ise: 'Çok odaklanmış/ciddi görünüyorsun, dalgın gibisin, bir sorun yok değil mi?' gibi bir giriş yap.\n"
        f"   - Eğer 'kızgın' ise: 'Sakin ol şampiyon, kim sinirlendirdi seni?' gibi yaklaş.\n"
        f"3. Kullanıcı sana 'Nasıl görünüyorum?' derse, kamera verisine dayanarak yorum yap.\n\n"
        
        f"Sohbet Geçmişi (Bağlam):\n{context_text}"
    )
    
    print("AI duyguna göre hazırlanıyor... (Lütfen bekleyin)\n")

    try:
        ilk_cevap = chat_session.send_message(prompt)
        print(f"🤖 AI: {ilk_cevap.text}\n")
        
        # Loglama
        zaman_damgasi = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(dosya_yolu, "a", encoding="utf-8") as f:
            f.write(f"[{zaman_damgasi}] GİRİŞ (Duygu: {anlik_duygu})\n")
            f.write(f"[{zaman_damgasi}] AI: {ilk_cevap.text}\n")

    except Exception as e:
        print(f"AI Başlatma Hatası: {e}")

    # Sohbet Döngüsü (Aynı)
    while True:
        try:
            kullanici_mesaji = input(f"Sen: ")
            
            if kullanici_mesaji.lower() in ["cikis", "exit", "q"]:
                cikis_yap(kullanici_adi)
                break
            
            response = chat_session.send_message(kullanici_mesaji)
            print(f"\n🤖 AI: {response.text}\n")
            
            zaman_damgasi = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(dosya_yolu, "a", encoding="utf-8") as f:
                f.write(f"[{zaman_damgasi}] {kullanici_adi}: {kullanici_mesaji}\n")
                f.write(f"[{zaman_damgasi}] AI: {response.text}\n")
                
        except KeyboardInterrupt:
            cikis_yap(kullanici_adi)
            break
        except Exception as e:
            print(f"Hata: {e}")
            break
def cikis_yap(kullanici_adi):
    print(f"\nGüle güle {kullanici_adi}! Oturum kapatılıyor...")
    log_file = "giris_kayitlari.csv"
    simdi = datetime.now()
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([simdi.strftime("%Y-%m-%d"), simdi.strftime("%H:%M:%S"), kullanici_adi, "Cikis Yapildi"])
    time.sleep(2)