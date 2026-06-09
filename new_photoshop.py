import os
# Mencegah log error dari TensorFlow dan konflik OpenMP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import time

# Import Model AI
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# ================= 1. KONFIGURASI HALAMAN & TEMA UTAMA =================
st.set_page_config(page_title="Mini Photoshop Pro | Ultimate", page_icon="💎", layout="wide", initial_sidebar_state="expanded")

# CSS Injection Level Enterprise (Glassmorphism, Animasi, Custom Scrollbar)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }

    /* Header Glassmorphism Premium */
    .hero-container {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(0,168,255,0.1) 0%, transparent 60%);
        animation: rotate 20s linear infinite;
        z-index: 0;
    }
    @keyframes rotate { 100% { transform: rotate(360deg); } }
    
    .hero-content { position: relative; z-index: 1; }
    .hero-title { background: -webkit-linear-gradient(45deg, #00a8ff, #00fcce); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 3rem; margin: 0; letter-spacing: -1px; }
    .hero-subtitle { color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; font-weight: 400; }

    /* Styling Tombol Modern */
    div.stButton > button {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #00a8ff;
        border-color: #00a8ff;
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 168, 255, 0.3);
    }
    
    /* Label Metrics yang lebih elegan */
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; color: #00fcce; }
</style>
""", unsafe_allow_html=True)

# ================= 2. CACHING AI MODELS (Performa Super Cepat) =================
@st.cache_resource(show_spinner=False)
def load_models():
    yolo = YOLO('yolov8n.pt')
    cnn = MobileNetV2(weights='imagenet')
    return yolo, cnn

with st.spinner("Menginisialisasi Engine Deep Learning..."):
    yolo_model, cnn_model = load_models()

# ================= 3. SESSION STATE MANAGEMENT =================
if 'img_original' not in st.session_state: st.session_state.img_original = None
if 'img_current' not in st.session_state: st.session_state.img_current = None
if 'cnn_result' not in st.session_state: st.session_state.cnn_result = None

def reset_image():
    if st.session_state.img_original is not None:
        st.session_state.img_current = st.session_state.img_original.copy()
        st.session_state.cnn_result = None
        st.toast('🔄 Gambar dikembalikan ke kondisi asali.', icon='♻️')

# ================= KAMUS PENERJEMAH AI KE BAHASA INDONESIA =================
# Kamus YOLOv8 (80 Kelas COCO Dataset)
KAMUS_YOLO = {
    0: 'Orang', 1: 'Sepeda', 2: 'Mobil', 3: 'Motor', 4: 'Pesawat', 5: 'Bus', 6: 'Kereta', 7: 'Truk', 8: 'Kapal', 
    9: 'Lampu Lalu Lintas', 10: 'Hidran Api', 11: 'Rambu Stop', 12: 'Meteran Parkir', 13: 'Bangku Taman', 14: 'Burung', 
    15: 'Kucing', 16: 'Anjing', 17: 'Kuda', 18: 'Domba', 19: 'Sapi', 20: 'Gajah', 21: 'Beruang', 22: 'Zebra', 
    23: 'Jerapah', 24: 'Ransel', 25: 'Payung', 26: 'Tas Tangan', 27: 'Dasi', 28: 'Koper', 29: 'Frisbee', 30: 'Ski', 
    31: 'Papan Salju', 32: 'Bola Olahraga', 33: 'Layang-layang', 34: 'Tongkat Bisbol', 35: 'Sarung Tangan Bisbol', 
    36: 'Skateboard', 37: 'Papan Selancar', 38: 'Raket Tenis', 39: 'Botol', 40: 'Gelas Anggur', 41: 'Cangkir', 
    42: 'Garpu', 43: 'Pisau', 44: 'Sendok', 45: 'Mangkuk', 46: 'Pisang', 47: 'Apel', 48: 'Sandwich', 49: 'Jeruk', 
    50: 'Brokoli', 51: 'Wortel', 52: 'Hot Dog', 53: 'Pizza', 54: 'Donat', 55: 'Kue', 56: 'Kursi', 57: 'Sofa', 
    58: 'Tanaman Pot', 59: 'Tempat Tidur', 60: 'Meja Makan', 61: 'Toilet', 62: 'Monitor / TV', 63: 'Laptop', 
    64: 'Mouse Komputer', 65: 'Remote', 66: 'Keyboard', 67: 'Ponsel', 68: 'Microwave', 69: 'Oven', 70: 'Pemanggang Roti', 
    71: 'Wastafel', 72: 'Kulkas', 73: 'Buku', 74: 'Jam', 75: 'Vas', 76: 'Gunting', 77: 'Boneka Beruang', 
    78: 'Pengering Rambut', 79: 'Sikat Gigi'
}

# Fungsi Penerjemah Keras (ImageNet Dataset 1000 Kelas)
def terjemahkan_imagenet(label_inggris):
    kamus_keras = {
        'suit': 'Setelan Jas Formal',
        'windsor_tie': 'Dasi Windsor',
        'loafer': 'Sepatu Pantofel',
        'bow_tie': 'Dasi Kupu-kupu',
        'military_uniform': 'Seragam Militer',
        'laptop': 'Laptop',
        'cellular_telephone': 'Ponsel Pintar / HP',
        'desktop_computer': 'Komputer Desktop',
        'mouse': 'Mouse Komputer',
        'keyboard': 'Keyboard',
        'coffee_mug': 'Cangkir Kopi',
        'sports_car': 'Mobil Sport',
        'passenger_car': 'Mobil Penumpang',
        'minivan': 'Mobil Minivan',
        'tabby': 'Kucing Tabby',
        'golden_retriever': 'Anjing Golden Retriever',
        'sunglasses': 'Kacamata Hitam',
        'backpack': 'Tas Ransel'
    }
    # Jika ada di kamus, gunakan terjemahan. Jika tidak, bersihkan teks Inggrisnya.
    label_bersih = label_inggris.lower().strip()
    return kamus_keras.get(label_bersih, label_bersih.replace('_', ' ').title())

# ================= 4. HEADER HERO COMPONENT =================
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <h1 class="hero-title">💎 Mini Photoshop Pro Workspace</h1>
        <div class="hero-subtitle">High-Fidelity Digital Image Processing & Dual-Core AI Architecture</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= 5. SIDEBAR: PROFESSIONAL TOOLS PANEL =================
st.sidebar.markdown("<h2 style='color:#00a8ff; font-weight:800; text-align:center;'>🛠️ WORKSPACE TOOLS</h2><hr style='border-color:#334155;'>", unsafe_allow_html=True)

with st.sidebar.expander("📁 1. Manajemen & Ekspor Resolusi", expanded=True):
    uploaded_file = st.file_uploader("Drop file citra di sini", type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif'], label_visibility="collapsed")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if st.session_state.img_original is None or st.session_state.get('last_file') != uploaded_file.name:
            st.session_state.img_original = img_rgb.copy()
            st.session_state.img_current = img_rgb.copy()
            st.session_state.last_file = uploaded_file.name
            st.session_state.cnn_result = None
            st.toast("✅ File berhasil dimuat ke dalam Workspace!", icon="🚀")

    if st.session_state.img_current is not None:
        col_res1, col_res2 = st.columns(2)
        col_res1.button("🔄 Reset", on_click=reset_image, use_container_width=True)
        
        quality_val = st.slider("Kualitas Kompresi JPEG (%)", 10, 100, 85)
        img_bgr_save = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr_save, [int(cv2.IMWRITE_JPEG_QUALITY), quality_val])
        st.download_button("💾 Ekspor Citra", io.BytesIO(buffer), "ps_output_hq.jpg", "image/jpeg", use_container_width=True)

# Proteksi UI jika belum ada gambar
if st.session_state.img_current is None:
    st.info("💡 Sistem standby. Silakan unggah citra pada panel *Workspace Tools* di sebelah kiri untuk memulai komputasi matriks.")
    st.stop()

with st.sidebar.expander("✨ 2. Enhancement & Koreksi Warna"):
    cb_bright = st.slider("Kecerahan (Brightness)", -100, 100, 0)
    cb_contrast = st.slider("Kontras (Contrast)", 1.0, 3.0, 1.0, 0.1)
    if st.button("Terapkan Koreksi", use_container_width=True):
        st.session_state.img_current = cv2.convertScaleAbs(st.session_state.img_current, alpha=cb_contrast, beta=cb_bright)
        st.session_state.cnn_result = None
        st.toast("Koreksi cahaya & kontras diterapkan", icon="✨")
        st.rerun()
        
    col_enh1, col_enh2 = st.columns(2)
    if col_enh1.button("Hist. Eq", use_container_width=True):
        yuv = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        st.session_state.img_current = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        st.session_state.cnn_result = None; st.rerun()
        
    if col_enh2.button("Sharpen", use_container_width=True):
        st.session_state.img_current = cv2.filter2D(st.session_state.img_current, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        st.session_state.cnn_result = None; st.rerun()

with st.sidebar.expander("📐 3. Transformasi Spasial"):
    geom_action = st.selectbox("Operasi Matriks:", ["Rotasi Spasial", "Mirroring", "Penskalaan (Resize 50%)"])
    if geom_action == "Rotasi Spasial":
        deg = st.slider("Derajat", 0, 360, 0)
        if st.button("Putar Matriks", use_container_width=True):
            h, w = st.session_state.img_current.shape[:2]
            st.session_state.img_current = cv2.warpAffine(st.session_state.img_current, cv2.getRotationMatrix2D((w//2, h//2), deg, 1.0), (w, h))
            st.session_state.cnn_result = None; st.rerun()
    elif geom_action == "Mirroring":
        c_m1, c_m2 = st.columns(2)
        if c_m1.button("Flip H"): st.session_state.img_current = cv2.flip(st.session_state.img_current, 1); st.session_state.cnn_result = None; st.rerun()
        if c_m2.button("Flip V"): st.session_state.img_current = cv2.flip(st.session_state.img_current, 0); st.session_state.cnn_result = None; st.rerun()
    elif geom_action == "Penskalaan (Resize 50%)" and st.button("Eksekusi Skala"):
        h, w = st.session_state.img_current.shape[:2]
        st.session_state.img_current = cv2.resize(st.session_state.img_current, (w//2, h//2))
        st.session_state.cnn_result = None; st.rerun()

with st.sidebar.expander("🩹 4. Filter Reduksi Noise"):
    c_n1, c_n2 = st.columns(2)
    if c_n1.button("Gaussian", use_container_width=True):
        st.session_state.img_current = cv2.GaussianBlur(st.session_state.img_current, (11, 11), 0); st.session_state.cnn_result = None; st.rerun()
    if c_n2.button("Median", use_container_width=True):
        st.session_state.img_current = cv2.medianBlur(st.session_state.img_current, 5); st.session_state.cnn_result = None; st.rerun()

with st.sidebar.expander("💠 5 & 6. Biner, Tepi & Splitting"):
    c_b1, c_b2 = st.columns(2)
    if c_b1.button("Biner", use_container_width=True):
        g = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
        _, b = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
        st.session_state.img_current = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB); st.session_state.cnn_result = None; st.rerun()
    if c_b2.button("Split Merah", use_container_width=True) and len(st.session_state.img_current.shape) == 3:
        r, g, b = cv2.split(st.session_state.img_current)
        st.session_state.img_current = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)]); st.session_state.cnn_result = None; st.rerun()
        
    metode_tepi = st.selectbox("Algoritma Tepi:", ["Canny", "Sobel", "Laplacian"])
    if st.button("Deteksi Tepi", use_container_width=True):
        g = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
        if metode_tepi == "Canny": edges = cv2.Canny(g, 100, 200)
        elif metode_tepi == "Sobel": edges = cv2.convertScaleAbs(cv2.magnitude(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)))
        elif metode_tepi == "Laplacian": edges = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F))
        st.session_state.img_current = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB); st.session_state.cnn_result = None; st.rerun()

# ================= 6. MAIN AREA: TABS DASHBOARD =================
tab_visual, tab_hist, tab_ai = st.tabs(["👁️ Visual Workspace", "📊 Analisis Histogram", "🧠 Dual-Core AI Engine"])

# --- TAB 1: VISUAL WORKSPACE ---
with tab_visual:
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("<h4 style='color:#94a3b8; text-align:center;'>Source Image</h4>", unsafe_allow_html=True)
        st.image(st.session_state.img_original, use_container_width=True)
        st.caption(f"📏 Resolusi Orisinal: {st.session_state.img_original.shape[1]} x {st.session_state.img_original.shape[0]} px")
        
    with col_img2:
        st.markdown("<h4 style='color:#00fcce; text-align:center;'>Active Layer (Rendered)</h4>", unsafe_allow_html=True)
        st.image(st.session_state.img_current, use_container_width=True)
        st.caption(f"📏 Resolusi Matriks Terkini: {st.session_state.img_current.shape[1]} x {st.session_state.img_current.shape[0]} px")

# --- TAB 2: HISTOGRAM ANALYSIS ---
with tab_hist:
    st.markdown("<h3 style='color:#00a8ff;'>Analisis Distribusi Frekuensi Piksel (Grayscale)</h3>", unsafe_allow_html=True)
    if st.button("📉 Render Plot Matplotlib Real-Time"):
        with st.spinner("Mengomputasi array piksel..."):
            g_ori = cv2.cvtColor(st.session_state.img_original, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_original.shape)==3 else st.session_state.img_original
            g_cur = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape)==3 else st.session_state.img_current
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            fig.patch.set_facecolor('#0e1117')
            
            ax[0].set_facecolor('#1e293b'); ax[0].grid(color='#334155', linestyle='--', alpha=0.5)
            ax[0].hist(g_ori.ravel(), bins=256, range=[0,256], color='#94a3b8', alpha=0.8)
            ax[0].set_title('Distribusi Orisinal', color='#f8fafc', pad=15)
            
            ax[1].set_facecolor('#1e293b'); ax[1].grid(color='#334155', linestyle='--', alpha=0.5)
            ax[1].hist(g_cur.ravel(), bins=256, range=[0,256], color='#00fcce', alpha=0.8)
            ax[1].set_title('Distribusi Layer Aktif', color='#f8fafc', pad=15)
            
            st.pyplot(fig)

# --- TAB 3: DUAL-CORE AI ENGINE ---
with tab_ai:
    st.markdown("<h3 style='color:#00a8ff; margin-bottom: 20px;'>Infrastruktur Deep Learning</h3>", unsafe_allow_html=True)
    
    col_ai1, col_ai2 = st.columns(2)
    
    # KORE CORE 1: YOLOv8
    with col_ai1:
        with st.container(border=True):
            st.markdown("#### 🎯 Core 1: YOLOv8 Object Detection")
            st.write("Menganalisis koordinat spasial (*bounding box*) dan tingkat keyakinan (*confidence score*) secara dinamis.")
            if st.button("🚀 Run YOLOv8 Engine", use_container_width=True):
                with st.spinner('Menjalankan feed-forward network YOLO...'):
                    img_detect = st.session_state.img_current.copy()
                    results = yolo_model(img_detect)
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        id_cls, conf = int(box.cls[0]), float(box.conf[0])
                        
                        # Penerjemahan ke Bahasa Indonesia menggunakan KAMUS_YOLO
                        nama_asli = results[0].names[id_cls]
                        label_indo = KAMUS_YOLO.get(id_cls, nama_asli.capitalize())
                        label = f"{label_indo} {conf*100:.0f}%"
                        
                        cv2.rectangle(img_detect, (x1, y1), (x2, y2), (0, 252, 206), 3)
                        (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(img_detect, (x1, max(0, y1 - h_t - 15)), (x1 + w_t, max(0, y1)), (0, 252, 206), -1)
                        cv2.putText(img_detect, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                    st.session_state.img_current = img_detect
                    st.toast(f"YOLO Selesai! Mendeteksi {len(results[0].boxes)} objek visual.", icon="🎯")
                    time.sleep(0.5)
                    st.rerun()

    # KORE CORE 2: CNN KERAS
    with col_ai2:
        with st.container(border=True):
            st.markdown("#### 🧠 Core 2: MobileNetV2 CNN Keras")
            st.write("Mengekstrak fitur matriks konvolusi murni `224x224` untuk probabilitas akurasi (*True Positives*).")
            if st.button("🧪 Run MobileNetV2 CNN", use_container_width=True):
                with st.spinner('Memproses Layer Konvolusi dan Softmax...'):
                    img_resized = cv2.resize(st.session_state.img_current.copy(), (224, 224))
                    x = preprocess_input(np.expand_dims(img_resized, axis=0))
                    preds = cnn_model.predict(x)
                    st.session_state.cnn_result = decode_predictions(preds, top=3)[0]
                    st.toast("Konvolusi CNN berhasil dikomputasi!", icon="🧠")

    # TAMPILAN HASIL CNN KERAS (Dengan Bahasa Indonesia)
    if st.session_state.cnn_result is not None:
        st.markdown("---")
        st.markdown("<h4 style='color:#00fcce;'>📈 Hasil Klasifikasi True Positives (Top 3):</h4>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        cols_m = [m1, m2, m3]
        for i, (_, label_inggris, prob) in enumerate(st.session_state.cnn_result):
            with cols_m[i]:
                # Menerjemahkan menggunakan fungsi Keras
                label_indo = terjemahkan_imagenet(label_inggris)
                st.metric(label=f"Peringkat #{i+1}", value=label_indo, delta=f"{prob*100:.2f}% Akurasi", delta_color="normal")