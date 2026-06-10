import os
# Mencegah konflik library OpenMP agar aplikasi tidak crash saat AI dan Matplotlib berjalan bersamaan
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import time

from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# ================= 1. KONFIGURASI HALAMAN =================
st.set_page_config(page_title="Mini Photoshop Pro | Ultimate", page_icon="💎", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }
    .hero-container { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); padding: 2.5rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 20px 40px rgba(0,0,0,0.4); position: relative; overflow: hidden; }
    .hero-container::before { content: ""; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(0,168,255,0.1) 0%, transparent 60%); animation: rotate 20s linear infinite; z-index: 0; }
    @keyframes rotate { 100% { transform: rotate(360deg); } }
    .hero-content { position: relative; z-index: 1; }
    .hero-title { background: -webkit-linear-gradient(45deg, #00a8ff, #00fcce); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 3rem; margin: 0; letter-spacing: -1px; }
    .hero-subtitle { color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; font-weight: 400; }
    div.stButton > button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; font-weight: 600; transition: all 0.3s ease; width: 100%; }
    div.stButton > button:hover { background: #00a8ff; border-color: #00a8ff; color: #ffffff; transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0, 168, 255, 0.3); }
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; color: #00fcce; }
</style>
""", unsafe_allow_html=True)

# ================= 2. CACHING AI MODELS =================
@st.cache_resource(show_spinner=False)
def load_models():
    # Memuat model Deep Learning ke dalam RAM secara efisien
    yolo = YOLO('yolov8n.pt')
    cnn = MobileNetV2(weights='imagenet')
    return yolo, cnn

with st.spinner("Menginisialisasi Engine Deep Learning..."):
    yolo_model, cnn_model = load_models()

# State management untuk menyimpan matriks citra agar tidak hilang saat UI direfresh
if 'img_original' not in st.session_state: st.session_state.img_original = None
if 'img_current' not in st.session_state: st.session_state.img_current = None
if 'cnn_result' not in st.session_state: st.session_state.cnn_result = None

def reset_image():
    # Spesifikasi 1: Reset ke gambar awal (mengembalikan matriks orisinal)
    if st.session_state.img_original is not None:
        st.session_state.img_current = st.session_state.img_original.copy()
        st.session_state.cnn_result = None
        st.toast('🔄 Gambar dikembalikan ke kondisi awal.', icon='♻️')

# ================= KAMUS PENERJEMAH =================
KAMUS_YOLO = {0: 'Orang', 1: 'Sepeda', 2: 'Mobil', 3: 'Motor', 4: 'Pesawat', 5: 'Bus', 6: 'Kereta', 7: 'Truk', 8: 'Kapal', 14: 'Burung', 15: 'Kucing', 16: 'Anjing', 17: 'Kuda', 18: 'Domba', 19: 'Sapi', 20: 'Gajah', 24: 'Ransel', 25: 'Payung', 26: 'Tas Tangan', 27: 'Dasi', 39: 'Botol', 41: 'Cangkir', 56: 'Kursi', 62: 'Monitor / TV', 63: 'Laptop', 64: 'Mouse Komputer', 66: 'Keyboard', 67: 'Ponsel', 73: 'Buku'}

def terjemahkan_imagenet(label_inggris):
    kamus_keras = {'suit': 'Setelan Jas Formal', 'windsor_tie': 'Dasi Windsor', 'loafer': 'Sepatu Pantofel', 'bow_tie': 'Dasi Kupu-kupu', 'laptop': 'Laptop', 'cellular_telephone': 'Ponsel Pintar / HP', 'mouse': 'Mouse Komputer', 'keyboard': 'Keyboard', 'coffee_mug': 'Cangkir Kopi', 'sports_car': 'Mobil Sport', 'passenger_car': 'Mobil Penumpang', 'backpack': 'Tas Ransel'}
    label_bersih = label_inggris.lower().strip()
    return kamus_keras.get(label_bersih, label_bersih.replace('_', ' ').title())

# ================= HEADER HERO =================
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <h1 class="hero-title">💎 Mini Photoshop Pro Workspace</h1>
        <div class="hero-subtitle">High-Fidelity Digital Image Processing & Dual-Core AI Architecture</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color:#00a8ff; font-weight:800; text-align:center;'>🛠️ WORKSPACE TOOLS</h2><hr style='border-color:#334155;'>", unsafe_allow_html=True)


# ================== SPESIFIKASI 1 SAMPAI 8 ==================

# --- 1. IMAGE MANAGEMENT & 8. IMAGE COMPRESSION ---
with st.sidebar.expander("📁 1. Manajemen & Ekspor Resolusi", expanded=True):
    uploaded_file = st.file_uploader("Load Image (JPG, PNG, BMP)", type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif'])
    if uploaded_file is not None:
        # Spesifikasi 1: Load image file lokal & konversi byte array ke matriks citra RGB
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if st.session_state.img_original is None or st.session_state.get('last_file') != uploaded_file.name:
            st.session_state.img_original = img_rgb.copy()
            st.session_state.img_current = img_rgb.copy()
            st.session_state.last_file = uploaded_file.name
            st.session_state.cnn_result = None
            st.toast("✅ File berhasil dimuat!", icon="🚀")

    if st.session_state.img_current is not None:
        col_res1, col_res2 = st.columns(2)
        col_res1.button("🔄 Reset", on_click=reset_image, use_container_width=True)
        
        # Spesifikasi 8: Image Compression (Simulasi kompresi JPEG)
        # Teknis: Menggunakan metode Kuantisasi dan Huffman bawaan algoritma JPEG OpenCV
        quality_val = st.slider("Kualitas Kompresi JPEG (%)", 10, 100, 85)
        img_bgr_save = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr_save, [int(cv2.IMWRITE_JPEG_QUALITY), quality_val])
        # Spesifikasi 1: Save image dengan custom filename & format
        st.download_button("💾 Ekspor Citra", io.BytesIO(buffer), "ps_output_hq.jpg", "image/jpeg", use_container_width=True)

if st.session_state.img_current is None:
    st.info("💡 Sistem standby. Silakan unggah citra pada panel *Workspace Tools* di sebelah kiri untuk memulai komputasi matriks.")
    st.stop()


# --- 2. IMAGE ENHANCEMENT ---
with st.sidebar.expander("✨ 2. Image Enhancement"):
    st.caption("Point Processing & Spatial Enhancement")
    
    # Brightness & Contrast Adjustment
    # Teknis: Fungsi linear g(x) = alpha * f(x) + beta (alpha=contrast, beta=brightness)
    cb_bright = st.slider("Brightness Adjustment", -100, 100, 0)
    cb_contrast = st.slider("Contrast Adjustment", 1.0, 3.0, 1.0, 0.1)
    if st.button("Terapkan Brightness & Contrast", use_container_width=True):
        st.session_state.img_current = cv2.convertScaleAbs(st.session_state.img_current, alpha=cb_contrast, beta=cb_bright)
        st.session_state.cnn_result = None; st.rerun()
        
    c_e1, c_e2, c_e3 = st.columns(3)
    # Histogram Equalization
    # Teknis: Meratakan distribusi intensitas pada channel Luminance (YUV) agar warna asli tidak rusak
    if c_e1.button("Hist Eq"):
        yuv = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        st.session_state.img_current = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        st.session_state.cnn_result = None; st.rerun()
        
    # Sharpening
    # Teknis: Kernel convolution dengan matriks High-Pass Filter 3x3 untuk menebalkan tepi
    if c_e2.button("Sharpen"):
        kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        st.session_state.img_current = cv2.filter2D(st.session_state.img_current, -1, kernel_sharpen)
        st.session_state.cnn_result = None; st.rerun()
        
    # Smoothing (Blur)
    # Teknis: Spatial filtering menggunakan box filter (rata-rata/mean filter)
    if c_e3.button("Smooth"):
        st.session_state.img_current = cv2.blur(st.session_state.img_current, (5, 5))
        st.session_state.cnn_result = None; st.rerun()


# --- 3. GEOMETRIC TRANSFORMATION ---
with st.sidebar.expander("📐 3. Geometric Transformation"):
    geom_action = st.selectbox("Operasi Geometri:", ["Rotate (0°-360°)", "Flip (horizontal/vertical)", "Crop (drag area / custom)", "Resize (scaling)", "Translation (geser posisi)"])
    
    # Rotate (0-360 derajat)
    if geom_action == "Rotate (0°-360°)":
        deg = st.slider("Derajat", 0, 360, 0)
        if st.button("Terapkan Rotate", use_container_width=True):
            # Teknis: Transformasi matriks affine rotasi berdasarkan titik tengah (center) citra
            h, w = st.session_state.img_current.shape[:2]
            M_rotasi = cv2.getRotationMatrix2D((w//2, h//2), deg, 1.0)
            st.session_state.img_current = cv2.warpAffine(st.session_state.img_current, M_rotasi, (w, h))
            st.session_state.cnn_result = None; st.rerun()
            
    # Flip (Horizontal/Vertical)
    # Teknis: Pencerminan matriks berdasarkan sumbu
    elif geom_action == "Flip (horizontal/vertical)":
        c_m1, c_m2 = st.columns(2)
        if c_m1.button("Flip Horizontal"): st.session_state.img_current = cv2.flip(st.session_state.img_current, 1); st.session_state.cnn_result = None; st.rerun()
        if c_m2.button("Flip Vertical"): st.session_state.img_current = cv2.flip(st.session_state.img_current, 0); st.session_state.cnn_result = None; st.rerun()
        
    # Resize (Scaling)
    elif geom_action == "Resize (scaling)":
        skala = st.slider("Persentase Skala", 10, 200, 50)
        if st.button("Terapkan Resize", use_container_width=True):
            # Teknis: Perubahan dimensi matriks dengan Interpolasi Bilinear (INTER_LINEAR)
            h, w = st.session_state.img_current.shape[:2]
            st.session_state.img_current = cv2.resize(st.session_state.img_current, (int(w*skala/100), int(h*skala/100)), interpolation=cv2.INTER_LINEAR)
            st.session_state.cnn_result = None; st.rerun()
            
    # Crop (Pemotongan)
    elif geom_action == "Crop (drag area / custom)":
        c_t, c_b = st.slider("Potong Atas (%)", 0, 50, 0), st.slider("Potong Bawah (%)", 0, 50, 0)
        c_l, c_r = st.slider("Potong Kiri (%)", 0, 50, 0), st.slider("Potong Kanan (%)", 0, 50, 0)
        if st.button("Terapkan Crop", use_container_width=True):
            # Teknis: Slicing (pemotongan index array matriks) secara spesifik
            h, w = st.session_state.img_current.shape[:2]
            t_px, b_px = int(h*(c_t/100)), h - int(h*(c_b/100))
            l_px, r_px = int(w*(c_l/100)), w - int(w*(c_r/100))
            if t_px < b_px and l_px < r_px:
                st.session_state.img_current = st.session_state.img_current[t_px:b_px, l_px:r_px]
                st.session_state.cnn_result = None; st.rerun()
                
    # Translation (Geser posisi)
    elif geom_action == "Translation (geser posisi)":
        tx, ty = st.number_input("Geser Sumbu X", value=0), st.number_input("Geser Sumbu Y", value=0)
        if st.button("Terapkan Translation", use_container_width=True):
            # Teknis: Transformasi matriks affine translasi (menambah nilai konstan x dan y)
            h, w = st.session_state.img_current.shape[:2]
            M_translasi = np.float32([[1, 0, tx], [0, 1, ty]])
            st.session_state.img_current = cv2.warpAffine(st.session_state.img_current, M_translasi, (w, h))
            st.session_state.cnn_result = None; st.rerun()


# --- 4. IMAGE RESTORATION (NOISE REDUCTION) ---
with st.sidebar.expander("🩹 4. Image Restoration"):
    # Gaussian Blur
    # Teknis: Spatial filtering konvolusi menggunakan distribusi normal (Low-pass filter)
    if st.button("Gaussian Blur", use_container_width=True):
        st.session_state.img_current = cv2.GaussianBlur(st.session_state.img_current, (11, 11), 0)
        st.session_state.cnn_result = None; st.rerun()
        
    # Median Filter
    # Teknis: Kernel convolution non-linear (mengambil nilai median dari neighbor pixel)
    if st.button("Median Filter", use_container_width=True):
        st.session_state.img_current = cv2.medianBlur(st.session_state.img_current, 5)
        st.session_state.cnn_result = None; st.rerun()
        
    # Noise Removal (Salt & Pepper)
    # Teknis: Menggunakan spatial filtering Median kernel besar (7x7) khusus untuk salt & pepper
    if st.button("Noise Removal (Salt & Pepper)", use_container_width=True):
        st.session_state.img_current = cv2.medianBlur(st.session_state.img_current, 7)
        st.session_state.cnn_result = None; st.rerun()


# --- 5. BINARY & EDGE PROCESSING ---
with st.sidebar.expander("💠 5. Binary & Edge Processing"):
    
    # Thresholding (Binary Image)
    # Teknis: Operasi piksel biner dengan nilai batas thresholding 127
    if st.button("Thresholding (binary image)", use_container_width=True):
        g = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
        _, b = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
        st.session_state.img_current = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB); st.session_state.cnn_result = None; st.rerun()

    # Edge Detection
    metode_tepi = st.selectbox("Edge Detection:", ["Canny", "Sobel", "Prewitt", "Robert", "Laplacian", "Laplacian of gaussian"])
    if st.button("Terapkan Edge Detection", use_container_width=True):
        g = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
        edges = g.copy()
        
        # Teknis Edge Detection: Mengekstrak fitur gradien/turunan spasial pada matriks citra
        if metode_tepi == "Canny": 
            edges = cv2.Canny(g, 100, 200) # Canny menggunakan Hysteresis thresholding
        elif metode_tepi == "Sobel": 
            # Sobel menghitung aproksimasi turunan orde pertama (Horizontal & Vertikal)
            edges = cv2.convertScaleAbs(cv2.magnitude(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)))
        elif metode_tepi == "Prewitt": 
            kx, ky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]), np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            edges = cv2.convertScaleAbs(cv2.filter2D(g, -1, kx) + cv2.filter2D(g, -1, ky))
        elif metode_tepi == "Robert": 
            kx, ky = np.array([[1,0],[0,-1]]), np.array([[0,1],[-1,0]])
            edges = cv2.convertScaleAbs(cv2.filter2D(g, -1, kx) + cv2.filter2D(g, -1, ky))
        elif metode_tepi == "Laplacian": 
            edges = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F)) # Turunan orde kedua isotropik
        elif metode_tepi == "Laplacian of gaussian": 
            # LoG: Spatial filtering Gaussian Blur diikuti dengan Laplacian
            edges = cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(g, (3,3), 0), cv2.CV_64F))
            
        st.session_state.img_current = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB); st.session_state.cnn_result = None; st.rerun()

    # Morphology (Erosion & Dilation)
    c_m1, c_m2 = st.columns(2)
    # Teknis Morfologi: Operasi himpunan matematika menggunakan Kernel structuring element 5x5
    kernel_structuring = np.ones((5,5), np.uint8) 
    
    if c_m1.button("Erosion", use_container_width=True):
        st.session_state.img_current = cv2.erode(st.session_state.img_current, kernel_structuring, iterations=1)
        st.session_state.cnn_result = None; st.rerun()
    if c_m2.button("Dilation", use_container_width=True):
        st.session_state.img_current = cv2.dilate(st.session_state.img_current, kernel_structuring, iterations=1)
        st.session_state.cnn_result = None; st.rerun()


# --- 6. COLOR PROCESSING ---
with st.sidebar.expander("🎨 6. Color Processing"):
    
    # RGB -> Grayscale
    # Teknis: Transformasi ruang warna dari 3 channel RGB menjadi 1 channel intensitas
    if st.button("RGB -> Grayscale", use_container_width=True) and len(st.session_state.img_current.shape) == 3:
        st.session_state.img_current = cv2.cvtColor(cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        st.session_state.cnn_result = None; st.rerun()
        
    # Channel Splitting
    split_chan = st.selectbox("Channel splitting (R, G, B):", ["Red Channel", "Green Channel", "Blue Channel"])
    if st.button("Terapkan Splitting", use_container_width=True) and len(st.session_state.img_current.shape) == 3:
        # Teknis: Manipulasi channel array menggunakan fungsi split dan merge OpenCV
        r, g, b = cv2.split(st.session_state.img_current)
        zeros = np.zeros_like(r) # Array kosong untuk mematikan channel lain
        if split_chan == "Red Channel": st.session_state.img_current = cv2.merge([r, zeros, zeros])
        elif split_chan == "Green Channel": st.session_state.img_current = cv2.merge([zeros, g, zeros])
        elif split_chan == "Blue Channel": st.session_state.img_current = cv2.merge([zeros, zeros, b])
        st.session_state.cnn_result = None; st.rerun()
        
    # Color Adjustment (Hue/Saturation sederhana)
    st.markdown("**Color adjustment (hue/saturation):**")
    adj_hue, adj_sat = st.slider("Hue Offset", -90, 90, 0), st.slider("Saturation Offset", -90, 90, 0)
    if st.button("Terapkan Adjust Color", use_container_width=True):
        # Teknis: Transformasi ke ruang warna HSV lalu menggeser nilai H dan S secara matematis
        hsv = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2HSV).astype(np.int16)
        h, s, v = cv2.split(hsv)
        h = np.clip(h + adj_hue, 0, 180)
        s = np.clip(s + adj_sat, 0, 255)
        st.session_state.img_current = cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2RGB)
        st.session_state.cnn_result = None; st.rerun()


# --- 7. IMAGE SEGMENTATION ---
with st.sidebar.expander("🧩 7. Image Segmentation"):
    metode_seg = st.selectbox("Metode:", ["Threshold-based segmentation", "Edge-based segmentation", "Region-based sederhana"])
    if st.button("Eksekusi Segmentasi", use_container_width=True):
        
        # 1. Threshold-based segmentation
        # Teknis: Clustering sederhana / Masking region citra berdasarkan ambang batas HSV
        if metode_seg == "Threshold-based segmentation":
            hsv = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
            st.session_state.img_current = cv2.bitwise_and(st.session_state.img_current, st.session_state.img_current, mask=mask)
            
        # 2. Edge-based segmentation
        # Teknis: Region extraction menggunakan batas gradien objek (Canny) yang dilasi sebagai mask
        elif metode_seg == "Edge-based segmentation":
            gray = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
            edges = cv2.Canny(gray, 50, 150)
            edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
            st.session_state.img_current = cv2.bitwise_and(st.session_state.img_current, st.session_state.img_current, mask=edges_dilated)
            
        # 3. Region-based sederhana
        # Teknis: Clustering piksel unsupervised menggunakan algoritma Machine Learning K-Means (K=3 region)
        elif metode_seg == "Region-based sederhana":
            pixels = st.session_state.img_current.reshape((-1, 3)).astype(np.float32)
            kriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 3, None, kriteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            st.session_state.img_current = np.uint8(centers)[labels.flatten()].reshape(st.session_state.img_current.shape)
            
        st.session_state.cnn_result = None; st.rerun()


# ================= MAIN AREA: TABS (SPESIFIKASI 9 & 10) =================
# Spesifikasi 10: User Interface (GUI) -> Panel preview before vs after
tab_visual, tab_hist, tab_ai = st.tabs(["👁️ 10. Panel Preview (Before vs After)", "📊 9. Histogram Analysis", "🧠 Pengenalan Objek AI (Nilai Tambah)"])

with tab_visual:
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("<h4 style='color:#94a3b8; text-align:center;'>Source Image (Before)</h4>", unsafe_allow_html=True)
        st.image(st.session_state.img_original, use_container_width=True)
    with col_img2:
        st.markdown("<h4 style='color:#00fcce; text-align:center;'>Active Layer (After)</h4>", unsafe_allow_html=True)
        st.image(st.session_state.img_current, use_container_width=True)


# Spesifikasi 9: Histogram Analysis
with tab_hist:
    st.markdown("<h3 style='color:#00a8ff;'>9. Perbandingan Histogram Grayscale Before–After</h3>", unsafe_allow_html=True)
    if st.button("📉 Menampilkan Histogram Grayscale"):
        with st.spinner("Mengomputasi array piksel..."):
            g_ori = cv2.cvtColor(st.session_state.img_original, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_original.shape)==3 else st.session_state.img_original
            g_cur = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape)==3 else st.session_state.img_current
            
            # Teknis: Visualisasi distribusi intensitas pixel (0-255) menggunakan grafik Matplotlib
            plt.style.use('dark_background')
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            fig.patch.set_facecolor('#0e1117')
            
            # Meratakan matriks piksel (ravel) untuk dihitung frekuensinya
            ax[0].set_facecolor('#1e293b'); ax[0].grid(color='#334155', linestyle='--', alpha=0.5)
            ax[0].hist(g_ori.ravel(), bins=256, range=[0,256], color='#94a3b8', alpha=0.8)
            ax[0].set_title('Distribusi Intensitas Before', color='#f8fafc', pad=15)
            
            ax[1].set_facecolor('#1e293b'); ax[1].grid(color='#334155', linestyle='--', alpha=0.5)
            ax[1].hist(g_cur.ravel(), bins=256, range=[0,256], color='#00fcce', alpha=0.8)
            ax[1].set_title('Distribusi Intensitas After', color='#f8fafc', pad=15)
            st.pyplot(fig)


# Spesifikasi Nilai Tambah: Pengenalan Objek Machine Learning (Metode CNN)
with tab_ai:
    st.markdown("<h3 style='color:#00a8ff; margin-bottom: 20px;'>Pengenalan Objek dengan Machine Learning (Metode CNN)</h3>", unsafe_allow_html=True)
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        with st.container(border=True):
            st.markdown("#### 🎯 Rekognisi Objek: YOLOv8")
            if st.button("🚀 Run YOLOv8 Engine", use_container_width=True):
                with st.spinner('Menjalankan deteksi...'):
                    img_detect = st.session_state.img_current.copy()
                    
                    # Teknis: Feed-forward image network ke model YOLO untuk deteksi spasial bounding box
                    results = yolo_model(img_detect)
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        id_cls, conf = int(box.cls[0]), float(box.conf[0])
                        
                        nama_asli = results[0].names[id_cls]
                        label_indo = KAMUS_YOLO.get(id_cls, nama_asli.capitalize())
                        label = f"{label_indo} {conf*100:.0f}%"
                        
                        # Menggambar bounding box dan probabilitas (Confidence score) di atas citra
                        cv2.rectangle(img_detect, (x1, y1), (x2, y2), (0, 252, 206), 3)
                        (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(img_detect, (x1, max(0, y1 - h_t - 15)), (x1 + w_t, max(0, y1)), (0, 252, 206), -1)
                        cv2.putText(img_detect, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                    st.session_state.img_current = img_detect
                    st.toast(f"YOLO Selesai! Mendeteksi {len(results[0].boxes)} objek.", icon="🎯")
                    time.sleep(0.5)
                    st.rerun()

    with col_ai2:
        with st.container(border=True):
            st.markdown("#### 🧠 Rekognisi Objek: MobileNetV2 CNN Keras")
            if st.button("🧪 Run MobileNetV2 CNN", use_container_width=True):
                with st.spinner('Memproses CNN Keras...'):
                    # Teknis: Resize wajib ke 224x224 sebelum diproses oleh layer konvolusi matriks CNN Keras
                    img_resized = cv2.resize(st.session_state.img_current.copy(), (224, 224))
                    x = preprocess_input(np.expand_dims(img_resized, axis=0))
                    
                    # Forward propagation pada layer-layer MobileNetV2 untuk probabilitas klasifikasi
                    preds = cnn_model.predict(x)
                    st.session_state.cnn_result = decode_predictions(preds, top=3)[0]
                    st.toast("CNN Klasifikasi berhasil!", icon="🧠")

    if st.session_state.cnn_result is not None:
        st.markdown("---")
        st.markdown("<h4 style='color:#00fcce;'>📈 Hasil Prediksi AI (Top 3):</h4>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        cols_m = [m1, m2, m3]
        for i, (_, label_inggris, prob) in enumerate(st.session_state.cnn_result):
            with cols_m[i]:
                label_indo = terjemahkan_imagenet(label_inggris)
                st.metric(label=f"Prediksi #{i+1}", value=label_indo, delta=f"{prob*100:.2f}%", delta_color="normal")