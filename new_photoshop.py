import os
# [PENTING] Mencegah konflik library OpenMP antara PyTorch (YOLO) dan Matplotlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import io

# ================= 10. KONFIGURASI HALAMAN & TEMA UTAMA =================
st.set_page_config(page_title="Mini Photoshop Pro", page_icon="🎨", layout="wide")

# CSS Injection Premium untuk UI Studio Estetik & Profesional
st.markdown("""
<style>
    /* Mengubah font global ke Inter / Segoe UI */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [data-testid="stSidebar"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Panel Utama Dashboard Header */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 14px;
        border: 1px solid #334155;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.4rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }
    
    /* Desain Elegan untuk Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0f17;
        border-right: 1px solid #1e293b;
    }
    
    /* Percantik tombol dekoratif */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 122, 204, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ================= MODEL INFERENCE CACHING =================
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO('yolov8n.pt')
    except Exception:
        return None

yolo_model = load_yolo_model()

# ================= INISIALISASI SESSION STATE =================
if 'img_original' not in st.session_state: st.session_state.img_original = None
if 'img_current' not in st.session_state: st.session_state.img_current = None

def reset_image():
    if st.session_state.img_original is not None:
        st.session_state.img_current = st.session_state.img_original.copy()

# ================= HEADER COMPONENT =================
st.markdown("""
<div class="header-box">
    <h1 class="header-title">🎨 Mini Photoshop Pro v2.0</h1>
    <div class="header-subtitle">Aplikasi Pengolahan Citra Digital Interaktif • Teknik Informatika PNJ</div>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR MENU (WORK STUDIO TOOLS) =================
st.sidebar.markdown("<h2 style='color:#00a8ff; font-weight:800; font-size:1.4rem;'>🛠️ STUDIO PANEL</h2>", unsafe_allow_html=True)

# --- 1 & 8. IMAGE MANAGEMENT & COMPRESSION ---
with st.sidebar.expander("📦 1. Image Management & Compression", expanded=True):
    uploaded_file = st.file_uploader("Pilih File Gambar", type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if st.session_state.img_original is None or st.session_state.get('last_file') != uploaded_file.name:
            st.session_state.img_original = img_rgb.copy()
            st.session_state.img_current = img_rgb.copy()
            st.session_state.last_file = uploaded_file.name

    if st.session_state.img_current is not None:
        st.button("🔄 Reset Gambar ke Awal", on_click=reset_image, use_container_width=True)
        
        st.markdown("**Simulasi Kompresi Ekspor (JPEG):**")
        quality_val = st.slider("Kualitas Simpan (%)", min_value=10, max_value=100, value=70)
        
        img_bgr_save = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr_save, [int(cv2.IMWRITE_JPEG_QUALITY), quality_val])
        st.download_button(label="💾 Download Hasil Citra", data=io.BytesIO(buffer), 
                           file_name="photoshop_output.jpg", mime="image/jpeg", use_container_width=True)

# Proteksi agar menu di bawah tidak merusak web jika belum ada berkas citra
if st.session_state.img_current is None:
    st.info("👈 Silakan unggah file citra (JPG, PNG, BMP) pada Studio Panel untuk memulai proses.")
    st.stop()

# --- 2. IMAGE ENHANCEMENT ---
with st.sidebar.expander("✨ 2. Image Enhancement"):
    cb_bright = st.slider("Kecerahan (Brightness)", -100, 100, 0)
    cb_contrast = st.slider("Kontras (Contrast)", 1.0, 3.0, 1.0, 0.1)
    if st.button("Terapkan Kecerahan & Kontras", use_container_width=True):
        st.session_state.img_current = cv2.convertScaleAbs(st.session_state.img_current, alpha=cb_contrast, beta=cb_bright)
        st.rerun()
        
    if st.button("Histogram Equalization", use_container_width=True):
        yuv = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        st.session_state.img_current = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        st.rerun()
        
    if st.button("Sharpening (Konvolusi)", use_container_width=True):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        st.session_state.img_current = cv2.filter2D(st.session_state.img_current, -1, kernel)
        st.rerun()

# --- 3. GEOMETRIC TRANSFORMATION ---
with st.sidebar.expander("📐 3. Transformasi Geometri"):
    geom_action = st.selectbox("Pilih Aksi Geometri:", ["Rotate Bebas", "Flip Horizontal", "Flip Vertical", "Manual Crop", "Translation (Geser)"])
    
    if geom_action == "Rotate Bebas":
        deg = st.slider("Derajat Putar", 0, 360, 0)
        if st.button("Putar Gambar"):
            h, w = st.session_state.img_current.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), deg, 1.0)
            st.session_state.img_current = cv2.warpAffine(st.session_state.img_current, M, (w, h))
            st.rerun()
            
    elif geom_action == "Flip Horizontal":
        if st.button("Terapkan Flip H"):
            st.session_state.img_current = cv2.flip(st.session_state.img_current, 1)
            st.rerun()
            
    elif geom_action == "Flip Vertical":
        if st.button("Terapkan Flip V"):
            st.session_state.img_current = cv2.flip(st.session_state.img_current, 0)
            st.rerun()
            
    elif geom_action == "Manual Crop":
        c_top = st.slider("Potong Atas (%)", 0, 50, 0)
        c_bottom = st.slider("Potong Bawah (%)", 0, 50, 0)
        c_left = st.slider("Potong Kiri (%)", 0, 50, 0)
        c_right = st.slider("Potong Kanan (%)", 0, 50, 0)
        if st.button("Terapkan Crop"):
            h, w = st.session_state.img_current.shape[:2]
            t_px, b_px = int(h*(c_top/100)), h - int(h*(c_bottom/100))
            l_px, r_px = int(w*(c_left/100)), w - int(w*(c_right/100))
            if t_px < b_px and l_px < r_px:
                st.session_state.img_current = st.session_state.img_current[t_px:b_px, l_px:r_px]
                st.rerun()
                
    elif geom_action == "Translation (Geser)":
        tx = st.number_input("Geser X (Piksel)", value=0)
        ty = st.number_input("Geser Y (Piksel)", value=0)
        if st.button("Geser Posisi"):
            h, w = st.session_state.img_current.shape[:2]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            st.session_state.img_current = cv2.warpAffine(st.session_state.img_current, M, (w, h))
            st.rerun()

# --- 4. IMAGE RESTORATION ---
with st.sidebar.expander("🩹 4. Restorasi & Noise Reduction"):
    if st.button("Gaussian Blur (Smoothing)", use_container_width=True):
        st.session_state.img_current = cv2.GaussianBlur(st.session_state.img_current, (11, 11), 0)
        st.rerun()
    if st.button("Median Filter (Hapus Salt & Pepper)", use_container_width=True):
        st.session_state.img_current = cv2.medianBlur(st.session_state.img_current, 5)
        st.rerun()

# --- 5. BINARY & EDGE PROCESSING ---
with st.sidebar.expander("💠 5. Biner, Tepi & Morfologi"):
    if st.button("Binary Thresholding (Citra Biner)", use_container_width=True):
        gray = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        st.session_state.img_current = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        st.rerun()
        
    metode_tepi = st.selectbox("Metode Segmentasi Tepi:", ["Canny", "Sobel", "Prewitt", "Robert", "Laplacian", "LoG"])
    if st.button("Jalankan Deteksi Tepi", use_container_width=True):
        gray = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
        edges = gray.copy()
        if metode_tepi == "Canny": edges = cv2.Canny(gray, 100, 200)
        elif metode_tepi == "Sobel":
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        elif metode_tepi == "Prewitt":
            kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]); ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            edges = cv2.convertScaleAbs(cv2.filter2D(gray, -1, kx) + cv2.filter2D(gray, -1, ky))
        elif metode_tepi == "Robert":
            kx = np.array([[1,0],[0,-1]]); ky = np.array([[0,1],[-1,0]])
            edges = cv2.convertScaleAbs(cv2.filter2D(gray, -1, kx) + cv2.filter2D(gray, -1, ky))
        elif metode_tepi == "Laplacian": edges = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
        elif metode_tepi == "LoG": edges = cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(gray, (3,3), 0), cv2.CV_64F))
        
        st.session_state.img_current = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        st.rerun()

    col_m1, col_m2 = st.columns(2)
    if col_m1.button("Erosion", use_container_width=True):
        kernel = np.ones((5,5), np.uint8)
        st.session_state.img_current = cv2.erode(st.session_state.img_current, kernel, iterations=1)
        st.rerun()
    if col_m2.button("Dilation", use_container_width=True):
        kernel = np.ones((5,5), np.uint8)
        st.session_state.img_current = cv2.dilate(st.session_state.img_current, kernel, iterations=1)
        st.rerun()

# --- 6 & 7. COLOR & SEGMENTATION ---
with st.sidebar.expander("🎨 6 & 7. Pemrosesan Warna & Segmentasi"):
    if st.button("Konversi RGB ke Grayscale", use_container_width=True):
        if len(st.session_state.img_current.shape) == 3:
            st.session_state.img_current = cv2.cvtColor(cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            st.rerun()
            
    if st.button("Channel Splitting (Ekstrak Channel Red)", use_container_width=True):
        if len(st.session_state.img_current.shape) == 3:
            r, g, b = cv2.split(st.session_state.img_current)
            zeros = np.zeros_like(r)
            st.session_state.img_current = cv2.merge([r, zeros, zeros])
            st.rerun()
            
    adj_hue = st.slider("Color Adjustment (Hue Shift)", 0, 180, 0)
    if st.button("Sesuaikan Warna (Hue)", use_container_width=True):
        hsv = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        h = cv2.add(h, adj_hue)
        st.session_state.img_current = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
        st.rerun()
        
    segmentasi_method = st.selectbox("Metode Segmentasi Citra:", ["Threshold-based (Masking)", "Region-based (K-Means)"])
    if st.button("Terapkan Segmentasi", use_container_width=True):
        if segmentasi_method == "Threshold-based (Masking)":
            hsv = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([180, 255, 255]))
            st.session_state.img_current = cv2.bitwise_and(st.session_state.img_current, st.session_state.img_current, mask=mask)
        elif segmentasi_method == "Region-based (K-Means)":
            data_pixels = st.session_state.img_current.reshape((-1, 3)).astype(np.float32)
            kriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data_pixels, 3, None, kriteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            st.session_state.img_current = centers[labels.flatten()].reshape((st.session_state.img_current.shape))
        st.rerun()

# --- 11. MACHINE LEARNING (CNN BONUS VALUE) ---
st.sidebar.markdown("---")
if st.sidebar.button("🤖 Deteksi Objek (CNN YOLOv8)", type="primary", use_container_width=True):
    with st.spinner('AI sedang memindai objek pada gambar...'):
        if yolo_model:
            kamus_indo = {0: 'Orang', 1: 'Sepeda', 2: 'Mobil', 3: 'Motor', 4: 'Pesawat', 5: 'Bus', 6: 'Kereta', 7: 'Truk', 8: 'Kapal', 14: 'Burung', 15: 'Kucing', 16: 'Anjing', 24: 'Ransel', 26: 'Tas Tangan', 27: 'Dasi', 39: 'Botol', 41: 'Cangkir', 56: 'Kursi', 62: 'TV', 63: 'Laptop', 64: 'Mouse', 66: 'Keyboard', 67: 'HP', 73: 'Buku'}
            
            img_detect = st.session_state.img_current.copy()
            results = yolo_model(img_detect)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                nama_default = results[0].names[class_id]
                nama_objek = kamus_indo.get(class_id, nama_default.capitalize())
                label = f"{nama_objek} {conf*100:.0f}%"
                
                warna = (0, 255, 0)
                cv2.rectangle(img_detect, (x1, y1), (x2, y2), warna, 2)
                (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_detect, (x1, max(0, y1 - h_t - 10)), (x1 + w_t, max(0, y1)), warna, -1)
                cv2.putText(img_detect, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
            st.session_state.img_current = img_detect
            st.sidebar.success(f"Pemindaian selesai! Menemukan {len(results[0].boxes)} objek.")
            st.rerun()

# ================= MAIN AREA: PREMIUM BEFORE-AFTER COMPONENT =================
col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        h_o, w_o = st.session_state.img_original.shape[:2]
        st.markdown(f"<h3 style='text-align:center; margin-bottom:2px;'>🖼️ Before (Original)</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#00a8ff; font-weight:600; margin-bottom:15px;'>Dimensi: {w_o} x {h_o} px</p>", unsafe_allow_html=True)
        st.image(st.session_state.img_original, use_container_width=True)

with col_right:
    with st.container(border=True):
        h_c, w_c = st.session_state.img_current.shape[:2]
        st.markdown(f"<h3 style='text-align:center; margin-bottom:2px;'>🎨 After (Edited)</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#00a8ff; font-weight:600; margin-bottom:15px;'>Dimensi: {w_c} x {h_c} px</p>", unsafe_allow_html=True)
        st.image(st.session_state.img_current, use_container_width=True)

# ================= 9. HISTOGRAM ANALYSIS AREA =================
st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("<h3 style='color:#ffffff; margin-top:5px; margin-bottom:15px;'>📊 9. Histogram Analysis (Grayscale)</h3>", unsafe_allow_html=True)
    
    if st.button("🔄 Hitung & Tampilkan Perbandingan Histogram Real-Time", use_container_width=True):
        gray_ori = cv2.cvtColor(st.session_state.img_original, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_original.shape) == 3 else st.session_state.img_original
        gray_cur = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
        
        # Plotting Premium Matplotlib Chart
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor('#0f1115')
        
        ax[0].set_facecolor('#1b1e24')
        ax[0].set_title("Distribusi Piksel Gambar Sebelum (Original)", color='#94a3b8', weight='bold')
        ax[0].hist(gray_ori.ravel(), bins=256, range=[0,256], color='#64748b', alpha=0.85)
        ax[0].set_ylabel('Frekuensi Piksel')
        ax[0].set_xlabel('Intensitas Warna (0 - 255)')
        ax[0].grid(True, linestyle='--', alpha=0.3)
        
        ax[1].set_facecolor('#1b1e24')
        ax[1].set_title("Distribusi Piksel Gambar Sesudah (Edited)", color='#00a8ff', weight='bold')
        ax[1].hist(gray_cur.ravel(), bins=256, range=[0,256], color='#00a8ff', alpha=0.85)
        ax[1].set_xlabel('Intensitas Warna (0 - 255)')
        ax[1].grid(True, linestyle='--', alpha=0.3)
        
        st.pyplot(fig)