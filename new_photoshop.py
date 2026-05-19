import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import io

# ================= KONFIGURASI HALAMAN =================
st.set_page_config(page_title="Mini Photoshop Pro", page_icon="🎨", layout="wide")

# ================= CACHE MODEL AI =================
# Menggunakan cache agar model YOLO tidak di-load berulang kali yang bikin lemot
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

yolo_model = load_yolo_model()

# ================= MANAJEMEN STATE (SESSION) =================
if 'img_original' not in st.session_state:
    st.session_state.img_original = None
if 'img_current' not in st.session_state:
    st.session_state.img_current = None

# ================= FUNGSI HELPER =================
def reset_image():
    if st.session_state.img_original is not None:
        st.session_state.img_current = st.session_state.img_original.copy()

# ================= ANTARMUKA UTAMA (MAIN AREA) =================
st.title("🎨 Mini Photoshop Pro - PNJ")
st.markdown("**UAS Pengolahan Citra Digital** | Terintegrasi dengan CNN YOLOv8")
st.markdown("---")

# ================= SIDEBAR (MENU KIRI) =================
st.sidebar.title("🛠️ Tools Photoshop")

# --- 1. Manajemen File ---
with st.sidebar.expander("1. Manajemen File & Kompresi", expanded=True):
    uploaded_file = st.file_uploader("Upload Gambar", type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif'])
    
    if uploaded_file is not None:
        # Konversi file upload ke array OpenCV (RGB)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Simpan ke session state jika baru diupload
        if st.session_state.img_original is None or st.session_state.get('last_file') != uploaded_file.name:
            st.session_state.img_original = img_rgb.copy()
            st.session_state.img_current = img_rgb.copy()
            st.session_state.last_file = uploaded_file.name

    if st.session_state.img_current is not None:
        st.button("🔄 Reset ke Gambar Awal", on_click=reset_image, use_container_width=True)
        
        # Fitur Save / Download (Simulasi Kompresi JPEG)
        img_bgr_save = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".jpg", img_bgr_save, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        io_buf = io.BytesIO(buffer)
        st.download_button(label="💾 Download Hasil (JPEG Kualitas 50%)", 
                           data=io_buf, file_name="edited_image.jpg", mime="image/jpeg", use_container_width=True)

# Jika belum ada gambar, hentikan render menu di bawahnya
if st.session_state.img_current is None:
    st.info("👈 Silakan upload gambar dari panel sebelah kiri untuk memulai pengeditan.")
    st.stop()

# --- 2. Enhancement ---
with st.sidebar.expander("2. Enhancement"):
    # Karena slider real-time bisa memberatkan web, kita pakai tombol "Apply"
    col_b, col_c = st.columns(2)
    bright = col_b.number_input("Brightness", min_value=-100, max_value=100, value=0)
    contrast = col_c.number_input("Contrast", min_value=1.0, max_value=3.0, value=1.0, step=0.1)
    
    if st.button("Terapkan B/C", use_container_width=True):
        st.session_state.img_current = cv2.convertScaleAbs(st.session_state.img_current, alpha=contrast, beta=bright)
        st.rerun()

    if st.button("Histogram Equalization", use_container_width=True):
        img = st.session_state.img_current
        if len(img.shape) == 3:
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            st.session_state.img_current = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            st.session_state.img_current = cv2.equalizeHist(img)
        st.rerun()

    if st.button("Sharpening", use_container_width=True):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        st.session_state.img_current = cv2.filter2D(st.session_state.img_current, -1, kernel)
        st.rerun()

# --- 3. Geometri ---
with st.sidebar.expander("3. Transformasi Geometri"):
    col1, col2, col3 = st.columns(3)
    if col1.button("Rotate 90°"):
        st.session_state.img_current = cv2.rotate(st.session_state.img_current, cv2.ROTATE_90_CLOCKWISE)
        st.rerun()
    if col2.button("Flip (H)"):
        st.session_state.img_current = cv2.flip(st.session_state.img_current, 1)
        st.rerun()
    if col3.button("Resize 50%"):
        h, w = st.session_state.img_current.shape[:2]
        st.session_state.img_current = cv2.resize(st.session_state.img_current, (w//2, h//2))
        st.rerun()

# --- 4. Restorasi ---
with st.sidebar.expander("4. Restorasi (Reduksi Noise)"):
    if st.button("Gaussian Blur"):
        st.session_state.img_current = cv2.GaussianBlur(st.session_state.img_current, (11, 11), 0)
        st.rerun()
    if st.button("Median Filter"):
        st.session_state.img_current = cv2.medianBlur(st.session_state.img_current, 5)
        st.rerun()

# --- 5. Biner & Tepi ---
with st.sidebar.expander("5. Biner, Tepi & Morfologi"):
    if st.button("Binary Thresholding"):
        img = st.session_state.img_current
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        st.session_state.img_current = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        st.rerun()

    metode_tepi = st.selectbox("Metode Tepi:", ["Canny", "Sobel", "Prewitt", "Robert", "Laplacian", "LoG"])
    if st.button("Deteksi Tepi"):
        img = st.session_state.img_current
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
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

    if st.button("Operasi Morfologi (Opening)"):
        kernel = np.ones((5,5), np.uint8)
        st.session_state.img_current = cv2.dilate(cv2.erode(st.session_state.img_current, kernel, iterations=1), kernel, iterations=1)
        st.rerun()

# --- 6 & 7. Warna & Segmentasi ---
with st.sidebar.expander("6 & 7. Warna & Segmentasi"):
    if st.button("RGB ke Grayscale"):
        img = st.session_state.img_current
        if len(img.shape) == 3:
            st.session_state.img_current = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            st.rerun()
    if st.button("Ekstrak Channel Merah"):
        img = st.session_state.img_current
        if len(img.shape) == 3:
            r, g, b = cv2.split(img) # Dalam format RGB
            zeros = np.zeros_like(r)
            st.session_state.img_current = cv2.merge([r, zeros, zeros])
            st.rerun()
    if st.button("Segmentasi Threshold (HSV)"):
        img = st.session_state.img_current
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        st.session_state.img_current = cv2.bitwise_and(img, img, mask=mask)
        st.rerun()

# --- 11. AI / CNN ---
st.sidebar.markdown("---")
if st.sidebar.button("🤖 Deteksi Objek (YOLOv8)", type="primary", use_container_width=True):
    with st.spinner('Menganalisis gambar menggunakan AI...'):
        if yolo_model:
            kamus_indo = {0: 'Orang', 1: 'Sepeda', 2: 'Mobil', 3: 'Motor', 4: 'Pesawat', 5: 'Bus', 6: 'Kereta', 7: 'Truk', 8: 'Kapal', 14: 'Burung', 15: 'Kucing', 16: 'Anjing', 24: 'Ransel', 26: 'Tas Tangan', 27: 'Dasi', 39: 'Botol', 41: 'Cangkir', 56: 'Kursi', 62: 'TV', 63: 'Laptop', 64: 'Mouse', 66: 'Keyboard', 67: 'HP', 73: 'Buku'}
            
            img = st.session_state.img_current.copy()
            results = yolo_model(img)
            jumlah_objek = len(results[0].boxes)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                nama_default = results[0].names[class_id]
                nama_objek = kamus_indo.get(class_id, nama_default.capitalize())
                label = f"{nama_objek} {conf*100:.0f}%"
                
                warna_kotak = (0, 255, 0) # Hijau RGB
                cv2.rectangle(img, (x1, y1), (x2, y2), warna_kotak, 2)
                (w_teks, h_teks), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, max(0, y1 - h_teks - 10)), (x1 + w_teks, max(0, y1)), warna_kotak, -1)
                cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
            st.session_state.img_current = img
            st.sidebar.success(f"Ditemukan {jumlah_objek} objek!")

# ================= RENDER PREVIEW (TAMPILAN GAMBAR) =================
col1, col2 = st.columns(2)

with col1:
    h_ori, w_ori = st.session_state.img_original.shape[:2]
    st.markdown(f"<h3 style='text-align: center;'>Before (Original)</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>Dimensi: {w_ori} x {h_ori} px</p>", unsafe_allow_html=True)
    st.image(st.session_state.img_original, use_container_width=True)

with col2:
    h_cur, w_cur = st.session_state.img_current.shape[:2]
    st.markdown(f"<h3 style='text-align: center;'>After (Edited)</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>Dimensi: {w_cur} x {h_cur} px</p>", unsafe_allow_html=True)
    st.image(st.session_state.img_current, use_container_width=True)

# ================= 9. ANALISIS HISTOGRAM =================
st.markdown("---")
if st.button("📊 Tampilkan Analisis Histogram (Before vs After)"):
    img_ori_gray = cv2.cvtColor(st.session_state.img_original, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_original.shape) == 3 else st.session_state.img_original
    img_cur_gray = cv2.cvtColor(st.session_state.img_current, cv2.COLOR_RGB2GRAY) if len(st.session_state.img_current.shape) == 3 else st.session_state.img_current
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].set_title("Distribusi Pixel (Before)")
    ax[0].hist(img_ori_gray.ravel(), bins=256, range=[0,256], color='gray')
    ax[0].set_ylabel('Jumlah Pixel')
    ax[0].set_xlabel('Intensitas (0-255)')
    
    ax[1].set_title("Distribusi Pixel (After)")
    ax[1].hist(img_cur_gray.ravel(), bins=256, range=[0,256], color='blue')
    ax[1].set_xlabel('Intensitas (0-255)')
    
    st.pyplot(fig)