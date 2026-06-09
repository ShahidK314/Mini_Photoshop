import os
# [PENTING] Mencegah program crash saat menggabungkan AI dan Matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QGroupBox, QMessageBox, QScrollArea, QComboBox, 
                             QSizePolicy, QStatusBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# IMPORT CNN MURNI DARI TENSORFLOW KERAS (PENGGANTI YOLO)
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

class MiniPhotoshopPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Photoshop Pro - Versi Murni CNN (Keras/TensorFlow)")
        self.setGeometry(50, 50, 1300, 750)

        # State Gambar
        self.cv_image_original = None
        self.cv_image_current = None

        # Terapkan Tema UI Modern (DARK MODE - ADOBE PHOTOSHOP STYLE)
        self.apply_stylesheet()

        # Load Model Murni CNN (MobileNetV2)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Memuat Model Arsitektur CNN Murni (TensorFlow/Keras)...")
        self.statusBar.setStyleSheet("background-color: #007acc; color: white; font-weight: bold;")
        
        try:
            self.cnn_model = MobileNetV2(weights='imagenet') 
            self.statusBar.showMessage("Sistem Siap digunakan (Mode Klasifikasi CNN).", 5000)
        except Exception as e:
            self.cnn_model = None
            self.statusBar.showMessage(f"Gagal memuat CNN Keras: {e}")

        self.initUI()

    def apply_stylesheet(self):
        # Desain UI Dark Mode Premium ala Aplikasi Editing Profesional
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #cccccc; font-family: 'Segoe UI', Arial, sans-serif; }
            
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #3a3a3a; 
                border-radius: 8px; 
                margin-top: 15px; 
                padding-top: 20px; 
                background-color: #252526;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left; 
                padding: 0 10px; 
                color: #00a8ff; 
                left: 10px;
                top: -5px;
            }
            
            QPushButton { 
                background-color: #333333; 
                border: 1px solid #4a4a4a; 
                border-radius: 5px; 
                padding: 8px; 
                color: #e0e0e0; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #007acc; border-color: #007acc; color: white; }
            QPushButton:pressed { background-color: #005f9e; }
            
            QSlider::groove:horizontal { 
                border: 1px solid #3a3a3a; 
                height: 6px; 
                background: #1e1e1e; 
                border-radius: 3px; 
            }
            QSlider::handle:horizontal { 
                background: #00a8ff; 
                width: 14px; 
                margin: -4px 0; 
                border-radius: 7px; 
            }
            
            QLabel { color: #e0e0e0; }
            
            QComboBox { 
                background-color: #333333; 
                border: 1px solid #4a4a4a; 
                border-radius: 5px; 
                padding: 6px; 
                color: #e0e0e0; 
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #333333;
                color: #e0e0e0;
                selection-background-color: #007acc;
            }
            
            QScrollArea { border: none; background-color: transparent; }
            QScrollArea > QWidget > QWidget { background-color: transparent; }
        """)

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ================= KIRI: SCROLLABLE TOOLBAR =================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(340)
        
        toolbar_widget = QWidget()
        toolbar_layout = QVBoxLayout(toolbar_widget)
        toolbar_layout.setAlignment(Qt.AlignTop)

        # 1. Image Management
        grp_file = QGroupBox("1. Manajemen File & 8. Kompresi")
        lay_file = QVBoxLayout()
        btn_load = QPushButton("Load Image"); btn_load.clicked.connect(self.load_image)
        btn_save = QPushButton("Save Image (JPEG Compression)"); btn_save.clicked.connect(self.save_image)
        btn_reset = QPushButton("Reset ke Gambar Awal"); btn_reset.clicked.connect(self.reset_image)
        lay_file.addWidget(btn_load); lay_file.addWidget(btn_save); lay_file.addWidget(btn_reset)
        grp_file.setLayout(lay_file); toolbar_layout.addWidget(grp_file)

        # 2. Image Enhancement
        grp_enhance = QGroupBox("2. Enhancement")
        lay_enhance = QVBoxLayout()
        self.sl_bright = QSlider(Qt.Horizontal); self.sl_bright.setRange(-100, 100); self.sl_bright.valueChanged.connect(self.apply_brightness_contrast)
        self.sl_contrast = QSlider(Qt.Horizontal); self.sl_contrast.setRange(10, 30); self.sl_contrast.setValue(10); self.sl_contrast.valueChanged.connect(self.apply_brightness_contrast)
        btn_hist = QPushButton("Histogram Equalization"); btn_hist.clicked.connect(self.apply_hist_eq)
        btn_sharp = QPushButton("Sharpening"); btn_sharp.clicked.connect(self.apply_sharpen)
        lay_enhance.addWidget(QLabel("Kecerahan (Brightness):")); lay_enhance.addWidget(self.sl_bright)
        lay_enhance.addWidget(QLabel("Kontras (Contrast):")); lay_enhance.addWidget(self.sl_contrast)
        lay_enhance.addWidget(btn_hist); lay_enhance.addWidget(btn_sharp)
        grp_enhance.setLayout(lay_enhance); toolbar_layout.addWidget(grp_enhance)

        # 3. Geometric Transformation
        grp_geom = QGroupBox("3. Transformasi Geometri")
        lay_geom = QVBoxLayout()
        btn_rot = QPushButton("Rotate 90°"); btn_rot.clicked.connect(lambda: self.apply_geometry("rotate"))
        btn_flip = QPushButton("Flip Horizontal"); btn_flip.clicked.connect(lambda: self.apply_geometry("flip"))
        btn_resize = QPushButton("Resize 50% (Ubah Resolusi)"); btn_resize.clicked.connect(lambda: self.apply_geometry("resize"))
        lay_geom.addWidget(btn_rot); lay_geom.addWidget(btn_flip); lay_geom.addWidget(btn_resize)
        grp_geom.setLayout(lay_geom); toolbar_layout.addWidget(grp_geom)

        # 4. Image Restoration
        grp_rest = QGroupBox("4. Restorasi (Reduksi Noise)")
        lay_rest = QVBoxLayout()
        btn_gauss = QPushButton("Gaussian Blur (Pelemahan Halus)"); btn_gauss.clicked.connect(lambda: self.apply_restoration("gaussian"))
        btn_median = QPushButton("Median Filter (Hapus Bintik Noise)"); btn_median.clicked.connect(lambda: self.apply_restoration("median"))
        lay_rest.addWidget(btn_gauss); lay_rest.addWidget(btn_median)
        grp_rest.setLayout(lay_rest); toolbar_layout.addWidget(grp_rest)

        # 5. Binary & Edge Processing
        grp_edge = QGroupBox("5. Biner, Tepi & Morfologi")
        lay_edge = QVBoxLayout()
        btn_thresh = QPushButton("Binary Thresholding"); btn_thresh.clicked.connect(self.apply_thresholding)
        self.combo_edge = QComboBox()
        self.combo_edge.addItems(["Canny", "Sobel", "Prewitt", "Robert", "Laplacian", "LoG"])
        btn_edge = QPushButton("Deteksi Tepi"); btn_edge.clicked.connect(self.apply_edge_detection)
        btn_morph = QPushButton("Operasi Morfologi (Opening)"); btn_morph.clicked.connect(self.apply_morphology)
        lay_edge.addWidget(btn_thresh); lay_edge.addWidget(QLabel("Metode Tepi:")); lay_edge.addWidget(self.combo_edge)
        lay_edge.addWidget(btn_edge); lay_edge.addWidget(btn_morph)
        grp_edge.setLayout(lay_edge); toolbar_layout.addWidget(grp_edge)

        # 6 & 7. Color & Segmentation
        grp_color = QGroupBox("6 & 7. Pemrosesan Warna & Segmentasi")
        lay_color = QVBoxLayout()
        btn_gray = QPushButton("RGB ke Grayscale"); btn_gray.clicked.connect(self.apply_grayscale)
        btn_split = QPushButton("Channel Splitting (Ekstrak Merah)"); btn_split.clicked.connect(self.apply_split)
        btn_seg = QPushButton("Segmentasi Berbasis Threshold"); btn_seg.clicked.connect(self.apply_segmentation)
        lay_color.addWidget(btn_gray); lay_color.addWidget(btn_split); lay_color.addWidget(btn_seg)
        grp_color.setLayout(lay_color); toolbar_layout.addWidget(grp_color)

        # 9 & 11. Analysis & AI (MURNI CNN)
        grp_ai = QGroupBox("9. Analisis & 11. Klasifikasi CNN")
        lay_ai = QVBoxLayout()
        btn_show_hist = QPushButton("Analisis Histogram (Before-After)"); btn_show_hist.clicked.connect(self.show_histogram)
        btn_cnn = QPushButton("Klasifikasi Objek (Murni CNN Keras)")
        btn_cnn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; border: none; padding: 10px; border-radius: 5px;")
        btn_cnn.clicked.connect(self.classify_image_cnn)
        lay_ai.addWidget(btn_show_hist); lay_ai.addWidget(btn_cnn)
        grp_ai.setLayout(lay_ai); toolbar_layout.addWidget(grp_ai)

        scroll_area.setWidget(toolbar_widget)
        main_layout.addWidget(scroll_area)

        # ================= KANAN: PREVIEW AREA =================
        preview_layout = QHBoxLayout()
        
        # Panel Before
        before_layout = QVBoxLayout()
        self.lbl_title_before = QLabel("<b>Before (Original)</b>", alignment=Qt.AlignCenter)
        self.lbl_title_before.setFont(QFont("Segoe UI", 12))
        before_layout.addWidget(self.lbl_title_before)
        
        self.lbl_before = QLabel("Silakan Load Image")
        self.lbl_before.setAlignment(Qt.AlignCenter)
        self.lbl_before.setStyleSheet("border: 2px dashed #4a4a4a; background-color: #252526; border-radius: 10px;")
        self.lbl_before.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) 
        before_layout.addWidget(self.lbl_before, stretch=1)
        preview_layout.addLayout(before_layout)

        # Panel After
        after_layout = QVBoxLayout()
        self.lbl_title_after = QLabel("<b>After (Edited)</b>", alignment=Qt.AlignCenter)
        self.lbl_title_after.setFont(QFont("Segoe UI", 12))
        after_layout.addWidget(self.lbl_title_after)
        
        self.lbl_after = QLabel("Area Pratinjau")
        self.lbl_after.setAlignment(Qt.AlignCenter)
        self.lbl_after.setStyleSheet("border: 2px dashed #4a4a4a; background-color: #252526; border-radius: 10px;")
        self.lbl_after.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) 
        after_layout.addWidget(self.lbl_after, stretch=1)
        preview_layout.addLayout(after_layout)

        main_layout.addLayout(preview_layout, stretch=1)

    # ================= KUMPULAN FUNGSI (LOGIKA SISTEM KLASIK) =================
    
    def load_image(self):
        file_filter = "Semua Gambar (*.png *.jpg *.jpeg *.bmp *.webp *.tif);;Semua File (*.*)"
        fname, _ = QFileDialog.getOpenFileName(self, 'Pilih Gambar', '', file_filter)
        if fname:
            self.statusBar.showMessage("Memuat gambar...")
            img = cv2.imread(fname)
            if img is None:
                QMessageBox.warning(self, "Error", "File tidak didukung atau korup!")
                self.statusBar.showMessage("Gagal memuat gambar.", 3000)
                return
            self.cv_image_original = img
            self.cv_image_current = self.cv_image_original.copy()
            self.sl_bright.setValue(0); self.sl_contrast.setValue(10)
            self.update_previews()
            self.statusBar.showMessage("Gambar berhasil dimuat.", 3000)

    def save_image(self):
        if self.cv_image_current is None: return
        fname, _ = QFileDialog.getSaveFileName(self, 'Simpan Gambar', '', "JPEG Files (*.jpg);;PNG Files (*.png)")
        if fname:
            self.statusBar.showMessage("Menyimpan gambar...")
            if fname.lower().endswith(('.jpg', '.jpeg')):
                cv2.imwrite(fname, self.cv_image_current, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            else:
                cv2.imwrite(fname, self.cv_image_current)
            QMessageBox.information(self, "Sukses", f"Gambar disimpan ke:\n{fname}")
            self.statusBar.showMessage("Gambar berhasil disimpan.", 3000)

    def reset_image(self):
        if self.cv_image_original is not None:
            self.cv_image_current = self.cv_image_original.copy()
            self.sl_bright.setValue(0); self.sl_contrast.setValue(10)
            self.update_previews()
            self.statusBar.showMessage("Gambar di-reset ke kondisi awal.", 3000)

    def apply_brightness_contrast(self):
        if self.cv_image_original is None: return
        bright = self.sl_bright.value()
        contrast = self.sl_contrast.value() / 10.0
        self.cv_image_current = cv2.convertScaleAbs(self.cv_image_original, alpha=contrast, beta=bright)
        self.update_previews()

    def apply_hist_eq(self):
        if self.cv_image_current is None: return
        if len(self.cv_image_current.shape) == 3:
            yuv = cv2.cvtColor(self.cv_image_current, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            self.cv_image_current = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            self.cv_image_current = cv2.equalizeHist(self.cv_image_current)
        self.update_previews()
        self.statusBar.showMessage("Histogram Equalization diterapkan.", 3000)

    def apply_sharpen(self):
        if self.cv_image_current is None: return
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.cv_image_current = cv2.filter2D(self.cv_image_current, -1, kernel)
        self.update_previews()
        self.statusBar.showMessage("Efek Sharpening diterapkan.", 3000)

    def apply_geometry(self, action):
        if self.cv_image_current is None: return
        if action == "rotate":
            self.cv_image_current = cv2.rotate(self.cv_image_current, cv2.ROTATE_90_CLOCKWISE)
        elif action == "flip":
            self.cv_image_current = cv2.flip(self.cv_image_current, 1) 
        elif action == "resize":
            h, w = self.cv_image_current.shape[:2]
            self.cv_image_current = cv2.resize(self.cv_image_current, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
        self.update_previews()
        self.statusBar.showMessage(f"Transformasi {action} selesai.", 3000)

    def apply_restoration(self, action):
        if self.cv_image_current is None: return
        if action == "gaussian":
            self.cv_image_current = cv2.GaussianBlur(self.cv_image_current, (11, 11), 0)
        elif action == "median":
            self.cv_image_current = cv2.medianBlur(self.cv_image_current, 5)
        self.update_previews()
        self.statusBar.showMessage(f"Filter {action} diterapkan.", 3000)

    def apply_thresholding(self):
        if self.cv_image_current is None: return
        gray = cv2.cvtColor(self.cv_image_current, cv2.COLOR_BGR2GRAY) if len(self.cv_image_current.shape) == 3 else self.cv_image_current
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.cv_image_current = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.update_previews()

    def apply_edge_detection(self):
        if self.cv_image_current is None: return
        gray = cv2.cvtColor(self.cv_image_current, cv2.COLOR_BGR2GRAY) if len(self.cv_image_current.shape) == 3 else self.cv_image_current
        method = self.combo_edge.currentText()
        edges = gray.copy()
        
        if method == "Canny": edges = cv2.Canny(gray, 100, 200)
        elif method == "Sobel":
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        elif method == "Prewitt":
            kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]); ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            edges = cv2.convertScaleAbs(cv2.filter2D(gray, -1, kx) + cv2.filter2D(gray, -1, ky))
        elif method == "Robert":
            kx = np.array([[1,0],[0,-1]]); ky = np.array([[0,1],[-1,0]])
            edges = cv2.convertScaleAbs(cv2.filter2D(gray, -1, kx) + cv2.filter2D(gray, -1, ky))
        elif method == "Laplacian": edges = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
        elif method == "LoG": edges = cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(gray, (3,3), 0), cv2.CV_64F))

        self.cv_image_current = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.update_previews()
        self.statusBar.showMessage(f"Deteksi Tepi {method} diterapkan.", 3000)

    def apply_morphology(self):
        if self.cv_image_current is None: return
        kernel = np.ones((5,5), np.uint8)
        self.cv_image_current = cv2.dilate(cv2.erode(self.cv_image_current, kernel, iterations=1), kernel, iterations=1)
        self.update_previews()

    def apply_grayscale(self):
        if self.cv_image_current is None: return
        if len(self.cv_image_current.shape) == 3:
            self.cv_image_current = cv2.cvtColor(cv2.cvtColor(self.cv_image_current, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            self.update_previews()

    def apply_split(self):
        if self.cv_image_current is None or len(self.cv_image_current.shape) != 3: return
        b, g, r = cv2.split(self.cv_image_current)
        zeros = np.zeros_like(b)
        self.cv_image_current = cv2.merge([zeros, zeros, r]) 
        self.update_previews()

    def apply_segmentation(self):
        if self.cv_image_current is None: return
        hsv = cv2.cvtColor(self.cv_image_current, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        self.cv_image_current = cv2.bitwise_and(self.cv_image_current, self.cv_image_current, mask=mask)
        self.update_previews()

    def show_histogram(self):
        if self.cv_image_original is None or self.cv_image_current is None: return
        self.statusBar.showMessage("Merender grafik histogram...")
        
        gray_ori = cv2.cvtColor(self.cv_image_original, cv2.COLOR_BGR2GRAY) if len(self.cv_image_original.shape)==3 else self.cv_image_original
        gray_cur = cv2.cvtColor(self.cv_image_current, cv2.COLOR_BGR2GRAY) if len(self.cv_image_current.shape)==3 else self.cv_image_current

        plt.figure("Analisis Histogram PNJ", figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.title("Before (Original)")
        plt.hist(gray_ori.ravel(), bins=256, range=[0,256], color='gray')
        plt.subplot(1, 2, 2); plt.title("After (Edited)")
        plt.hist(gray_cur.ravel(), bins=256, range=[0,256], color='blue')
        plt.tight_layout()
        plt.show()
        self.statusBar.showMessage("Grafik histogram berhasil ditampilkan.", 3000)

    # ================= LOGIKA MURNI CNN (TENSORFLOW KERAS) =================
    def classify_image_cnn(self):
        if self.cv_image_current is None: return
        if self.cnn_model is None:
            QMessageBox.warning(self, "Error", "Library Keras/TensorFlow tidak tersedia.")
            return

        self.statusBar.showMessage("Mengeksekusi Konvolusi Matriks CNN...")
        QApplication.processEvents() 

        # Pastikan gambar adalah RGB (3 channel) sebelum masuk ke model CNN
        img_rgb = cv2.cvtColor(self.cv_image_current, cv2.COLOR_BGR2RGB) if len(self.cv_image_current.shape) == 3 else cv2.cvtColor(self.cv_image_current, cv2.COLOR_GRAY2RGB)
        
        # 1. Resize menjadi matriks standar CNN
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # 2. Preprocessing & Normalisasi
        x = np.expand_dims(img_resized, axis=0)
        x = preprocess_input(x)
        
        # 3. Prediksi Klasifikasi
        preds = self.cnn_model.predict(x)
        results = decode_predictions(preds, top=3)[0]
        
        # 4. Tampilkan hasil probabilitas True Positive
        annotated_frame = self.cv_image_current.copy()
        
        # Buat kotak gelap agar teks bisa terbaca
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (420, 130), (20, 20, 20), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0)
        
        cv2.putText(annotated_frame, "Hasil Klasifikasi Murni CNN:", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 168, 255), 2)
        
        y_offset = 65
        for i, (imagenet_id, label, prob) in enumerate(results):
            text = f"{i+1}. {label.capitalize()} (True Pos: {prob*100:.2f}%)"
            cv2.putText(annotated_frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
            
        self.cv_image_current = annotated_frame
        self.update_previews()
        self.statusBar.showMessage("Konvolusi CNN Selesai.", 5000)

    # ================= RENDER & UPDATE PREVIEW =================
    def update_previews(self):
        if self.cv_image_original is not None:
            h_ori, w_ori = self.cv_image_original.shape[:2]
            self.lbl_title_before.setText(f"<b>Before (Original)</b><br><span style='color:#00a8ff; font-size: 14px;'>{w_ori} x {h_ori} px</span>")
            self.display_image(self.cv_image_original, self.lbl_before)
            
        if self.cv_image_current is not None:
            h_cur, w_cur = self.cv_image_current.shape[:2]
            self.lbl_title_after.setText(f"<b>After (Edited)</b><br><span style='color:#00a8ff; font-size: 14px;'>{w_cur} x {h_cur} px</span>")
            self.display_image(self.cv_image_current, self.lbl_after)

    def display_image(self, cv_img, label_widget):
        qformat = QImage.Format_Indexed8
        if len(cv_img.shape) == 3:
            qformat = QImage.Format_RGBA8888 if cv_img.shape[2] == 4 else QImage.Format_RGB888
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv_img

        out_image = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], qformat)
        pixmap = QPixmap.fromImage(out_image)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_previews()
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MiniPhotoshopPro()
    window.show()
    sys.exit(app.exec_())