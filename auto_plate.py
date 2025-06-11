#!/usr/bin/env python3
"""
🚗 Otomatik Plaka Tespit Sistemi - Semihocakli 2-Model Yaklaşımı
Kamera açıldığında plakayı otomatik seçer ve yazar
"""

import cv2
import numpy as np
from ultralytics import YOLO
import re
import sqlite3
from datetime import datetime
import os
import time
import signal
import sys

# PaddleOCR import
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️ PaddleOCR yüklü değil. EasyOCR fallback kullanılacak.")
    import easyocr

class AutoPlateDetector:
    def __init__(self):
        """Semihocakli 2-Model Plaka Tespit Sistemi"""
        print("🤖 Otomatik Plaka Tespit Sistemi Başlatılıyor...")
        print("🔧 Semihocakli 2-Model Yaklaşımı: Detection + OCR")
        
        # Model öncelik sırası - DETECTION modelleri (sadece plaka tespiti)
        detection_model_configs = [
            {
                'path': 'detection_weights/best.pt',  # Semihocakli Detection modeli
                'name': 'Semihocakli Detection Model',
                'type': 'detection_only',
                'precision': '99.8% precision Turkish plate detection'
            },
            {
                'path': 'best.pt',  # Root'ta best.pt
                'name': 'Best.pt Detection Model',
                'type': 'detection_only', 
                'precision': 'Best training detection'
            },
            {
                'path': 'last.pt',  # Son sırada last.pt 
                'name': 'Last.pt Detection Model',
                'type': 'detection_only',
                'precision': 'Latest training detection'
            },
            {
                'path': 'yolo11s.pt',  # YOLOv11 araç tespit
                'name': 'YOLOv11s Car Detection', 
                'type': 'car_detection',
                'precision': 'Standard car detection'
            }
        ]
        
        # Detection model yükleme
        self.detection_model = None
        self.use_roboflow = False  # Artık lokal kullanıyoruz
        
        for config in detection_model_configs:
            try:
                if os.path.exists(config['path']):
                    self.detection_model = YOLO(config['path'])
                    print(f"✅ {config['name']} yüklendi ({config['precision']})")
                    self.model_type = config['type']
                    
                    # Model dosya bilgileri
                    model_size = os.path.getsize(config['path']) / (1024*1024)
                    print(f"📊 Model boyutu: {model_size:.1f} MB")
                    
                    # Detection tipine göre ayarla
                    if config['type'] == 'detection_only':
                        self.direct_plate_detection = True
                        self.detection_class = 0  # class 0 = license_plate
                        print("🎯 PLAKA TESPİT MODU: Detection Model + OCR")
                    else:
                        self.direct_plate_detection = False 
                        self.detection_class = 2  # class 2 = car
                        print("🚗 ARAÇ TESPİT MODU: Car Detection + OCR")
                    
                    break
                else:
                    print(f"⚠️ {config['path']} dosyası bulunamadı")
                    
            except Exception as e:
                print(f"⚠️ {config['name']} yüklenemedi: {e}")
                continue
        
        if self.detection_model is None:
            raise Exception("Hiçbir detection model yüklenemedi")
        
        # OCR engine - PaddleOCR öncelikli (READING için)
        self.ocr_engine = None
        self.ocr_type = ''
        try:
            if PADDLE_AVAILABLE:
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
                self.ocr_type = 'paddle'
                print("✅ PaddleOCR motoru yüklendi (GPU)")
            else:
                self.ocr_engine = easyocr.Reader(['en'], gpu=True)
                self.ocr_type = 'easy'
                print("✅ EasyOCR motoru yüklendi (GPU)")
        except Exception as e_gpu:
            print(f"⚠️ GPU üzerinde OCR motoru başlatılamadı: {e_gpu}. CPU deniyor...")
            try:
                if PADDLE_AVAILABLE:
                    self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
                    self.ocr_type = 'paddle'
                    print("✅ PaddleOCR motoru yüklendi (CPU)")
                else:
                    self.ocr_engine = easyocr.Reader(['en'], gpu=False)
                    self.ocr_type = 'easy'
                    print("✅ EasyOCR motoru yüklendi (CPU)")
            except Exception as e_cpu:
                print(f"❌ Kritik Hata: OCR motoru CPU üzerinde de başlatılamadı: {e_cpu}")
                # Hata durumunda motor None olarak kalacak ve program run() içinde duracak
        
        if self.ocr_engine is None:
            print("❌ Kritik Hata: OCR motoru yüklenemedi. Program sonlandırılıyor.")
            return
        
        # Tracking için değişkenler - Semihocakli optimizasyonları
        self.tracked_cars = []
        self.tracked_plates = []
        self.smoothing_factor = 0.8  # Daha stabil tracking (Semihocakli önerisi)
        self.min_car_area = 3000  # Daha hassas tespit
        self.iou_threshold = 0.4  # Daha kesin tracking
        
        # ROI (Region of Interest) - Semihocakli tekniği
        self.use_roi = True
        self.roi_factor = 0.6  # Alt %60'da plaka ara
        
        # Türk plaka optimizasyonları
        self.turkish_plate_patterns = [
            r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$',  # 34 ABC 1234
            r'^[0-9]{2}[A-Z]{2}[0-9]{3}$',      # 34 AB 123
            r'^[0-9]{2}[A-Z]{3}[0-9]{2}$'       # 34 ABC 12
        ]
        
        # Performans monitoring - GitHub değerleri
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        self.fps_list = []
        self.processing_times = []
        self.memory_usages = []
        self.cpu_usages = []
        self.inference_times = []
        
        # PERFORMANS OPTİMİZASYONU - GitHub'dan alınan Semihocakli parametreleri
        self.frame_skip_count = 0
        self.frame_skip_interval = 1  # GitHub'da her frame işleniyor
        self.fast_mode = True  # Ultra hızlı mod
        
        # Semihocakli GitHub hedef değerleri
        self.target_inference_time = 3.0  # 3ms hedef (Semihocakli standardı)
        self.target_processing_time = 4.3  # 4.3ms hedef
        self.optimization_level = 'yolov8'  # GitHub'da YOLOv8 kullanıyor
        
        # GitHub'dan alınan performans izleme
        self.prev_frame_time = 0
        self.start_time = time.time()
        
        # GÖRSEL İYİLEŞTİRMELER
        self.detected_plates_list = []  # Yakalanan plakalar listesi
        self.last_detections = []  # Son tespitler için minimum süre
        self.min_display_time = 3.0  # Çerçevelerin minimum 3 saniye kalması
        
        # Veritabanı
        self.init_database()
        
        print("🚀 Sistem hazır! [LOKAL TÜRK PLAKA OPTIMIZED MODE]")
        print(f"📊 Model: {getattr(self, 'model_type', 'unknown')}")
        print(f"🎯 Doğrudan Plaka Tespit: {getattr(self, 'direct_plate_detection', False)}")
        print(f"⚡ Hedef Hız: {self.target_inference_time}ms çıkarım + {self.target_processing_time}ms işlem")
        print(f"🔍 ROI Aktif: {self.use_roi}")
        print(f"🔧 Optimizasyon: {self.optimization_level}")
        
        if getattr(self, 'direct_plate_detection', False):
            model_path = getattr(self, 'detection_model', None)
            if hasattr(model_path, 'ckpt_path'):
                model_file = os.path.basename(model_path.ckpt_path) if model_path.ckpt_path else "unknown"
            else:
                model_file = "custom_model"
            
            print(f"🎯 DOĞRUDAN PLAKA TESPİT: {model_file}")
            print("⚡ ULTRA HIZLI: Plakalar doğrudan tespit edilecek!")
            
            if 'last.pt' in str(model_file):
                print("⚡ SON EPOCH - En güncel eğitim ağırlıkları")
            elif 'best.pt' in str(model_file):
                print("🏆 EN İYİ - Validation'da en yüksek skor")
        else:
            print("🚗 Geleneksel Araç+OCR Yöntemi")
    
    def init_database(self):
        """Veritabanını başlat"""
        try:
            self.conn = sqlite3.connect('plates.db')
            cursor = self.conn.cursor()
            
            # Tablo oluştur (varsa kontrol et)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    UNIQUE(plate_text, timestamp)
                )
            ''')
            
            self.conn.commit()
            print("📊 Veritabanı hazır")
            
        except Exception as e:
            print(f"⚠️ Veritabanı hatası: {e}")
            # Fallback - bellek veritabanı
            try:
                self.conn = sqlite3.connect(':memory:')
                cursor = self.conn.cursor()
                cursor.execute('''
                    CREATE TABLE plates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_text TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        UNIQUE(plate_text, timestamp)
                    )
                ''')
                self.conn.commit()
                print("📊 Bellek veritabanı kullanılıyor")
            except Exception as e2:
                print(f"❌ Veritabanı başlatılamadı: {e2}")
                self.conn = None
    
    def calculate_iou(self, box1, box2):
        """İki kutu arasındaki IoU hesapla"""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        
        # Kesişim alanı
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Toplam alan
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def smooth_bbox(self, new_bbox, old_bbox, factor=0.7):
        """Çerçeve koordinatlarını yumuşat"""
        if old_bbox is None:
            return new_bbox
        
        smoothed = []
        for i in range(4):
            smoothed.append(int(old_bbox[i] * factor + new_bbox[i] * (1 - factor)))
        
        return smoothed
    
    def detect_cars_or_plates(self, frame):
        """Araç veya plaka tespit et - Model tipine göre"""
        # Model tipine göre confidence ayarla
        conf_threshold = 0.25 if getattr(self, 'model_type', 'general') in ['turkish_optimized', 'plate_detection'] else 0.3
        
        # DOĞRUDAN PLAKA TESPİT MODU
        if getattr(self, 'direct_plate_detection', False):
            return self.detect_plates_directly(frame, conf_threshold)
        
        # Normal araç tespiti
        return self.detect_cars_normal(frame, conf_threshold)
    
    def detect_plates_directly(self, frame, conf_threshold):
        """Doğrudan plaka tespit et - Semihocakli optimizasyonları ile"""
        # Confidence'ı daha hassas hale getir (daha fazla aday için)
        conf_threshold = 0.25
        
        # ROI uygula
        if self.use_roi:
            h, w = frame.shape[:2]
            roi_frame = frame[int(h*0.3):h, :]  # Alt %70'i kullan
            results = self.detection_model(roi_frame, conf=conf_threshold, classes=[0])  # class 0 = license_plate
        else:
            results = self.detection_model(frame, conf=conf_threshold, classes=[0])
        
        # Tüm tespitleri topla
        all_boxes = []
        all_scores = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # ROI koordinat düzeltmesi
                if self.use_roi:
                    h_frame = frame.shape[0]
                    y1 += int(h_frame * 0.3)
                    y2 += int(h_frame * 0.3)
                
                # Plaka boyut kontrolü
                w, h = x2 - x1, y2 - y1
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Plaka aspect ratio ve alan kontrolü (daha esnek hale getirildi)
                if area > 800 and 1.5 < aspect_ratio < 7.0 and conf > conf_threshold:
                    all_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    all_scores.append(float(conf))
        
        # NMS (Non-Maximum Suppression) uygula - ÇOKLU KUTU PROBLEMİNİ ÇÖZER
        detected_plates = []
        if len(all_boxes) > 0:
            # Convert to format needed for NMS
            boxes_for_nms = []
            for box in all_boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                boxes_for_nms.append([x1, y1, w, h])
            
            # Semihocakli parametreleri: score_threshold=0.5, nms_threshold=0.5
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms, 
                all_scores, 
                score_threshold=0.5,  # GitHub'dan alınan değer
                nms_threshold=0.5     # GitHub'dan alınan değer
            )
            
            # NMS sonrası filtrelenmiş kutular
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = all_boxes[i]
                    conf = all_scores[i]
                    
                    # Plaka bölgesini çıkar ve OCR uygula
                    plate_region = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    if plate_region.size > 0:
                        ocr_results = self.process_ocr(plate_region)
                        
                        for ocr_result in ocr_results:
                            if ocr_result['confidence'] > 0.3:
                                detected_plates.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'text': ocr_result['text'],
                                    'confidence': float(conf) * ocr_result['confidence'],
                                    'method': 'direct_plate_detection_nms',
                                    'formatted_text': self.format_plate_text(ocr_result['text'])
                                })
        
        return detected_plates
    
    def detect_cars_normal(self, frame, conf_threshold):
        """Normal araç tespiti"""
        # ROI uygula (Semihocakli tekniği)
        if self.use_roi:
            h, w = frame.shape[:2]
            roi_frame = frame[int(h*0.3):h, :]  # Alt %70'i kullan
            results = self.detection_model(roi_frame, conf=conf_threshold, classes=[2])  # class 2 = car
        else:
            results = self.detection_model(frame, conf=conf_threshold, classes=[2])
        
        new_cars = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # ROI koordinat düzeltmesi
                if self.use_roi:
                    h_frame = frame.shape[0]
                    y1 += int(h_frame * 0.3)
                    y2 += int(h_frame * 0.3)
                
                # Semihocakli boyut kontrolü (daha hassas)
                w, h = x2 - x1, y2 - y1
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Araç aspect ratio kontrolü (1.2-4.0 arası normal)
                if area > self.min_car_area and 1.0 < aspect_ratio < 5.0:
                    new_cars.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
        
        # Tracking uygula
        updated_cars = []
        for new_car in new_cars:
            best_match = None
            best_iou = 0
            
            # Mevcut tracked cars ile karşılaştır (None olanları geç)
            for i, tracked_car in enumerate(self.tracked_cars):
                if tracked_car is None:  # None değerleri atla
                    continue
                    
                iou = self.calculate_iou(new_car['bbox'], tracked_car['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                # Mevcut araç - koordinatları yumuşat
                old_bbox = self.tracked_cars[best_match]['bbox']
                smoothed_bbox = self.smooth_bbox(new_car['bbox'], old_bbox, self.smoothing_factor)
                
                updated_cars.append({
                    'bbox': smoothed_bbox,
                    'confidence': new_car['confidence'],
                    'area': new_car['area'],
                    'tracked': True,
                    'id': self.tracked_cars[best_match].get('id', len(updated_cars))
                })
                
                # İşlenmiş olanı kaldır
                self.tracked_cars[best_match] = None
            else:
                # Yeni araç
                updated_cars.append({
                    'bbox': new_car['bbox'],
                    'confidence': new_car['confidence'],
                    'area': new_car['area'],
                    'tracked': False,
                    'id': len(self.tracked_cars) + len(updated_cars)
                })
        
        # Güncel listeyi kaydet (None değerleri filtrele)
        self.tracked_cars = [car for car in updated_cars if car is not None]
        
        return self.tracked_cars
    
    def process_roboflow_results(self, roboflow_results, frame, roi_offset=0):
        """Roboflow API sonuçlarını işle"""
        detected_plates = []
        
        try:
            predictions = roboflow_results.get('predictions', [])
            
            for prediction in predictions:
                # Roboflow koordinatları al
                x = prediction.get('x', 0)
                y = prediction.get('y', 0) + roi_offset  # ROI offset ekle
                width = prediction.get('width', 0)
                height = prediction.get('height', 0)
                confidence = prediction.get('confidence', 0)
                
                # YOLO formatına çevir
                x1 = int(x - width/2)
                y1 = int(y - height/2)
                x2 = int(x + width/2)
                y2 = int(y + height/2)
                
                # Çerçeve sınırları kontrolü
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0, min(x1, frame_w))
                y1 = max(0, min(y1, frame_h))
                x2 = max(0, min(x2, frame_w))
                y2 = max(0, min(y2, frame_h))
                
                if confidence > 0.4:  # Minimum confidence
                    # Plaka bölgesini çıkar ve OCR uygula
                    plate_region = frame[y1:y2, x1:x2]
                    
                    if plate_region.size > 0:
                        # OCR ile plaka metnini oku
                        ocr_results = self.process_ocr(plate_region)
                        
                        for ocr_result in ocr_results:
                            if ocr_result['confidence'] > 0.3:
                                detected_plates.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'text': ocr_result['text'],
                                    'confidence': confidence * ocr_result['confidence'],
                                    'method': 'roboflow_ocr',
                                    'formatted_text': self.format_plate_text(ocr_result['text'])
                                })
        
        except Exception as e:
            print(f"Roboflow sonuç işleme hatası: {e}")
        
        return detected_plates
    
    def extract_plate_region(self, frame, car_bbox):
        """Araba içinden plaka bölgesini çıkar - ultra hassas"""
        x1, y1, x2, y2 = car_bbox
        car_img = frame[y1:y2, x1:x2]
        
        # Plaka bölgesini daha hassas belirle
        h, w = car_img.shape[:2]
        
        # Alt %25-90 arası (daha dar ve hassas)
        plate_y_start = int(h * 0.25)
        plate_y_end = int(h * 0.90)
        
        # Yanlardan biraz kırp (plakanın tam ortasını al)
        plate_x_start = int(w * 0.15)
        plate_x_end = int(w * 0.85)
        
        plate_region = car_img[plate_y_start:plate_y_end, plate_x_start:plate_x_end]
        
        # Global koordinatlara çevir
        global_plate_bbox = [
            x1 + plate_x_start,
            y1 + plate_y_start,
            x1 + plate_x_end,
            y1 + plate_y_end
        ]
        
        return plate_region, global_plate_bbox
    
    def enhance_plate_image(self, image):
        """Plaka görüntüsünü OCR için optimize et"""
        if image.size == 0:
            return image
            
        # Görüntüyü büyüt (OCR daha iyi çalışır)
        height, width = image.shape[:2]
        if height < 50:  # Çok küçükse büyüt
            scale = 50 / height
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        # Tekrar büyüt (OCR için ideal boyut)
        image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def detect_license_plate(self, frame, car_bbox):
        """Daha hassas plaka tespiti - aggressive mode"""
        # Araç bölgesini al
        x1, y1, x2, y2 = car_bbox
        car_region = frame[y1:y2, x1:x2]
        
        # Alt %70'de plaka ara - çok geniş bölge
        h, w = car_region.shape[:2]
        lower_region = car_region[int(h*0.3):h, :]  # 0.4'ten 0.3'e düşürdük
        
        try:
            # Çoklu bölge tarama
            plate_boxes = []
            
            # 1. Alt bölge
            results1 = self.ocr_engine.readtext(lower_region)
            for (bbox, text, conf) in results1:
                if conf > 0.2 and len(text) >= 4:  # Düşük threshold
                    min_x = min([p[0] for p in bbox])
                    min_y = min([p[1] for p in bbox])
                    max_x = max([p[0] for p in bbox])
                    max_y = max([p[1] for p in bbox])
                    
                    global_min_x = x1 + min_x
                    global_min_y = y1 + int(h*0.3) + min_y
                    global_max_x = x1 + max_x
                    global_max_y = y1 + int(h*0.3) + max_y
                    
                    plate_boxes.append({
                        'bbox': [int(global_min_x), int(global_min_y), 
                                int(global_max_x), int(global_max_y)],
                        'text': self.clean_plate_text(text),
                        'confidence': conf
                    })
            
            # 2. Tüm araç bölgesi (backup)
            results2 = self.ocr_engine.readtext(car_region)
            for (bbox, text, conf) in results2:
                if conf > 0.15 and len(text) >= 4:  # Çok düşük threshold
                    min_x = min([p[0] for p in bbox])
                    min_y = min([p[1] for p in bbox])
                    max_x = max([p[0] for p in bbox])
                    max_y = max([p[1] for p in bbox])
                    
                    global_min_x = x1 + min_x
                    global_min_y = y1 + min_y
                    global_max_x = x1 + max_x
                    global_max_y = y1 + max_y
                    
                    plate_boxes.append({
                        'bbox': [int(global_min_x), int(global_min_y), 
                                int(global_max_x), int(global_max_y)],
                        'text': self.clean_plate_text(text),
                        'confidence': conf
                    })
            
            return plate_boxes
            
        except Exception as e:
            print(f"Plaka tespit hatası: {e}")
            return []
    
    def call_ocr(self, image, **kwargs):
        """OCR motorunu çağır - NoneType çökmesine karşı SÜPER robust hale getirildi"""
        if image is None or image.size == 0:
            return []
        try:
            if self.ocr_type == 'paddle':
                # PaddleOCR can return None, [None], or [[...]]
                result = self.ocr_engine.ocr(image, cls=True)
                
                # Step 1: Handle top-level None or empty list
                if not result:
                    return []

                # Step 2: Unwrap the typical extra list layer from PaddleOCR
                processed_result = result[0]
                
                # Step 3: Handle unwrapped list being None or empty
                if not processed_result:
                    return []

                # Step 4: Clean the final list of any internal None values
                final_results = [item for item in processed_result if item is not None]
                
                return final_results

            elif self.ocr_type == 'easy':
                result = self.ocr_engine.readtext(image, **kwargs)
                return result if result is not None else []
            
        except Exception as e:
            # print(f"OCR Çağrı Hatası: {e}") # Konsolu boğmamak için kapalı
            return []
    
    def process_ocr(self, image):
        """OCR ile plaka metnini oku - KARAKTER BİRLEŞTİRME VE ÇOKLU TEKNİK"""
        try:
            enhanced_image = self.enhance_plate_image(image)
            
            if len(enhanced_image.shape) == 3:
                gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = enhanced_image
            
            ocr_pipeline_results = []
            
            # 1. Orijinal Görüntü
            ocr_pipeline_results.extend(self.call_ocr(gray))
            # 2. CLAHE ile Kontrast Artırma
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            ocr_pipeline_results.extend(self.call_ocr(enhanced))
            
            if not self.fast_mode:
                # Daha fazla teknik (Normal modda)
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                ocr_pipeline_results.extend(self.call_ocr(adaptive))
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ocr_pipeline_results.extend(self.call_ocr(otsu))

            if not ocr_pipeline_results:
                return []

            # KARAKTER BİRLEŞTİRME MANTIĞI BURADA
            # Sonuçları x-koordinatına göre sırala
            # EasyOCR: (bbox, text, conf) -> bbox[0][0]
            # PaddleOCR: ([bbox], (text, conf)) -> bbox[0][0]
            try:
                if self.ocr_type == 'easy':
                    ocr_pipeline_results.sort(key=lambda res: res[0][0][0])
                elif self.ocr_type == 'paddle':
                     # Paddle'ın garip formatını düzelt
                    flat_results = []
                    for item in ocr_pipeline_results:
                        if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
                             flat_results.extend(item)
                        else:
                             flat_results.append(item)
                    ocr_pipeline_results = flat_results
                    ocr_pipeline_results.sort(key=lambda res: res[0][0][0])
            except Exception as sort_e:
                print(f"OCR sıralama hatası: {sort_e}") # Sıralama başarısız olursa devam et
            
            # Birleştirilmiş metinleri ve bireysel metinleri topla
            text_candidates = {}

            # 1. Bireysel sonuçları işle
            for res in ocr_pipeline_results:
                text = res[1][0] if self.ocr_type == 'paddle' else res[1]
                conf = res[1][1] if self.ocr_type == 'paddle' else res[2]

                if conf > 0.3:
                    clean_text = self.clean_plate_text(text)
                    if self.validate_turkish_plate_format(clean_text):
                        if clean_text not in text_candidates or conf > text_candidates[clean_text]:
                            text_candidates[clean_text] = conf
            
            # 2. Birleştirilmiş sonucu işle
            if len(ocr_pipeline_results) > 1:
                full_text_list = []
                total_confidence = 0
                item_count = 0

                for res in ocr_pipeline_results:
                    text = res[1][0] if self.ocr_type == 'paddle' else res[1]
                    conf = res[1][1] if self.ocr_type == 'paddle' else res[2]
                    full_text_list.append(self.clean_plate_text(text))
                    total_confidence += conf
                    item_count += 1
                
                full_text = "".join(full_text_list)
                avg_confidence = total_confidence / item_count if item_count > 0 else 0
                
                clean_full_text = self.clean_plate_text(full_text)
                if self.validate_turkish_plate_format(clean_full_text):
                     if clean_full_text not in text_candidates or avg_confidence > text_candidates[clean_full_text]:
                            text_candidates[clean_full_text] = avg_confidence

            if not text_candidates:
                return []

            # En yüksek güven skoruna sahip adayı seç
            best_text = max(text_candidates, key=text_candidates.get)
            best_conf = text_candidates[best_text]
            
            # Dönecek format: [{'text': ..., 'confidence': ...}]
            return [{'text': best_text, 'confidence': best_conf}]

        except Exception as e:
            print(f"OCR işleme hatası: {e}")
            return []
        
    def correct_ocr_mistakes(self, text):
        """OCR hatalarını düzelt (yaygın harf karışıklıkları)"""
        # Yaygın OCR hataları
        corrections = {
            # Harf karışıklıkları
            'MLI': 'TIH',  # M->T, L->I, I->H yaygın hatalar
            'MH': 'TH',
            'ML': 'TI', 
            'LI': 'TI',
            'MIH': 'TIH',
            'TLH': 'TIH',
            'TLI': 'TIH',
            'TIM': 'TIH',
            'THI': 'TIH',
            
            # Rakam karışıklıkları
            'O': '0',
            'I': '1',
            'S': '5',
            'B': '8',
            'G': '6',
            
            # Yaygın karışıklıklar
            '0MLI': '06TIH',
            '0MH': '06TH',
            '06MLI': '06TIH',
            '06MH': '06TH',
            '06ML': '06TI',
        }
        
        # Önce tam metin kontrolü
        if text in corrections:
            return corrections[text]
        
        # Kısmi düzeltmeler
        corrected = text
        for mistake, correction in corrections.items():
            corrected = corrected.replace(mistake, correction)
        
        return corrected
    
    def clean_plate_text(self, text):
        """Türk plaka metni temizleme - Semihocakli optimizasyonu ile"""
        # Sadece harf ve rakam
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Çok kısa metinleri atla
        if len(text) < 4:
            return None
        
        # OCR hatalarını düzelt
        corrected_text = self.correct_ocr_mistakes(text)
        
        # Semihocakli Türk plaka pattern kontrolü (daha katı)
        for pattern in self.turkish_plate_patterns:
            if re.match(pattern, corrected_text):
                # Türk plaka format kontrolü
                if self.validate_turkish_plate_format(corrected_text):
                    return corrected_text
        
        # Kısmi plaka formatları (daha esnek)
        partial_patterns = [
            r'^[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$',
            r'^[A-Z]{1,3}[0-9]{1,4}$'
        ]
        
        for pattern in partial_patterns:
            if re.match(pattern, corrected_text) and len(corrected_text) >= 5:
                return corrected_text
        
        return None
    
    def validate_turkish_plate_format(self, text):
        """Türk plaka formatını doğrula - Semihocakli kuralları"""
        if len(text) < 5 or len(text) > 8:
            return False
        
        # Yasak harf kombinasyonları (Türkiye)
        forbidden_letters = ['Q', 'W', 'X']
        for letter in forbidden_letters:
            if letter in text:
                return False
        
        # 01 ile başlayan plakalar yasak
        if text.startswith('01'):
            return False
            
        return True
    
    def save_to_database(self, plate_text, confidence):
        """Plakayı veritabanına kaydet"""
        try:
            if not self.conn:
                return
                
            cursor = self.conn.cursor()
            
            # Son 10 saniye içinde aynı plaka var mı kontrol et
            cursor.execute('''
                SELECT COUNT(*) FROM plates 
                WHERE plate_text = ? AND 
                datetime(timestamp) > datetime('now', '-10 seconds')
            ''', (plate_text,))
            
            if cursor.fetchone()[0] == 0:  # Yeni tespit
                cursor.execute('''
                    INSERT INTO plates (plate_text, confidence, timestamp)
                    VALUES (?, ?, datetime('now'))
                ''', (plate_text, confidence))
                self.conn.commit()
                
                # Yakalanan plakalar listesine ekle
                detection_info = {
                    'plate': plate_text,
                    'confidence': confidence,
                    'time': time.time(),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                self.detected_plates_list.append(detection_info)
                
                # Listede sadece son 10 tespit kalsın
                if len(self.detected_plates_list) > 10:
                    self.detected_plates_list.pop(0)
                
                print(f"💾 Yeni plaka kaydedildi: {plate_text} (Conf: {confidence:.2f})")
        
        except Exception as e:
            print(f"Veritabanı kayıt hatası: {e}")
    
    def draw_results(self, frame, cars, detected_plates):
        """Sonuçları çiz - hem araç hem plaka çerçeveleri"""
        
        # Araç çerçeveleri
        for car in cars:
            x1, y1, x2, y2 = car['bbox']
            color = (0, 255, 0) if car.get('tracked', False) else (0, 165, 255)  # Yeşil=tracked, Turuncu=yeni
            thickness = 3 if car.get('tracked', False) else 2
            
            # Araç çerçevesi
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Araç bilgisi
            label = f"Araba #{car.get('id', '?')}: {car['confidence']:.2f}"
            if car.get('tracked', False):
                label += " [TRACKED]"
                
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Plaka çerçeveleri ve metinleri
        for plate_info in detected_plates:
            if 'bbox' in plate_info:
                px1, py1, px2, py2 = plate_info['bbox']
                
                # Plaka çerçevesi (kırmızı)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                
                # Formatlanmış plaka metni göster
                display_text = plate_info.get('formatted_text', plate_info['text'])
                text_label = f"PLAKA: {display_text}"
                cv2.putText(frame, text_label, (px1, py1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Güven skoru ve metod
                conf_label = f"Guven: {plate_info['confidence']:.2f}"
                cv2.putText(frame, conf_label, (px1, py2+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Metod bilgisi
                method_label = f"[{plate_info.get('method', 'unknown')}]"
                cv2.putText(frame, method_label, (px1, py2+45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Genel plaka listesi (sol üst köşe) - formatlanmış metinlerle
        y_offset = 30
        unique_plates = set()
        for plate_info in detected_plates:
            display_text = plate_info.get('formatted_text', plate_info['text'])
            if display_text not in unique_plates:
                unique_plates.add(display_text)
                cv2.putText(frame, f"✓ {display_text}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                y_offset += 30
        
        return frame
    
    def calculate_similarity(self, text1, text2):
        """İki plaka metni arasındaki benzerlik hesapla (Levenshtein distance)"""
        if len(text1) == 0:
            return len(text2)
        if len(text2) == 0:
            return len(text1)
        
        matrix = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        
        for i in range(len(text1) + 1):
            matrix[i][0] = i
        for j in range(len(text2) + 1):
            matrix[0][j] = j
            
        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                if text1[i-1] == text2[j-1]:
                    cost = 0
                else:
                    cost = 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )
        
        distance = matrix[len(text1)][len(text2)]
        similarity = 1 - (distance / max(len(text1), len(text2)))
        return similarity
    
    def merge_similar_plates(self, plate_results):
        """Benzer plaka tespitlerini birleştir ve en iyisini seç"""
        if not plate_results:
            return []
            
        # Grupları oluştur
        groups = []
        
        for plate_info in plate_results:
            plate_text = plate_info['text']
            added_to_group = False
            
            # Mevcut gruplarla karşılaştır
            for group in groups:
                for existing_plate in group:
                    similarity = self.calculate_similarity(plate_text, existing_plate['text'])
                    
                    # %80+ benzerlik varsa aynı gruba ekle
                    if similarity >= 0.8:
                        group.append(plate_info)
                        added_to_group = True
                        break
                
                if added_to_group:
                    break
            
            # Hiçbir gruba eklenmediyse yeni grup oluştur
            if not added_to_group:
                groups.append([plate_info])
        
        # Her gruptan en yüksek confidence'ı seç
        best_plates = []
        for group in groups:
            # En yüksek confidence'a sahip olanı seç
            best_plate = max(group, key=lambda x: x['confidence'])
            
            # En uzun ve en doğru formatı tercih et
            for plate in group:
                if (len(plate['text']) > len(best_plate['text']) and 
                    plate['confidence'] > best_plate['confidence'] * 0.8):
                    best_plate = plate
            
            best_plates.append(best_plate)
        
        return best_plates
    
    def format_plate_text(self, text):
        """Plaka metnini Türk formatına çevir (boşluklar ekle)"""
        # Sadece harf ve rakam
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Türk plaka formatları
        if len(clean) >= 7:
            # 34 ABC 1234 formatı
            if re.match(r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$', clean):
                if len(clean) == 8:  # 34ABC123
                    return f"{clean[:2]} {clean[2:5]} {clean[5:]}"
                elif len(clean) == 9:  # 34ABC1234
                    return f"{clean[:2]} {clean[2:5]} {clean[5:]}"
                elif len(clean) == 7:  # 34AB123
                    return f"{clean[:2]} {clean[2:4]} {clean[4:]}"
        
        return clean
    
    def detect_plate_in_region(self, car_region):
        """Araç bölgesinde plaka ara - iki farklı yöntem"""
        try:
            all_plates = []
            
            # Alt bölge (plakanın genelde burada olması)
            h, w = car_region.shape[:2]
            lower_third = car_region[int(h*0.25):int(h*0.9), int(w*0.15):int(w*0.85)]  # Alt %65
            
            # 1. Alt bölge
            results1 = self.call_ocr(lower_third)
            for (bbox, text, conf) in results1:
                if conf > 0.2 and len(text) >= 4:  # Düşük threshold
                    clean_text = self.clean_plate_text(text)
                    if clean_text:
                        all_plates.append((clean_text, conf, "lower"))
            
            # 2. Tüm araç bölgesi (backup)
            results2 = self.call_ocr(car_region)
            for (bbox, text, conf) in results2:
                if conf > 0.15 and len(text) >= 4:  # Çok düşük threshold
                    clean_text = self.clean_plate_text(text)
                    if clean_text:
                        all_plates.append((clean_text, conf, "full"))
            
            # En yüksek confidence'ı döndür
            if all_plates:
                best_plate = max(all_plates, key=lambda x: x[1])
                return best_plate[0], best_plate[1]
            
            return None, 0
            
        except Exception as e:
            print(f"Region detection hatası: {e}")
            return None, 0
    
    def get_unique_plates(self):
        """Unique plaka sayısını döndür"""
        try:
            cursor = self.conn.cursor()
            # Önce tablo var mı kontrol et
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plates'")
            if not cursor.fetchone():
                return 0
            
            cursor.execute("SELECT COUNT(DISTINCT plate_text) FROM plates")
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            #print(f"Unique plates hatası: {e}")  # Sessiz hata
            return 0

    def update_tracking(self, new_cars):
        """Araç tracking sistemini güncelle"""
        try:
            if not self.tracked_cars:
                # İlk araçları ekle
                for i, car in enumerate(new_cars):
                    car['id'] = i + 1
                    car['tracked'] = False
                    car['last_seen'] = time.time()
                self.tracked_cars = new_cars
                return
            
            # Mevcut araçları güncelle
            updated_cars = []
            used_new_cars = set()
            
            for tracked_car in self.tracked_cars:
                best_match = None
                best_iou = 0
                best_idx = -1
                
                # En iyi eşleşmeyi bul
                for i, new_car in enumerate(new_cars):
                    if i in used_new_cars:
                        continue
                    
                    iou = self.calculate_iou(tracked_car['bbox'], new_car['bbox'])
                    if iou > best_iou and iou > self.iou_threshold:
                        best_match = new_car
                        best_iou = iou
                        best_idx = i
                
                if best_match:
                    # Koordinatları yumuşat
                    smoothed_bbox = self.smooth_bbox(best_match['bbox'], tracked_car['bbox'], self.smoothing_factor)
                    
                    updated_car = {
                        'id': tracked_car['id'],
                        'bbox': smoothed_bbox,
                        'confidence': best_match['confidence'],
                        'area': best_match['area'],
                        'tracked': True,
                        'last_seen': time.time()
                    }
                    updated_cars.append(updated_car)
                    used_new_cars.add(best_idx)
                else:
                    # Araç kayboldu, 2 saniye daha bekle
                    if time.time() - tracked_car['last_seen'] < 2.0:
                        updated_cars.append(tracked_car)
            
            # Yeni araçları ekle
            max_id = max([car['id'] for car in self.tracked_cars]) if self.tracked_cars else 0
            for i, new_car in enumerate(new_cars):
                if i not in used_new_cars:
                    new_car['id'] = max_id + 1
                    new_car['tracked'] = False
                    new_car['last_seen'] = time.time()
                    updated_cars.append(new_car)
                    max_id += 1
            
            self.tracked_cars = updated_cars
            
        except Exception as e:
            print(f"Tracking hatası: {e}")
    
    def add_detection(self, car_bbox, plate_text, confidence):
        """Yeni tespit ekle (minimum süre ile)"""
        detection = {
            'car_bbox': car_bbox,
            'plate_text': plate_text,
            'confidence': confidence,
            'start_time': time.time(),
            'formatted_text': self.format_plate_text(plate_text)
        }
        self.last_detections.append(detection)
    
    def update_detections(self):
        """Eski tespitleri temizle"""
        current_time = time.time()
        self.last_detections = [
            det for det in self.last_detections 
            if current_time - det['start_time'] < self.min_display_time
        ]
    
    def draw_persistent_detections(self, frame):
        """Kalıcı tespitleri çiz"""
        for detection in self.last_detections:
            x1, y1, x2, y2 = detection['car_bbox']
            
            # Araç çerçevesi (mavi - tespit edilmiş)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Plaka bölgesi
            plate_y1 = y1 + int((y2-y1) * 0.6)
            cv2.rectangle(frame, (x1, plate_y1), (x2, y2), (0, 0, 255), 2)
            
            # Plaka metni
            cv2.putText(frame, detection['formatted_text'], (x1, y2+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Conf: {detection['confidence']:.2f}", (x1, y2+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Kalan süre
            remaining_time = self.min_display_time - (time.time() - detection['start_time'])
            cv2.putText(frame, f"({remaining_time:.1f}s)", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    def draw_detected_plates_list(self, frame):
        """Yakalanan plakaları ekranda listele"""
        if not self.detected_plates_list:
            return
        
        # Sağ üst köşe
        frame_height, frame_width = frame.shape[:2]
        start_x = frame_width - 250
        start_y = 30
        
        # Başlık
        cv2.putText(frame, "YAKALANAN PLAKALAR:", (start_x, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Son 5 plakayı göster
        recent_plates = self.detected_plates_list[-5:]
        for i, detection in enumerate(recent_plates):
            y_pos = start_y + 25 + (i * 20)
            text = f"{detection['timestamp']} - {detection['plate']}"
            confidence = f"({detection['confidence']:.2f})"
            
            cv2.putText(frame, text, (start_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, confidence, (start_x + 150, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    def run(self):
        """Ana çalışma fonksiyonu"""
        if not self.ocr_engine:
            print("❌ Kritik Hata: OCR motoru yüklenemediği için program çalıştırılamıyor.")
            return

        # Kamera başlat
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Kamera açılamadı!")
            return
        
        # PERFORMANS OPTİMİZASYONU: Düşük çözünürlük
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 640->320 (daha hızlı)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 480->240 (daha hızlı)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📹 Kamera açıldı - ULTRA FAST MODE")
        print("🎮 Kontroller: Q=Çıkış, S=Ekran görüntüsü, R=Reset, F=Fast/Normal Mode")
        print("💡 GUI hata durumunda otomatik headless mode'a geçer")
        
        frame_count = 0
        last_detection_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # GitHub performans izleme başlangıcı
            frame_start_time = time.time()
            
            # Frame sayacı
            frame_count += 1
            self.fps_frame_count += 1
            
            # FPS hesapla - GitHub yöntemi
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
            self.prev_frame_time = new_frame_time
            self.fps_list.append(fps)
            self.current_fps = fps
            
            # İşlem süresi başlangıcı
            inference_start_time = time.time()
            
            # PERFORMANCE BOOST: Frame atlama
            self.frame_skip_count += 1
            if self.frame_skip_count < self.frame_skip_interval:
                # SKIP FRAME - Sadece görüntüyü göster ama TÜM çerçeveleri çiz
                
                # Normal tracking çerçevelerini çiz
                for i, car in enumerate(self.tracked_cars):
                    x1, y1, x2, y2 = car['bbox']
                    color = (0, 255, 0) if car.get('tracked', False) else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Car-{car['id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Kalıcı tespitleri güncelle ve çiz
                self.update_detections()
                self.draw_persistent_detections(frame)
                self.draw_detected_plates_list(frame)
                
                info_text = f"[FAST MODE] Frame: {frame_count} | FPS: {self.current_fps:.1f} | Cars: {len(self.tracked_cars)} | Plates: {self.get_unique_plates()}"
                cv2.putText(frame, info_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Frame Skip: {self.frame_skip_count}/{self.frame_skip_interval}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # OpenCV GUI desteği kontrolü
                try:
                    cv2.imshow('Plaka Tespit Sistemi', frame)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as e:
                    if "is not implemented" in str(e) or "cvShowImage" in str(e):
                        # GUI desteği yok - headless modda çalış
                        if frame_count == 1:  # İlk frame'de uyarı
                            print("⚠️ GUI desteği bulunamadı, headless modda çalışıyor...")
                            print("🔥 HEADLESS MODE: Ctrl+C ile durdurun")
                        if frame_count % 30 == 0:  # Her 30 frame'de bir mesaj
                            print(f"📊 Headless Mode - Frame: {frame_count}, FPS: {self.current_fps:.1f}, Plates: {self.get_unique_plates()}")
                        key = 0xFF  # GUI olmadan devam et
                    else:
                        raise e
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    self.fast_mode = not self.fast_mode
                    mode_text = "FAST" if self.fast_mode else "NORMAL"
                    print(f"🔄 Mod değiştirildi: {mode_text}")
                elif key == ord('r'):
                    self.tracked_cars = []
                    self.tracked_plates = []
                    print("🔄 Tracking sistemi sıfırlandı")
                
                continue
            
            # Frame atlama sayacını sıfırla
            self.frame_skip_count = 0
            
            # Model tipine göre tespit yap
            current_time = time.time()
            
            # DOĞRUDAN PLAKA TESPİT MODU (Güler Kandeger modeli)
            if getattr(self, 'direct_plate_detection', False):
                detected_plates = self.detect_cars_or_plates(frame)
                
                # Tespit edilen plakalar için kalıcı gösterim
                for plate_info in detected_plates:
                    if plate_info['confidence'] > 0.5:
                        # Plaka etrafında mock araç bbox'ı oluştur
                        px1, py1, px2, py2 = plate_info['bbox']
                        car_bbox = [px1-50, py1-30, px2+50, py2+30]
                        self.add_detection(car_bbox, plate_info['text'], plate_info['confidence'])
                        self.save_to_database(plate_info['text'], plate_info['confidence'])
                
                # Kalıcı tespitleri çiz
                self.update_detections()
                self.draw_persistent_detections(frame)
                self.draw_detected_plates_list(frame)
                
            else:
                # GELENEKSEL ARAÇ+OCR YÖNTEMİ
                cars = self.detect_cars_or_plates(frame)
                self.update_tracking(cars)
                
                # Plaka tespiti (sadece yüksek confidence araçlarda)
                for i, car in enumerate(self.tracked_cars):
                    x1, y1, x2, y2 = car['bbox']
                    
                    # Araç çerçevesi çiz (normal tracking)
                    color = (0, 255, 0) if car.get('tracked', False) else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Car-{car.get('id', i)}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Plaka tespiti
                    if car['confidence'] > 0.5:
                        car_region = frame[y1:y2, x1:x2]
                        
                        if car_region.size > 0:
                            # Hızlı plaka tespiti (process_ocr çağırır)
                            ocr_results = self.process_ocr(car_region)
                            
                            for ocr_result in ocr_results:
                                if ocr_result['confidence'] > 0.4:
                                    self.add_detection([x1, y1, x2, y2], ocr_result['text'], ocr_result['confidence'])
                                    
                                    if ocr_result['confidence'] > 0.5:
                                        self.save_to_database(ocr_result['text'], ocr_result['confidence'])
                
                # Kalıcı tespitleri çiz
                self.draw_persistent_detections(frame)
                self.draw_detected_plates_list(frame)
            
            # Kalıcı tespitleri çiz (mavi çerçeveler) - artık her frame'de yapılıyor
            self.draw_persistent_detections(frame)
            
            # Yakalanan plakalar listesini çiz - artık her frame'de yapılıyor  
            self.draw_detected_plates_list(frame)
            
            # GitHub performans ölçümleri
            processing_time = time.time() - frame_start_time
            inference_time = time.time() - inference_start_time if 'inference_start_time' in locals() else 0
            self.processing_times.append(processing_time)
            self.inference_times.append(inference_time)
            
            # Bellek ve CPU ölçümü (performans için daha seyrek)
            if frame_count % 10 == 0:  # Her 10 frame'de bir
                try:
                    import psutil
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    cpu_usage = psutil.cpu_percent()
                    self.memory_usages.append(memory_usage)
                    self.cpu_usages.append(cpu_usage)
                except ImportError:
                    memory_usage = 0
                    cpu_usage = 0
            else:
                memory_usage = self.memory_usages[-1] if self.memory_usages else 0
                cpu_usage = self.cpu_usages[-1] if self.cpu_usages else 0
            
            # İstatistikleri göster - GitHub formatında
            mode_text = "[SEMIHOCAKLI MODE]" if self.fast_mode else "[NORMAL MODE]"
            info_text = f"{mode_text} Frame: {frame_count} | FPS: {self.current_fps:.2f}"
            cv2.putText(frame, info_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # GitHub performans metrikleri
            perf_text = f"Processing: {processing_time*1000:.2f}ms | Memory: {memory_usage:.1f}MB | CPU: {cpu_usage:.1f}%"
            cv2.putText(frame, perf_text, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Plaka sayısı
            plate_text = f"Cars: {len(self.tracked_cars)} | Plates: {self.get_unique_plates()}"
            cv2.putText(frame, plate_text, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Ekranı göster - GUI desteği kontrolü ile
            try:
                cv2.imshow('Plaka Tespit Sistemi', frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as e:
                if "is not implemented" in str(e) or "cvShowImage" in str(e):
                    # GUI desteği yok - headless modda çalış
                    if frame_count == 1:  # İlk frame'de uyarı
                        print("⚠️ GUI desteği bulunamadı, headless modda çalışıyor...")
                        print("🔥 HEADLESS MODE: Ctrl+C ile durdurun")
                    if frame_count % 30 == 0:  # Her 30 frame'de bir mesaj
                        print(f"📊 Headless Mode - Frame: {frame_count}, FPS: {self.current_fps:.1f}, Plates: {self.get_unique_plates()}")
                    key = 0xFF  # GUI olmadan devam et
                else:
                    raise e
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'screenshot_{int(time.time())}.jpg', frame)
                print("📸 Ekran görüntüsü kaydedildi")
            elif key == ord('f'):
                self.fast_mode = not self.fast_mode
                mode_text = "FAST" if self.fast_mode else "NORMAL"
                print(f"🔄 Mod değiştirildi: {mode_text}")
            elif key == ord('r'):
                self.tracked_cars = []
                self.tracked_plates = []
                print("🔄 Tracking sistemi sıfırlandı")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # GitHub performans özeti
        total_time = time.time() - self.start_time
        print(f"\n📊 SEMIHOCAKLI PERFORMANS ÖZETİ:")
        print(f"Toplam çalışma süresi: {total_time:.2f} saniye")
        print(f"Toplam işlenen kare sayısı: {frame_count}")
        
        if self.fps_list:
            import numpy as np
            print(f"Ortalama FPS: {np.mean(self.fps_list):.2f}")
            print(f"Minimum FPS: {np.min(self.fps_list):.2f}")
            print(f"Maksimum FPS: {np.max(self.fps_list):.2f}")
        
        if self.processing_times:
            print(f"Ortalama işlem süresi: {np.mean(self.processing_times)*1000:.2f} ms")
            print(f"Minimum işlem süresi: {np.min(self.processing_times)*1000:.2f} ms")
            print(f"Maksimum işlem süresi: {np.max(self.processing_times)*1000:.2f} ms")
        
        if self.inference_times:
            print(f"Ortalama çıkarım süresi: {np.mean(self.inference_times)*1000:.2f} ms")
        
        if self.memory_usages:
            print(f"Ortalama bellek kullanımı: {np.mean(self.memory_usages):.2f} MB")
        
        if self.cpu_usages:
            print(f"Ortalama CPU kullanımı: {np.mean(self.cpu_usages):.2f}%")
        
        print(f"📊 Toplam tespit edilen plaka sayısı: {self.get_unique_plates()}")
        print("👋 Program sonlandırıldı")

def signal_handler(sig, frame):
    """Ctrl+C handler"""
    print("\n\n🛑 Program durduruldu (Ctrl+C)")
    print("👋 Güvenli çıkış yapılıyor...")
    sys.exit(0)

def main():
    """Ana fonksiyon"""
    print("🚗 OTOMATİK PLAKA TESPİT SİSTEMİ")
    print("=" * 40)
    
    # Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        detector = AutoPlateDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Program durduruldu (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
    finally:
        print("👋 Program sonlandırıldı")

if __name__ == "__main__":
    main() 