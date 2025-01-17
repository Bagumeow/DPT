import os
import numpy as np
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from PIL import Image

# Hàm để trích xuất đặc trưng Histogram
def extract_histogram(image_path):
    image = Image.open(image_path).convert('RGB')
    histogram = image.histogram()
    return np.array(histogram)

# Hàm để trích xuất đặc trưng HOG
def extract_hog(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))  # Thay đổi kích thước cho nhất quán
    gray_image = rgb2gray(np.array(image))
    features, _ = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

# Hàm để so sánh độ tương đồng
def compare_features(hist1, hist2, hog1, hog2):
    distance_hist = np.linalg.norm(hist1 - hist2)
    distance_hog = np.linalg.norm(hog1 - hog2)
    total_distance = distance_hist + distance_hog
    return round(total_distance, 4)  # Làm tròn 4 chữ số sau dấu phẩy

# Hàm để tìm kiếm ảnh tương tự
def find_orb_images(demo_image_path, data_folder):
    demo_histogram = extract_histogram(demo_image_path)
    demo_hog = extract_hog(demo_image_path)
    results = []  # Danh sách để lưu trữ kết quả

    for image_name in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_name)
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            print(f'Đang so sánh với: {image_path}')
            
            histogram = extract_histogram(image_path)
            hog_features = extract_hog(image_path)
            distance = compare_features(demo_histogram, histogram, demo_hog, hog_features)
            
            print(f'Khoảng cách: {distance}')
            
            results.append((image_path, distance))  # Thêm kết quả vào danh sách

    # Sắp xếp kết quả theo khoảng cách
    results.sort(key=lambda x: x[1])
    
    return results  # Trả về danh sách kết quả