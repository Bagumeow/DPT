import os
import numpy as np
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from PIL import Image

# Hàm để trích xuất đặc trưng HOG
def extract_hog(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))  # Thay đổi kích thước cho nhất quán
    gray_image = rgb2gray(np.array(image))
    features, _ = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

# Hàm để trích xuất đặc trưng hình dạng
def extract_shape_features(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.array([0, 0])  # Trả về mặc định nếu không tìm thấy contour

    # Chọn contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    return np.array([area, perimeter])  # Trả về một mảng 2D với diện tích và chu vi

# Hàm để chuẩn hóa đặc trưng
def normalize_features(features):
    mean = np.mean(features)
    std = np.std(features)
    return (features - mean) / std if std > 0 else features

# Hàm để so sánh độ tương đồng
def compare_features(hog1, hog2, shape1, shape2):
    hog1_normalized = normalize_features(hog1)
    hog2_normalized = normalize_features(hog2)
    shape1_normalized = normalize_features(shape1)
    shape2_normalized = normalize_features(shape2)

    distance_hog = np.linalg.norm(hog1_normalized - hog2_normalized)
    distance_shape = np.linalg.norm(shape1_normalized - shape2_normalized)
    
    total_distance = distance_hog + distance_shape
    return round(total_distance, 4)  # Làm tròn 4 chữ số sau dấu phẩy

# Hàm để tìm kiếm ảnh tương tự
def find_orb_images(demo_image_path, data_folder):
    demo_hog = extract_hog(demo_image_path)
    demo_shape = extract_shape_features(demo_image_path)
    results = []  # Danh sách để lưu trữ kết quả

    for image_name in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_name)
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            print(f'Đang so sánh với: {image_path}')
            
            hog_features = extract_hog(image_path)
            shape_features = extract_shape_features(image_path)
            distance = compare_features(demo_hog, hog_features, demo_shape, shape_features)
            
            print(f'Khoảng cách: {distance}')
            
            results.append((image_path, distance))  # Thêm kết quả vào danh sách

    # Sắp xếp kết quả theo khoảng cách
    results.sort(key=lambda x: x[1])
    
    return results  # Trả về danh sách kết quả