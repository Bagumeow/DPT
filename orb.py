import os
import numpy as np
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray

# Hàm để trích xuất đặc trưng HOG
def extract_hog(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)  # Resize ảnh về kích thước cố định
    gray_image = rgb2gray(image)  # Chuyển ảnh sang grayscale
    features, _ = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
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

# Hàm để trích xuất đặc trưng ORB
def extract_orb_features(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Khởi tạo ORB detector với các tham số tối ưu
    orb = cv2.ORB_create(
        nfeatures=500,          # Giới hạn số lượng keypoints
        scaleFactor=1.2,        # Tỷ lệ pyramid cho việc phát hiện đa tỷ lệ
        nlevels=8,              # Số level trong pyramid
        edgeThreshold=31,       # Kích thước border để không phát hiện features
        firstLevel=0,           # Level đầu tiên của pyramid
        WTA_K=2,               # Số điểm được so sánh trong descriptor
        patchSize=31,          # Kích thước patch để tính descriptor
        fastThreshold=20       # Ngưỡng FAST corner detector
    )

    # Phát hiện keypoints và tính toán descriptor
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    if descriptors is None:
        return np.zeros((1, 32), dtype=np.uint8)

    # Chỉ lấy một số lượng cố định descriptors để đảm bảo tính nhất quán
    if descriptors.shape[0] > 500:
        descriptors = descriptors[:500]
    
    return descriptors


# Hàm để chuẩn hóa đặc trưng
def normalize_features(features):
    mean = np.mean(features)
    std = np.std(features)
    return (features - mean) / std if std > 0 else features

# Hàm để so sánh độ tương đồng
def compare_features(hog1, hog2, shape1, shape2, orb1, orb2):
    # Chuẩn hóa các đặc trưng
    hog1_normalized = normalize_features(hog1)
    hog2_normalized = normalize_features(hog2)
    shape1_normalized = normalize_features(shape1)
    shape2_normalized = normalize_features(shape2)

    # Tính khoảng cách HOG bằng Euclidean Distance
    if hog1_normalized.shape == hog2_normalized.shape:
        distance_hog = np.linalg.norm(hog1_normalized - hog2_normalized)
    else:
        print("Lỗi: Kích thước HOG không khớp.")
        distance_hog = float('inf')

    # Tính khoảng cách hình dạng
    distance_shape = abs(np.linalg.norm(shape1_normalized - shape2_normalized))

    # Tính khoảng cách ORB
    if orb1 is not None and orb2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(orb1, orb2)
        orb_distance = sum([m.distance for m in matches]) / len(matches) if matches else float('inf')
    else:
        orb_distance = float('inf')  # Nếu không có đặc trưng ORB, gán khoảng cách vô hạn

    # Trọng số cho các đặc trưng
    w1, w2, w3 = 0.3, 0.4, 0.3  #
    total_distance = w1 * orb_distance + w2 * distance_hog + w3 * distance_shape
    return round(total_distance, 4)  # Làm tròn 4 chữ số sau dấu phẩy


# Hàm để tìm kiếm ảnh tương tự với đặc trưng ORB
def find_orb_images(demo_image_path, data_folder):
    demo_hog = extract_hog(demo_image_path)
    demo_shape = extract_shape_features(demo_image_path)
    demo_orb = extract_orb_features(demo_image_path)
    results = []  # Danh sách để lưu trữ kết quả

    for image_name in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_name)
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            print(f'Đang so sánh với: {image_path}')
            
            hog_features = extract_hog(image_path)
            shape_features = extract_shape_features(image_path)
            orb_features = extract_orb_features(image_path)
            distance = compare_features(demo_hog, hog_features, demo_shape, shape_features, demo_orb, orb_features)
            
            print(f'Khoảng cách: {distance}')
            
            results.append((image_path, distance))  # Thêm kết quả vào danh sách

    # Sắp xếp kết quả theo khoảng cách
    results.sort(key=lambda x: x[1])
    
    return results  # Trả về danh sách kết quả
