import numpy as np
import cv2
from pathlib import Path

def calculate_lbp(image, neighbors=8, radius=1):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows, cols = image.shape
    output = np.zeros_like(image)

    for row in range(radius, rows - radius):
        for col in range(radius, cols - radius):
            center = image[row, col]
            binary = []

            # Lấy 8 điểm lân cận theo radius
            for n in range(neighbors):
                x = col + radius * np.cos(2 * np.pi * n / neighbors)
                y = row - radius * np.sin(2 * np.pi * n / neighbors)
                
                # Làm tròn tọa độ
                x = int(round(x))
                y = int(round(y))

                # So sánh với điểm trung tâm
                if image[y, x] >= center:
                    binary.append(1)
                else:
                    binary.append(0)

            # Chuyển dãy bit thành số thập phân
            output[row, col] = sum([b << i for i, b in enumerate(binary)])

    return output

def extract_lbp_features(image):
    lbp = calculate_lbp(image)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    return hist

def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()

    return np.concatenate([h_hist, s_hist, v_hist])

def extract_shape_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        return -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return np.zeros(7)

def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def extract_all_features(image):
    lbp_feat = extract_lbp_features(image)
    color_feat = extract_color_features(image)
    shape_feat = extract_shape_features(image)
    return np.concatenate([lbp_feat, color_feat, shape_feat])

def calculate_similarity(features1, features2):
    lbp_len = 256 # Đã thay đổi do LBP mới dùng 256 bins
    color_len = 48

    lbp_dist = abs(cv2.compareHist(
        features1[:lbp_len].astype(np.float32),
        features2[:lbp_len].astype(np.float32),
        cv2.HISTCMP_CHISQR
    ))

    color_dist = abs(cv2.compareHist(
        features1[lbp_len:lbp_len + color_len].astype(np.float32),
        features2[lbp_len:lbp_len + color_len].astype(np.float32),
        cv2.HISTCMP_CHISQR
    ))

    shape_dist = abs(np.linalg.norm(
        features1[lbp_len + color_len:] - features2[lbp_len + color_len:]
    ))

    w1, w2, w3 = 0.3, 0.4, 0.3
    return w1 * lbp_dist + w2 * color_dist + w3 * shape_dist

def search_and_display_results(query_image, image_folder, top_k=5):
    query_image = preprocess_image(query_image)
    query_features = extract_all_features(query_image)

    results = []
    for img_path in Path(image_folder).glob('*.jpg'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = preprocess_image(img)
        features = extract_all_features(img)
        distance = calculate_similarity(query_features, features)
        results.append((str(img_path), distance, img))

    results.sort(key=lambda x: x[1])
    return results[:top_k]  # Return top K results

def find_lbp_images(query_image_path, image_folder, top_k=5):
    query_image = cv2.imread(query_image_path)
    if query_image is None:
        print("Không thể đọc ảnh query")
    return search_and_display_results(query_image, image_folder)