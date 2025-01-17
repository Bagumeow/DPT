import cv2
import os
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Hàm để trích xuất đặc trưng SIFT
def extract_sift_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors, image

# Hàm để so sánh độ tương đồng
def compare_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return matches

# Hàm xử lý khi nhấn nút "Chọn Ảnh"
def select_image():
    global demo_image_path
    
    demo_image_path = filedialog.askopenfilename(title="Chọn ảnh demo", filetypes=[("Image files", "*.jpg;*.png")])
    
    # Hiện thị ảnh demo
    if demo_image_path:
        img = Image.open(demo_image_path)
        img.thumbnail((250, 250))  # Thay đổi kích thước cho phù hợp
        img = ImageTk.PhotoImage(img)
        
        demo_label.config(image=img)
        demo_label.image = img

# Hàm xử lý khi nhấn nút "Dự Đoán"
def predict():
    demo_keypoints, demo_descriptors, demo_image = extract_sift_features(demo_image_path)
    best_match_path = None
    best_matches_count = 0

    # Đường dẫn đến folder dữ liệu
    data_folder = 'data'  # Cập nhật đường dẫn

    # Chỉ lặp qua các ảnh trong folder data
    for image_name in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_name)
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            # In đường dẫn của ảnh đang so sánh
            print(f'Đang so sánh với: {image_path}')
            
            keypoints, descriptors, match_image = extract_sift_features(image_path)
            matches = compare_features(demo_descriptors, descriptors)
            
            # In ra số lượng khớp
            print(f'Số lượng khớp: {len(matches)}')
            
            # Kiểm tra và cập nhật ảnh tốt nhất
            if len(matches) > best_matches_count:
                best_matches_count = len(matches)
                best_match_path = image_path
                print(best_match_path)

    # Hiển thị kết quả
    if best_match_path:
        best_match_name = os.path.basename(best_match_path)  # Lấy tên file từ đường dẫn
        result_label.config(text=f'Giống với: {best_match_name} ({best_matches_count} khớp)')
        
        # Hiển thị ảnh giống nhất
        try:
            img = Image.open(best_match_path)
            img.thumbnail((250, 250))  # Thay đổi kích thước cho phù hợp
            img = ImageTk.PhotoImage(img)
            
            match_label.config(image=img)
            match_label.image = img  # Lưu tham chiếu ảnh để không bị xóa
            
        except Exception as e:
            print(f'Lỗi khi mở ảnh giống nhất: {e}')
    else:
        result_label.config(text='Không tìm thấy ảnh tương đồng.')

# Tạo giao diện chính
root = tk.Tk()
root.title("Dự Đoán Động Vật")

# Nút chọn ảnh
select_button = Button(root, text="Chọn Ảnh Demo", command=select_image)
select_button.pack()

# Nhãn hiển thị ảnh demo
demo_label = Label(root)
demo_label.pack()

# Nút dự đoán
predict_button = Button(root, text="Dự Đoán", command=predict)
predict_button.pack()

# Nhãn hiển thị kết quả
result_label = Label(root, text="")
result_label.pack()

# Nhãn hiển thị ảnh giống nhất
match_label = Label(root)
match_label.pack()

# Chạy ứng dụng
root.mainloop()