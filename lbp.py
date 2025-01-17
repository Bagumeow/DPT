import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Hàm tính toán đặc trưng LBP
def calculate_lbp(image):
    lbp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = lbp.shape
    lbp_result = np.zeros_like(lbp)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = lbp[i, j]
            binary = [
                lbp[i - 1, j - 1] >= center,
                lbp[i - 1, j] >= center,
                lbp[i - 1, j + 1] >= center,
                lbp[i, j + 1] >= center,
                lbp[i + 1, j + 1] >= center,
                lbp[i + 1, j] >= center,
                lbp[i + 1, j - 1] >= center,
                lbp[i, j - 1] >= center,
            ]
            lbp_result[i, j] = sum([b << k for k, b in enumerate(binary)])
    hist, _ = np.histogram(lbp_result.ravel(), bins=np.arange(257), range=(0, 256))
    hist = hist.astype("float")
    hist /= hist.sum()  # Chuẩn hóa histogram
    return hist

# Hàm tìm ảnh tương tự
def search_similar_images(demo_image_path, data_folder):
    demo_image = cv2.imread(demo_image_path)
    demo_lbp = calculate_lbp(demo_image)
    best_match_path = None
    best_score = float('inf')

    for animal_folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, animal_folder)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    print(f'Đang so sánh với: {image_path}')
                    current_image = cv2.imread(image_path)
                    current_lbp = calculate_lbp(current_image)

                    # Sử dụng khoảng cách Chi-squared để so sánh histogram
                    score = 0.5 * np.sum(((demo_lbp - current_lbp) ** 2) / (demo_lbp + current_lbp + 1e-10))
                    print(f'Điểm số: {score}')

                    if score < best_score:
                        best_score = score
                        best_match_path = image_path
    return best_match_path, best_score

# Hàm xử lý khi nhấn nút "Chọn Ảnh"
def select_image():
    global demo_image_path
    demo_image_path = filedialog.askopenfilename(title="Chọn ảnh demo", filetypes=[("Image files", "*.jpg;*.png")])

    # Hiển thị ảnh demo
    if demo_image_path:
        img = Image.open(demo_image_path)
        img.thumbnail((250, 250))  # Thay đổi kích thước cho phù hợp
        img = ImageTk.PhotoImage(img)
        demo_label.config(image=img)
        demo_label.image = img

# Hàm xử lý khi nhấn nút "Dự Đoán"
def predict():
    if not demo_image_path:
        result_label.config(text="Vui lòng chọn ảnh demo!")
        return

    data_folder = 'data'  # Thay đổi đường dẫn dữ liệu của bạn
    best_match_path, best_score = search_similar_images(demo_image_path, data_folder)

    # Hiển thị kết quả
    if best_match_path:
        best_match_name = os.path.basename(best_match_path)
        result_label.config(text=f'Giống với: {best_match_name} (Điểm số: {best_score:.2f})')

        # Hiển thị ảnh giống nhất
        try:
            img = Image.open(best_match_path)
            img.thumbnail((250, 250))  # Thay đổi kích thước cho phù hợp
            img = ImageTk.PhotoImage(img)
            match_label.config(image=img)
            match_label.image = img
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
