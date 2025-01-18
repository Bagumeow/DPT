import os
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from orb import find_orb_images  # Import hàm từ file orb.py
from lbp import find_lbp_images  # Import hàm từ file lbp.py
from tkinterdnd2 import DND_FILES, TkinterDnD  # Nhập mô-đun kéo và thả

# Hàm để xóa nội dung trong ảnh tương tự
def clear_results():
    for label in result_labels:
        label.pack_forget()  # Ẩn nhãn kết quả
        label.config(text="")  # Xóa nội dung nhãn
    for img_label in match_labels:
        img_label.pack_forget()  # Ẩn nhãn ảnh kết quả
        img_label.config(image=None)  # Xóa ảnh

# Hàm xử lý khi nhấn nút "Chọn Ảnh"
def select_image():
    global demo_image_path
    clear_results()  # Xóa nội dung trước khi chọn ảnh mới
    demo_image_path = filedialog.askopenfilename(
        title="Chọn ảnh demo", 
        filetypes=[
            ("Image files", (
                "*.jpg", "*.jpeg", "*.png", "*.webp", 
                "*.bmp", "*.gif", "*.tiff", "*.ico",
                "*.avif"
            ))
        ]
    )
    load_image(demo_image_path)

# Hàm để tải ảnh vào nhãn
def load_image(image_path):
    if image_path:
        img = Image.open(image_path)

        # Thay đổi kích thước ảnh chiếm 50% chiều ngang của ứng dụng
        target_width = root.winfo_width() // 2  # 50% chiều rộng của cửa sổ
        img.thumbnail((target_width, root.winfo_height()), Image.LANCZOS)  # Giữ tỷ lệ khung hình
        img = ImageTk.PhotoImage(img)

        # Cập nhật nhãn hiển thị ảnh đầu vào
        input_image_label.config(image=img)
        input_image_label.image = img  # Giữ tham chiếu đến ảnh

# Hàm xử lý khi nhấn nút "ORB"
def orb():
    data_folder = 'data'  # Đường dẫn đến folder dữ liệu
    results = find_orb_images(demo_image_path, data_folder)  # Gọi hàm từ orb.py

    # Xóa các kết quả cũ
    clear_results()

    # Hiển thị ba kết quả tốt nhất
    for i in range(min(3, len(results))):
        best_match_path, best_distance = results[i]
        best_match_name = os.path.basename(best_match_path)

        # Tạo nhãn cho kết quả với kích thước font lớn hơn
        result_labels[i].config(text=f'{i+1}. {best_match_name} (Khoảng cách: {best_distance:.4f})', font=("Arial", 13))
        result_labels[i].pack(pady=10)

        # Tải và hiển thị ảnh giống nhất
        try:
            img = Image.open(best_match_path)
            target_height = root.winfo_height() // 4  # Chiếm 1/4 chiều cao của ứng dụng
            img.thumbnail((target_height, target_height), Image.LANCZOS)  # Giữ tỷ lệ khung hình
            img = ImageTk.PhotoImage(img)
            match_labels[i].config(image=img)
            match_labels[i].image = img  # Giữ tham chiếu đến ảnh
            match_labels[i].pack(pady=10)
        except Exception as e:
            print(f'Lỗi khi mở ảnh giống nhất: {e}')

# Hàm xử lý khi nhấn nút "LBP"
def lbp():
    data_folder = 'data'  # Đường dẫn đến folder dữ liệu
    try:
        results = find_lbp_images(demo_image_path, data_folder)  # Gọi hàm từ lbp.py
        if not results:
            print("Không tìm thấy kết quả phù hợp")
            return

        # Xóa các kết quả cũ
        clear_results()

        # Hiển thị ba kết quả tốt nhất
        for i in range(min(3, len(results))):
            # Giải nén kết quả - lbp trả về (path, distance, img)
            best_match_path, best_distance, _ = results[i]
            best_match_name = os.path.basename(best_match_path)

            # Tạo nhãn cho kết quả với kích thước font lớn hơn
            result_labels[i].config(text=f'{i+1}. {best_match_name} (Khoảng cách: {best_distance:.4f})', font=("Arial", 13))
            result_labels[i].pack(pady=10)

            # Tải và hiển thị ảnh giống nhất
            try:
                img = Image.open(best_match_path)
                target_height = root.winfo_height() // 4  # Chiếm 1/4 chiều cao của ứng dụng
                img.thumbnail((target_height, target_height), Image.LANCZOS)  # Giữ tỷ lệ khung hình
                img = ImageTk.PhotoImage(img)
                match_labels[i].config(image=img)
                match_labels[i].image = img  # Giữ tham chiếu đến ảnh
                match_labels[i].pack(pady=10)
            except Exception as e:
                print(f'Lỗi khi mở ảnh giống nhất: {e}')

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý LBP: {e}")

def cnn_sim():
    pass
# Hàm xử lý kéo và thả
def drop(event):
    global demo_image_path
    clear_results()  # Xóa nội dung trước khi kéo ảnh mới
    demo_image_path = event.data.strip('{}')  # Loại bỏ dấu ngoặc nhọn
    load_image(demo_image_path)

# Tạo giao diện chính
root = TkinterDnD.Tk()  # Sử dụng TkinterDnD thay vì tk.Tk()
root.title("Khuyến nghị ảnh tương tự")

# Thiết lập kích thước cửa sổ ứng dụng
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = int(screen_width * 0.6)
window_height = int(screen_height * 0.6)

# Căn giữa cửa sổ
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Thiết lập khung cho khu vực kéo và thả
frame = tk.Frame(root)
frame.pack(expand=True, fill=tk.BOTH)

# Khung chứa ảnh demo
demo_frame = tk.LabelFrame(frame, text="Ảnh được lựa chọn", font=("Arial", 13))
demo_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Nhãn hiển thị ảnh đầu vào
input_image_label = Label(demo_frame, bg="white", width=50, height=20)  # Nhãn nền trắng để hiển thị ảnh
input_image_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)  # Lấp đầy không gian

# Nút chọn ảnh
select_button = Button(demo_frame, text="Chọn Ảnh", font=("Arial", 13), command=select_image)
select_button.pack(pady=5)

# Khung chứa kết quả với thanh cuộn
result_frame = tk.LabelFrame(frame, text="Ảnh tương tự", font=("Arial", 13))
result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Tạo khung cuộn
canvas = tk.Canvas(result_frame)
scrollbar = tk.Scrollbar(result_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

# Đặt canvas và frame có thể cuộn
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Kết nối canvas với scrollbar
canvas.configure(yscrollcommand=scrollbar.set)

# Đặt canvas và scrollbar vào khung kết quả
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill="y")

# Nhãn hiển thị kết quả
result_labels = [Label(scrollable_frame, text="", font=("Arial", 13)) for _ in range(3)]  # Tăng kích thước font
match_labels = [Label(scrollable_frame) for _ in range(3)]  # Nhãn để hiển thị ảnh kết quả

# Đặt nhãn kết quả và ảnh vào khung
for i in range(3):
    result_labels[i].pack(pady=10)  # Tăng khoảng cách
    match_labels[i].pack(pady=10)  # Tăng khoảng cách

# Khung cho nút dự đoán và LBP
button_frame = tk.Frame(root)
button_frame.pack(pady=10)  # Thêm khoảng cách từ khung chính

# Nút ORB
orb_button = Button(button_frame, text="ORB", command=orb, font=("Arial", 13))  # Tăng kích thước font
orb_button.pack(side=tk.LEFT, padx=5)  # Đặt nút ở bên trái

# Nút LBP
lbp_button = Button(button_frame, text="LBP", command=lbp, font=("Arial", 13))  # Tăng kích thước font
lbp_button.pack(side=tk.LEFT, padx=5)  # Đặt nút ở bên trái

cnn_button = Button(button_frame, text="CNN", command=cnn_sim, font=("Arial", 13))  # Tăng kích thước font
cnn_button.pack(side=tk.LEFT, padx=5) 

# Kích hoạt tính năng kéo và thả
input_image_label.drop_target_register(DND_FILES)
input_image_label.dnd_bind('<<Drop>>', drop)

# Cấu hình bố cục cho các hàng và cột
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)

# Chạy ứng dụng
root.mainloop()