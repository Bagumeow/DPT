import cv2

# Tải ảnh
img1 = cv2.imread("demo.jpg" , cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("data/image47.png", cv2.IMREAD_GRAYSCALE)


# Khởi tạo SIFT
sift = cv2.SIFT_create()

# Phát hiện và mô tả đặc trưng
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Khớp các đặc trưng
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sắp xếp các khớp
matches = sorted(matches, key=lambda x: x.distance)

# Vẽ các khớp
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Thay đổi kích thước ảnh hiển thị
img_matches_resized = cv2.resize(img_matches, (1000, 1000))

# Hiển thị kết quả
cv2.imshow('Matches', img_matches_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()