import cv2

# Load image
image = cv2.imread("Sample.jpg")
if image is None:
    raise FileNotFoundError("Could not find Sample.jpg in the project folder.")

# Convert to grayscale and blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, 50, 150)

# Dilate edges to close small contour gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours detected: {len(contours)}")

# Save the first detected region of interest
saved = False
for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 1500:  # Ignore tiny noise
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]
        cv2.imwrite("crop_0.jpg", crop)
        print(f"Saved crop_0.jpg (x={x}, y={y}, w={w}, h={h}, area={area})")
        saved = True
        break

if not saved:
    print("No contours met the area threshold (> 1500 px).")