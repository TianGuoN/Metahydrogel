import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


image_folder 
direction = 'horizontal' 
output_csv = 'gray_profiles.csv'


image_paths = sorted([os.path.join(image_folder, f)
                      for f in os.listdir(image_folder)
                      if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))])

if len(image_paths) == 0:
    raise FileNotFoundError("error")


first_img = cv2.imread(image_paths[0])
roi = cv2.selectROI("Select region ROI", first_img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

x, y, w, h = map(int, roi)
print(f"Selected region ROI: x={x}, y={y}, w={w}, h={h}")

# Get gray value

def extract_gray_line(img_path, roi, direction='horizontal'):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x, y, w, h = roi
    roi_gray = gray[y:y + h, x:x + w]

    if direction == 'horizontal':
        profile = np.mean(roi_gray, axis=0)  
        coords = np.arange(w)
    else:
        profile = np.mean(roi_gray, axis=1)  
        coords = np.arange(h)

    return coords, profile

# For all picture

all_profiles = []
for path in image_paths:
    _, profile = extract_gray_line(path, roi, direction)
    all_profiles.append(profile)

all_profiles = np.array(all_profiles)

plt.figure(figsize=(10, 6))

for i, profile in enumerate(all_profiles):
    plt.plot(profile, label=f"{i}")

plt.xlabel("Position (pixels)")
plt.ylabel("Gray Intensity")
plt.title("Gray Profiles from All Images (Same ROI)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Save data

if save_csv:
    np.savetxt(output_csv, all_profiles, delimiter=',')
