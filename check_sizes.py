import os
import cv2
import pandas as pd

dataset_path = 'data/raw/leukemia_dataset/Segmented'
classes = ['Benign', 'Early', 'Pre', 'Pro']

sizes = []
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
    
    for img_name in images[:50]:  # Check first 50 images per class (faster)
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            sizes.append({'Class': class_name, 'Width': w, 'Height': h})

df = pd.DataFrame(sizes)
print("=" * 50)
print("IMAGE SIZE ANALYSIS")
print("=" * 50)
print(df.groupby('Class')[['Width', 'Height']].describe().round(0))
print("\n" + "=" * 50)
print("Most common sizes:")
print(df.groupby(['Width', 'Height']).size().sort_values(ascending=False).head(5))
print("=" * 50)
