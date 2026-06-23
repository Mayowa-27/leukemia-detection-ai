import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
source_path = 'data/raw/leukemia_dataset/Segmented'
split_path = 'data/split'
classes = ['Benign', 'Early', 'Pre', 'Pro']

# Create split directories
splits = ['train', 'val', 'test']
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(split_path, split, cls), exist_ok=True)

print('Splitting data...')
print('=' * 50)

# Split each class
for cls in classes:
    class_path = os.path.join(source_path, cls)
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
    
    # First split: 80% train+val, 20% test
    train_val, test = train_test_split(images, test_size=0.15, random_state=42, shuffle=True)
    
    # Second split: 75% train, 25% val (from the 80%)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42, shuffle=True)
    # 0.176 of 85% ≈ 15%, so final split is ~70% train, 15% val, 15% test
    
    print(f'{cls:12}: Train={len(train):4} | Val={len(val):4} | Test={len(test):4}')
    
    # Copy files
    for img in train:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_path, 'train', cls, img))
    for img in val:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_path, 'val', cls, img))
    for img in test:
        shutil.copy(os.path.join(class_path, img), os.path.join(split_path, 'test', cls, img))

print('=' * 50)
print('Data split complete!')
