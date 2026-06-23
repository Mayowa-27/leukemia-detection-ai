import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Path to dataset
dataset_path = 'data/raw/leukemia_dataset/Segmented'
classes = ['Benign', 'Early', 'Pre', 'Pro']

# Create a figure to display images
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Blood Smear Images - Sample from Each Class', fontsize=16, fontweight='bold')

for row, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
    
    # Show 5 random images from each class
    np.random.seed(42)  # For reproducibility
    sample_images = np.random.choice(images, min(5, len(images)), replace=False)
    
    for col, img_name in enumerate(sample_images):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'{class_name}', fontsize=10)
        axes[row, col].axis('off')  # Hide axes

plt.tight_layout()
plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
print("✅ Sample images saved to: results/sample_images.png")
print("\nWhat to look for:")
print("- Benign: Normal round cells, uniform size")
print("- Early/Pre/Pro: Abnormal cells, irregular shapes, larger nuclei")
plt.show()
