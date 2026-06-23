import os

dataset_path = 'data/raw/leukemia_dataset/Segmented'

print("=" * 60)
print("LEUKEMIA DATASET SUMMARY")
print("=" * 60)

total_images = 0
for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        # Count image files
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
        count = len(images)
        total_images += count
        print(f"{class_name:12} : {count:4} images")

print("-" * 60)
print(f"{'TOTAL':12} : {total_images:4} images")
print("=" * 60)
