import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print('=' * 60)
print('CREATING VISUALIZATIONS FOR REPORT')
print('=' * 60)

# Load model
model = tf.keras.models.load_model('models/best_model.h5')
print('Model loaded!')

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/split/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get predictions
print('Generating predictions...')
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Find correct and incorrect predictions
correct = np.where(predicted_classes == true_classes)[0]
incorrect = np.where(predicted_classes != true_classes)[0]

print(f'Total test images: {len(true_classes)}')
print(f'Correct predictions: {len(correct)}')
print(f'Incorrect predictions: {len(incorrect)}')

# Plot sample predictions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Sample Predictions - Top Row: Correct, Bottom Row: Incorrect', fontsize=14)

# Show 4 correct predictions
for i, idx in enumerate(correct[:4]):
    img_path = test_generator.filepaths[idx]
    img = plt.imread(img_path)
    
    true_label = class_labels[true_classes[idx]]
    pred_label = class_labels[predicted_classes[idx]]
    confidence = predictions[idx][predicted_classes[idx]] * 100
    
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'{true_label}\n{confidence:.1f}%', color='green', fontsize=10)
    axes[0, i].axis('off')

# Show 4 incorrect predictions
for i, idx in enumerate(incorrect[:4]):
    img_path = test_generator.filepaths[idx]
    img = plt.imread(img_path)
    
    true_label = class_labels[true_classes[idx]]
    pred_label = class_labels[predicted_classes[idx]]
    confidence = predictions[idx][predicted_classes[idx]] * 100
    
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'True: {true_label}\nPred: {pred_label}\n{confidence:.1f}%', color='red', fontsize=10)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150, bbox_inches='tight')
print('Saved to: results/sample_predictions.png')
plt.show()

print('=' * 60)
print('DONE!')
print('=' * 60)