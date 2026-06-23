import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('=' * 60)
print('LOADING BEST MODEL')
print('=' * 60)

# Load the best saved model
model = tf.keras.models.load_model('models/best_model.h5')
print('Model loaded successfully!')

print('=' * 60)
print('EVALUATING ON TEST SET')
print('=' * 60)

# Create test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/split/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f'\nTest Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'Test Loss:        {test_loss:.4f}')
print(f'Test Precision:   {test_precision:.4f}')
print(f'Test Recall:      {test_recall:.4f}')

print('=' * 60)
print('GENERATING PREDICTIONS')
print('=' * 60)

# Get predictions
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print('=' * 60)
print('CLASSIFICATION REPORT')
print('=' * 60)
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print('=' * 60)
print('CONFUSION MATRIX')
print('=' * 60)
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Leukemia Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
print('Confusion matrix saved to: results/confusion_matrix.png')
plt.show()

print('=' * 60)
print('EVALUATION COMPLETE!')
print('=' * 60)