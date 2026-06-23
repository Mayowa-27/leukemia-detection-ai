import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('=' * 60)
print('CREATING DATA GENERATORS')
print('=' * 60)

# Training data: WITH augmentation (to fix class imbalance + prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to 0-1
    rotation_range=20,        # Rotate images randomly up to 20 degrees
    width_shift_range=0.1,    # Shift horizontally up to 10%
    height_shift_range=0.1,   # Shift vertically up to 10%
    shear_range=0.1,          # Shear transformation
    zoom_range=0.1,           # Zoom in/out up to 10%
    horizontal_flip=True,     # Flip horizontally
    vertical_flip=True,       # Flip vertically
    fill_mode='nearest'       # Fill empty pixels after transformation
)

# Validation & Test data: NO augmentation (just normalize)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    'data/split/train',
    target_size=(224, 224),   # Resize to 224x224
    batch_size=32,            # Load 32 images at a time
    class_mode='categorical', # Multi-class classification
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    'data/split/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False             # Don't shuffle validation (for consistent metrics)
)

test_generator = val_test_datagen.flow_from_directory(
    'data/split/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print('=' * 60)
print('GENERATORS CREATED SUCCESSFULLY!')
print('=' * 60)
print(f'Training batches:   {len(train_generator)}')
print(f'Validation batches: {len(val_generator)}')
print(f'Test batches:       {len(test_generator)}')
print(f'Classes:            {train_generator.class_indices}')
print('=' * 60)
