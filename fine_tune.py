import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print('=' * 60)
print('FINE-TUNING PHASE')
print('=' * 60)

# Load the best model from Phase 1
print('Loading best model...')
model = tf.keras.models.load_model('models/best_model.h5')
print('Model loaded!')

# Find the MobileNetV2 layer (it's usually the first layer after input)
print('Finding base model...')
for i, layer in enumerate(model.layers):
    print(f'  Layer {i}: {layer.name} ({type(layer).__name__})')

# The base model is typically the first functional layer
base_model = None
for layer in model.layers:
    if hasattr(layer, 'layers'):  # This identifies the nested model (MobileNetV2)
        base_model = layer
        break

if base_model is None:
    print("ERROR: Could not find base model. Using alternative approach...")
    # Alternative: rebuild model architecture
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = True
    
    # Rebuild the model
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    # Load weights from best model
    temp_model = tf.keras.models.load_model('models/best_model.h5')
    model.set_weights(temp_model.get_weights())
else:
    print(f'Found base model: {base_model.name}')
    print(f'Total layers in base: {len(base_model.layers)}')
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Only unfreeze the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    print(f'Unfrozen layers: 30')
    print(f'Frozen layers: {len(base_model.layers) - 30}')

# Recompile with LOWER learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
             tf.keras.metrics.Recall(name='recall')]
)

print('Model recompiled with low learning rate (0.0001)')

# Recreate data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/split/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    'data/split/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    'data/split/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Callbacks
checkpoint = ModelCheckpoint(
    'models/best_model_finetuned.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.000001,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

print('=' * 60)
print('STARTING FINE-TUNING')
print('=' * 60)
print('This will take 20-40 minutes...')

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print('=' * 60)
print('FINE-TUNING COMPLETE!')
print('=' * 60)

# Evaluate on test set
print('Evaluating on test set...')
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f'\nFine-Tuned Test Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'Fine-Tuned Test Loss:        {test_loss:.4f}')
print(f'Fine-Tuned Test Precision:   {test_precision:.4f}')
print(f'Fine-Tuned Test Recall:      {test_recall:.4f}')

print('=' * 60)
print('COMPARISON:')
print(f'Before fine-tuning: 95.10%')
print(f'After fine-tuning:  {test_acc*100:.2f}%')
print(f'Improvement:        {((test_acc - 0.9510) * 100):.2f} percentage points')
print('=' * 60)