import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

print('=' * 60)
print('BUILDING THE MODEL')
print('=' * 60)

# Load pre-trained MobileNetV2 (without top layers)
base_model = MobileNetV2(
    weights='imagenet',           # Use pre-trained weights
    include_top=False,            # Remove the final classification layer
    input_shape=(224, 224, 3)     # Our image size
)

# Freeze the base model (don't train these layers yet)
base_model.trainable = False

print(f'Base model loaded: {base_model.name}')
print(f'Total layers in base: {len(base_model.layers)}')

# Add our own classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)   # Reduces dimensions
x = Dropout(0.5)(x)               # Prevents overfitting (randomly drops 50% neurons)
predictions = Dense(4, activation='softmax')(x)  # 4 classes, softmax gives probabilities

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

print('Custom layers added on top')
print('=' * 60)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
             tf.keras.metrics.Recall(name='recall')]
)

print('Model compiled successfully!')
print('=' * 60)

# Show model summary
model.summary()
print('=' * 60)
print('Model ready for training!')
print('=' * 60)
