import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

print('=' * 60)
print('SETTING UP CALLBACKS')
print('=' * 60)

# Callback 1: Save the BEST model (lowest validation loss)
checkpoint = ModelCheckpoint(
    'models/best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Callback 2: Stop training if validation loss doesn't improve for 5 epochs
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Callback 3: Reduce learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

print('Callbacks ready!')
print('- ModelCheckpoint: saves best model to models/best_model.h5')
print('- EarlyStopping: stops if no improvement for 5 epochs')
print('- ReduceLROnPlateau: lowers learning rate if stuck')
print('=' * 60)
print('STARTING TRAINING (Phase 1: Frozen Base)')
print('=' * 60)

# Train with frozen base model (fast training)
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print('=' * 60)
print('TRAINING COMPLETE!')
print('=' * 60)

best_val_acc = max(history.history['val_accuracy'])
best_val_loss = min(history.history['val_loss'])

print('Best validation accuracy: {:.4f}'.format(best_val_acc))
print('Best validation loss: {:.4f}'.format(best_val_loss))
