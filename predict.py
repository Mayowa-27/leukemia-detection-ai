import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

def predict_leukemia(image_path):
    """
    Predict leukemia type from a blood smear image.
    
    Args:
        image_path: Path to the blood smear image file
    
    Returns:
        predicted_class: The predicted class name
        confidence: Confidence percentage
        all_probabilities: Dictionary of all class probabilities
    """
    
    # Check if model exists
    if not os.path.exists('models/best_model.h5'):
        print("ERROR: Model not found! Please train the model first.")
        return None, None, None
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return None, None, None
    
    print('=' * 60)
    print('LEUKEMIA DETECTION SYSTEM')
    print('=' * 60)
    
    # Load model
    print('Loading model...')
    model = tf.keras.models.load_model('models/best_model.h5')
    print('Model loaded successfully!')
    
    # Load and preprocess image
    print(f'Loading image: {image_path}')
    img = Image.open(image_path)
    original_size = img.size
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f'Original image size: {original_size}')
    print('Preprocessing complete!')
    
    # Make prediction
    print('Analyzing...')
    prediction = model.predict(img_array, verbose=0)
    
    # Class labels
    class_labels = ['Benign', 'Early', 'Pre', 'Pro']
    
    # Get results
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_labels[predicted_index]
    confidence = prediction[0][predicted_index] * 100
    
    # All probabilities
    all_probabilities = {}
    for label, prob in zip(class_labels, prediction[0]):
        all_probabilities[label] = prob * 100
    
    # Display results
    print('=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence: {confidence:.2f}%')
    print('-' * 60)
    print('Detailed Probabilities:')
    for label, prob in all_probabilities.items():
        bar = '█' * int(prob / 5)  # Simple bar chart
        print(f'  {label:10} : {prob:6.2f}% {bar}')
    print('=' * 60)
    
    # Medical interpretation
    print('MEDICAL INTERPRETATION:')
    if predicted_class == 'Benign':
        print('  No leukemia detected. Blood cells appear normal.')
    elif predicted_class == 'Early':
        print('  Early-stage Acute Lymphoblastic Leukemia (ALL) detected.')
        print('  Further clinical evaluation recommended.')
    elif predicted_class == 'Pre':
        print('  Pre-stage Acute Lymphoblastic Leukemia (ALL) detected.')
        print('  Immediate medical attention advised.')
    elif predicted_class == 'Pro':
        print('  Pro-phase Acute Lymphoblastic Leukemia (ALL) detected.')
        print('  Urgent medical intervention required.')
    print('=' * 60)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show original image
    ax1.imshow(img)
    ax1.set_title(f'Input Image\n{os.path.basename(image_path)}', fontsize=12)
    ax1.axis('off')
    
    # Show probability chart
    colors = ['green' if label == predicted_class else 'gray' for label in class_labels]
    bars = ax2.bar(class_labels, [all_probabilities[l] for l in class_labels], color=colors)
    ax2.set_ylabel('Probability (%)', fontsize=12)
    ax2.set_title('Classification Probabilities', fontsize=12)
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, prob in zip(bars, [all_probabilities[l] for l in class_labels]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save result
    result_path = 'results/prediction_result.png'
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f'Result saved to: {result_path}')
    plt.show()
    
    return predicted_class, confidence, all_probabilities


if __name__ == "__main__":
    # If user provides an image path as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_leukemia(image_path)
    else:
        print("Usage: python predict.py <path_to_image>")
        print("\nExample:")
        print("  python predict.py data/split/test/Benign/some_image.jpg")
        print("\nOr use the function in your code:")
        print("  from predict import predict_leukemia")
        print("  predict_leukemia('path/to/image.jpg')")