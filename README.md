\# 🩸 LeukemiaDetector



AI-powered deep learning system for the detection and classification of leukemia from blood smear images.



\## 🇳🇬 Made for Nigeria



This project aims to assist medical professionals in Nigeria with early detection of Acute Lymphoblastic Leukemia (ALL) using artificial intelligence.



\## 📊 Model Performance



| Metric | Value |

|--------|-------|

| \*\*Accuracy\*\* | 95.10% |

| \*\*Precision\*\* | 95.47% |

| \*\*Recall\*\* | 94.69% |

| \*\*Dataset\*\* | 3,256 blood smear images |

| \*\*Classes\*\* | Benign, Early, Pre, Pro |



\## 🏗️ Architecture



\- \*\*Base Model:\*\* MobileNetV2 (Transfer Learning)

\- \*\*Framework:\*\* TensorFlow 2.21

\- \*\*Input Size:\*\* 224 × 224 pixels

\- \*\*Optimizer:\*\* Adam

\- \*\*Loss Function:\*\* Categorical Crossentropy



\## 🚀 Features



\- 🔬 AI-powered blood smear analysis

\- 📤 Web interface for image upload

\- 👤 User registration and login system

\- 📊 Real-time probability visualization

\- 🏥 Nigerian cancer center directory

\- 📚 Leukemia education and prevention tips



\## 🛠️ Installation



```bash

git clone https://github.com/Mayowa-27/leukemia-detection-ai.git

cd leukemia-detection-ai

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt



🖥️ Usage

Web App

streamlit run app.py

## 📸 Screenshots

### Login Page
&lt;img src="https://github.com/Mayowa-27/leukemia-detection-ai/raw/main/screenshots/login.png" width="800"/&gt;

### AI Diagnosis Dashboard
&lt;img src="https://github.com/Mayowa-27/leukemia-detection-ai/raw/main/screenshots/dashboard.png" width="800"/&gt;

### Scan Results with Probability Analysis
&lt;img src="https://github.com/Mayowa-27/leukemia-detection-ai/raw/main/screenshots/scan_result.png" width="800"/&gt;

### Leukemia Education & Prevention
&lt;img src="https://github.com/Mayowa-27/leukemia-detection-ai/raw/main/screenshots/about.png" width="800"/&gt;

### Nigerian Cancer Centers Directory
&lt;img src="https://github.com/Mayowa-27/leukemia-detection-ai/raw/main/screenshots/clinics.png" width="800"/&gt;![Clinics](screenshots/clinics.png)





Command Line Prediction

python predict.py path/to/image.jpg



📁 Project Structure

leukemia\_detection/

├── data/               # Dataset

├── models/             # Trained models

├── results/            # Visualizations

├── app.py              # Streamlit web app

├── predict.py          # CLI prediction

├── train\_complete.py   # Training script

└── requirements.txt    # Dependencies







👨‍💻 Author

Mayowa - Final Year Project

Year: 2026

⚠️ Disclaimer

This system is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.

