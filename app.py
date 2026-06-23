import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from datetime import datetime
import json
import os

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="LeukemiaDetector | AI Medical Diagnosis",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== USER DATABASE ==============
USER_DB_FILE = "users.json"

def load_users():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=2)

def register_user(username, password, full_name, email, profession):
    users = load_users()
    if username in users:
        return False, "Username already exists!"
    if len(username) < 3:
        return False, "Username must be at least 3 characters!"
    if len(password) < 6:
        return False, "Password must be at least 6 characters!"
    
    users[username] = {
        "password": password,
        "full_name": full_name,
        "email": email,
        "profession": profession,
        "registered_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scan_count": 0
    }
    save_users(users)
    return True, "Account created successfully!"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, None, "Username not found!"
    if users[username]["password"] != password:
        return False, None, "Incorrect password!"
    return True, users[username], "Login successful!"

# ============== SESSION STATE ==============
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'auth_page' not in st.session_state:
    st.session_state.auth_page = "login"
if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    
    .stApp { background: #ffffff; }
    #MainMenu, footer, header { visibility: hidden; }
    
    .login-container {
        max-width: 450px;
        margin: 50px auto;
        padding: 40px;
        background: white;
        border-radius: 25px;
        box-shadow: 0 20px 60px rgba(107, 70, 193, 0.2);
        border: 2px solid #e0d4f7;
    }
    
    .login-logo { text-align: center; font-size: 4rem; margin-bottom: 10px; }
    .login-title { text-align: center; color: #6b46c1; font-size: 2.2rem; font-weight: 800; }
    .login-subtitle { text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 20px; }
    
    .nigeria-badge {
        text-align: center;
        background: linear-gradient(135deg, #6b46c1, #9f7aea);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0 25px 0;
    }
    
    .greeting-banner {
        background: linear-gradient(135deg, #6b46c1, #805ad5, #9f7aea);
        border-radius: 25px;
        padding: 40px;
        margin-bottom: 30px;
        color: white;
    }
    
    .card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid #e0d4f7;
        margin-bottom: 20px;
    }
    
    .stat-counter {
        background: white;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        border: 2px solid #e0d4f7;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #6b46c1;
    }
    
    .result-normal {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
    }
    .result-warning {
        background: linear-gradient(135deg, #ed8936, #dd6b20);
        color: white;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
    }
    .result-danger {
        background: linear-gradient(135deg, #e53e3e, #c53030);
        color: white;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
    }
    .result-critical {
        background: linear-gradient(135deg, #805ad5, #6b46c1);
        color: white;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
    }
    
    .custom-progress {
        background: #f3efff;
        border-radius: 12px;
        height: 35px;
        overflow: hidden;
        margin: 10px 0;
    }
    .custom-progress-bar {
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 15px;
        color: white;
        font-weight: 600;
    }
    
    .clinic-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        border: 2px solid #f0e6ff;
    }
    .clinic-card:hover { border-color: #6b46c1; }
    
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #6b46c1;
    }
    
    .tip-item {
        background: #faf8ff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        display: flex;
        gap: 15px;
        border: 1px solid #e0d4f7;
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6b46c1, #9f7aea);
        color: white;
        padding: 8px 18px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 5px;
    }
    
    .footer {
        text-align: center;
        padding: 40px;
        color: #999;
        margin-top: 50px;
        background: #faf8ff;
        border-radius: 20px 20px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ============== LOAD MODEL ==============
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/best_model.h5')

# ============== LOGIN PAGE ==============
def show_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-logo">🩸</div>
            <div class="login-title">LeukemiaDetector</div>
            <div class="login-subtitle">AI-Powered Blood Cancer Detection</div>
            <div style="text-align: center;"><span class="nigeria-badge">🇳🇬 Made for Nigeria 🇳🇬</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'register_success' in st.session_state and st.session_state.register_success:
            st.success("✅ Account created! Please log in.")
            st.session_state.register_success = False
        
        with st.form(key="login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("🔐 Sign In", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("❌ Please fill in all fields!")
                else:
                    success, user_data, msg = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_data = user_data
                        st.session_state.scan_count = user_data.get("scan_count", 0)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
        
        st.markdown("<div style='text-align: center; margin: 20px 0; color: #888;'>Don't have an account?</div>", unsafe_allow_html=True)
        
        if st.button("📝 Create New Account", use_container_width=True):
            st.session_state.auth_page = "register"
            st.rerun()

# ============== REGISTER PAGE ==============
def show_register_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-logo">🩸</div>
            <div class="login-title">LeukemiaDetector</div>
            <div class="login-subtitle">Create your account</div>
            <div style="text-align: center;"><span class="nigeria-badge">🇳🇬 Made for Nigeria 🇳🇬</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form(key="register_form"):
            full_name = st.text_input("👤 Full Name", placeholder="Dr. John Doe")
            email = st.text_input("📧 Email", placeholder="doctor@hospital.ng")
            username = st.text_input("🔤 Username", placeholder="johndoe123")
            password = st.text_input("🔒 Password", type="password", placeholder="Min 6 characters")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password")
            profession = st.selectbox("💼 Profession", ["Medical Doctor", "Lab Scientist", "Nurse", "Student", "Researcher", "Other"])
            
            submit = st.form_submit_button("🚀 Create Account", use_container_width=True)
            
            if submit:
                if not all([full_name, email, username, password, confirm_password]):
                    st.error("❌ Please fill in all fields!")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match!")
                elif "@" not in email:
                    st.error("❌ Please enter a valid email!")
                else:
                    success, msg = register_user(username, password, full_name, email, profession)
                    if success:
                        st.session_state.register_success = True
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
        
        st.markdown("<div style='text-align: center; margin: 20px 0; color: #888;'>Already have an account?</div>", unsafe_allow_html=True)
        
        if st.button("🔐 Sign In", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()

# ============== ABOUT LEUKEMIA ==============
def show_about_leukemia():
    st.markdown("""
    <div class="greeting-banner">
        <h1 style="color: white; margin: 0;">📚 Understanding Leukemia</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Knowledge is the first step toward prevention and early detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #6b46c1; margin: 0 0 10px 0;">🔬 What is Leukemia?</h3>
        <p style="color: #666; line-height: 1.8; margin: 0;">
            Leukemia is a cancer of the blood or bone marrow. It occurs when the bone marrow produces 
            abnormal white blood cells. In Nigeria, leukemia accounts for about 3% of all cancer cases.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #6b46c1; margin: 30px 0 20px 0;'>🩸 Types of Leukemia</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card" style="border-left-color: #48bb78;">
            <h4 style="color: #48bb78; margin: 0 0 8px 0;">🟢 Acute Lymphoblastic (ALL)</h4>
            <p style="color: #666; margin: 0; line-height: 1.6;">Most common in children. Cure rates exceed 85% with early detection.</p>
        </div>
        <div class="info-card" style="border-left-color: #ed8936;">
            <h4 style="color: #ed8936; margin: 0 0 8px 0;">🟡 Acute Myeloid (AML)</h4>
            <p style="color: #666; margin: 0; line-height: 1.6;">Affects myeloid cells, progresses quickly. More common in adults over 55.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card" style="border-left-color: #e53e3e;">
            <h4 style="color: #e53e3e; margin: 0 0 8px 0;">🟠 Chronic Lymphocytic (CLL)</h4>
            <p style="color: #666; margin: 0; line-height: 1.6;">Slow-growing, most common in adults over 60. Many live years with monitoring.</p>
        </div>
        <div class="info-card" style="border-left-color: #805ad5;">
            <h4 style="color: #805ad5; margin: 0 0 8px 0;">🔴 Chronic Myeloid (CML)</h4>
            <p style="color: #666; margin: 0; line-height: 1.6;">Affects myeloid cells, grows slowly. Targeted therapy has revolutionized treatment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #6b46c1; margin: 30px 0 20px 0;'>🛡️ Prevention & Healthy Living</h2>", unsafe_allow_html=True)
    
    tips = [
        ("🚭", "Avoid Tobacco & Smoke", "Smoking increases leukemia risk by 40%. Avoid cigarettes, shisha, and secondhand smoke."),
        ("🥗", "Eat Nigerian Superfoods", "Consume ugu, spinach, waterleaf, oranges, pineapples, pawpaw. Limit processed meats."),
        ("🏃", "Daily Exercise", "30 minutes of brisk walking, jogging, or dancing daily boosts immune function."),
        ("☢️", "Avoid Harmful Chemicals", "Minimize exposure to benzene in petrol fumes, paints, and solvents."),
        ("💤", "Quality Sleep", "Get 7-9 hours nightly. Poor sleep weakens the immune system."),
        ("💧", "Stay Hydrated", "Drink 8 glasses of water daily to help flush toxins."),
        ("🧼", "Regular Check-ups", "Annual CBC tests. Early detection saves lives!"),
        ("🌿", "Reduce Stress", "Practice meditation, prayer, or hobbies. Connect with family for mental wellness."),
    ]
    
    for icon, title, desc in tips:
        st.markdown(f"""
        <div class="tip-item">
            <div style="font-size: 2rem;">{icon}</div>
            <div>
                <h4 style="color: #6b46c1; margin: 0 0 5px 0;">{title}</h4>
                <p style="color: #666; margin: 0; line-height: 1.6; font-size: 0.9rem;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============== NIGERIAN CLINICS ==============
def show_clinics():
    st.markdown("""
    <div class="greeting-banner">
        <h1 style="color: white; margin: 0;">🏥 Nigerian Cancer Centers</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Specialized hematology and oncology centers across Nigeria</p>
    </div>
    """, unsafe_allow_html=True)
    
    clinics = [
        {"name": "🏥 National Hospital Abuja", "loc": "Central Business District, Abuja FCT", "spec": "Comprehensive cancer treatment, bone marrow transplant unit", "contact": "+234 9 461 4000"},
        {"name": "🏥 Lagos University Teaching Hospital (LUTH)", "loc": "Idi-Araba, Mushin, Lagos", "spec": "Pediatric and adult hematology-oncology, chemotherapy center", "contact": "+234 1 493 4051"},
        {"name": "🏥 University College Hospital (UCH) Ibadan", "loc": "Queen Elizabeth Road, Ibadan", "spec": "West Africa's premier teaching hospital, hematology department", "contact": "+234 2 810 4411"},
        {"name": "🏥 Ahmadu Bello University Teaching Hospital", "loc": "Zaria, Kaduna State", "spec": "Northern Nigeria's leading cancer center, radiation therapy", "contact": "+234 69 231 050"},
        {"name": "🏥 University of Nigeria Teaching Hospital", "loc": "Ituku-Ozalla, Enugu State", "spec": "Southeast Nigeria's top oncology center, stem cell research", "contact": "+234 42 258 050"},
        {"name": "🏥 University of Benin Teaching Hospital", "loc": "Ugbowo, Benin City, Edo State", "spec": "South-South region cancer care, pediatric oncology unit", "contact": "+234 52 250 078"},
        {"name": "🏥 Obafemi Awolowo University Teaching Hospital", "loc": "Ile-Ife, Osun State", "spec": "Southwest Nigeria, comprehensive cancer management", "contact": "+234 36 230 290"},
        {"name": "🏥 Federal Medical Centre Umuahia", "loc": "Umuahia, Abia State", "spec": "Eastern region hematology services, affordable cancer care", "contact": "+234 88 220 036"},
    ]
    
    for clinic in clinics:
        st.markdown(f"""
        <div class="clinic-card">
            <h4 style="color: #6b46c1; margin: 0 0 10px 0;">{clinic['name']}</h4>
            <p style="color: #666; margin: 0; line-height: 1.7; font-size: 0.9rem;">
                📍 <b>Location:</b> {clinic['loc']}<br>
                🩺 <b>Specialty:</b> {clinic['spec']}<br>
                📞 <b>Contact:</b> {clinic['contact']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #fff5f5, #fff); border: 2px solid #e53e3e;">
        <div style="text-align: center; color: #e53e3e; font-weight: 700; padding: 20px;">
            🚨 Emergency? Call immediately:<br><br>
            📞 <b>Nigeria Emergency:</b> 112 or 199<br>
            💡 Early detection saves lives. Don't delay seeking medical help!
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============== SCAN PAGE ==============
def show_scan():
    model = load_model()
    
    st.markdown("""
    <div class="greeting-banner">
        <h1 style="color: white; margin: 0;">🔬 AI Diagnosis Center</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Upload blood smear images for instant AI-powered leukemia detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-counter"><div class="stat-value">95.1%</div><div style="color: #888;">Model Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-counter"><div class="stat-value">3,256</div><div style="color: #888;">Training Images</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-counter"><div class="stat-value">4</div><div style="color: #888;">Cancer Stages</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-counter"><div class="stat-value">{st.session_state.scan_count}</div><div style="color: #888;">Your Scans</div></div>', unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown('<div class="card"><h3 style="color: #6b46c1; margin: 0 0 15px 0;">📤 Upload Blood Smear Sample</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'bmp'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Sample Preview")
            st.markdown(f'<div style="background: #f3efff; border-radius: 10px; padding: 12px; margin-top: 10px;"><p style="margin: 0; color: #6b46c1; font-size: 0.9rem;">📐 {image.size[0]} × {image.size[1]} px</p></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown('<div class="card"><h3 style="color: #6b46c1; margin: 0 0 15px 0;">📋 Analysis Results</h3>', unsafe_allow_html=True)
        
        if uploaded_file is None:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px; color: #aaa;">
                <div style="font-size: 3rem; margin-bottom: 15px;">🔬</div>
                <h4 style="color: #6b46c1; margin-bottom: 8px;">Ready for Analysis</h4>
                <p style="font-size: 0.9rem;">Upload a blood smear image to begin AI diagnosis</p>
                <div style="margin-top: 15px;">
                    <span style="display: inline-block; background: linear-gradient(135deg, #6b46c1, #9f7aea); color: white; padding: 6px 15px; border-radius: 20px; font-size: 0.8rem; margin: 3px;">🔬 Microscopy</span>
                    <span style="display: inline-block; background: linear-gradient(135deg, #6b46c1, #9f7aea); color: white; padding: 6px 15px; border-radius: 20px; font-size: 0.8rem; margin: 3px;">🤖 AI Powered</span>
                    <span style="display: inline-block; background: linear-gradient(135deg, #6b46c1, #9f7aea); color: white; padding: 6px 15px; border-radius: 20px; font-size: 0.8rem; margin: 3px;">⚡ Fast</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("🔬 RUN AI DIAGNOSIS", key="analyze"):
                st.session_state.scan_count += 1
                users = load_users()
                if st.session_state.username in users:
                    users[st.session_state.username]["scan_count"] = st.session_state.scan_count
                    save_users(users)
                
                progress = st.empty()
                
                for i in range(5):
                    progress.markdown(f"""
                    <div style="text-align: center; padding: 30px;">
                        <p style="color: #6b46c1; font-weight: 600;">{['Initializing...', 'Preprocessing...', 'Extracting features...', 'Classifying...', 'Generating report...'][i]}</p>
                        <div style="background: #f3efff; border-radius: 10px; height: 30px; overflow: hidden;">
                            <div style="height: 100%; background: linear-gradient(90deg, #6b46c1, #9f7aea); width: {(i+1)*20}%; border-radius: 10px; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-weight: 600;">{(i+1)*20}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.7)
                
                progress.empty()
                
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_array, verbose=0)
                class_labels = ['Benign', 'Early', 'Pre', 'Pro']
                predicted_index = np.argmax(prediction[0])
                predicted_class = class_labels[predicted_index]
                confidence = prediction[0][predicted_index] * 100
                
                result_styles = {
                    'Benign': ('result-normal', '✅ NORMAL', '#48bb78'),
                    'Early': ('result-warning', '⚠️ EARLY STAGE', '#ed8936'),
                    'Pre': ('result-danger', '🚨 PRE-STAGE', '#e53e3e'),
                    'Pro': ('result-critical', '🔴 PRO-PHASE', '#805ad5')
                }
                
                style, title, color = result_styles[predicted_class]
                
                st.markdown(f"""
                <div class="{style}">
                    <h3 style="margin: 0 0 10px 0; font-weight: 500;">{title}</h3>
                    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">{predicted_class}</h1>
                    <p style="font-size: 1.2rem; margin: 15px 0 0 0; opacity: 0.9;">Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                recommendations = {
                    'Benign': 'No leukemia detected. Blood cells appear normal. Continue regular health check-ups.',
                    'Early': 'Early-stage ALL detected. Schedule follow-up tests and hematologist consultation immediately.',
                    'Pre': 'Pre-stage ALL detected. Immediate specialist consultation and treatment planning needed.',
                    'Pro': 'Pro-phase ALL detected. Urgent medical intervention required. Contact an oncology center immediately.'
                }
                
                st.markdown(f"""
                <div style="background: #faf8ff; border-left: 5px solid {color}; border-radius: 15px; padding: 20px; margin: 20px 0;">
                    <h4 style="color: {color}; margin: 0 0 10px 0;">Medical Recommendation</h4>
                    <p style="color: #555; margin: 0; line-height: 1.7;">{recommendations[predicted_class]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h4 style='color: #6b46c1; margin: 20px 0 10px 0;'>📊 Classification Probabilities</h4>", unsafe_allow_html=True)
                
                bar_colors = {'Benign': '#48bb78', 'Early': '#ed8936', 'Pre': '#e53e3e', 'Pro': '#805ad5'}
                
                for label, prob in zip(class_labels, prediction[0]):
                    prob_pct = prob * 100
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-weight: 600; color: #333;">{label}</span>
                            <span style="font-weight: 600; color: {bar_colors[label]};">{prob_pct:.2f}%</span>
                        </div>
                        <div class="custom-progress">
                            <div class="custom-progress-bar" style="width: {prob_pct}%; background: {bar_colors[label]};">{prob_pct:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align: center; margin-top: 20px; padding: 15px; background: #f8f8f8; border-radius: 10px;">
                    <p style="color: #999; font-size: 0.8rem; margin: 0;">🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ⚠️ For research purposes only</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============== DASHBOARD ==============
def show_dashboard():
    # Navbar
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #6b46c1, #805ad5, #9f7aea); 
                padding: 15px 30px; border-radius: 0 0 20px 20px;
                margin: -80px -80px 20px -80px; box-shadow: 0 10px 40px rgba(107,70,193,0.2);
                display: flex; justify-content: space-between; align-items: center;">
        <div style="color: white; font-size: 1.5rem; font-weight: 800;">🩸 LeukemiaDetector</div>
        <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">
            👤 {st.session_state.user_data['full_name']} | {st.session_state.user_data['profession']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_space, col_logout = st.columns([10, 1])
    with col_logout:
        if st.button("Logout", key="logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_data = None
            st.rerun()
    
    # Greeting
    first_name = st.session_state.user_data['full_name'].split()[0]
    st.markdown(f"""
    <div class="greeting-banner">
        <h1 style="color: white; margin: 0; font-size: 2.2rem;">👋 Hey {first_name}!</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.1rem;">
            Welcome to your AI-powered leukemia detection dashboard
        </p>
        <div style="margin-top: 15px;">
            <span class="feature-badge">🇳🇬 Made for Nigeria</span>
            <span class="feature-badge">🔬 AI Powered</span>
            <span class="feature-badge">🩸 Blood Cancer Detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔬 AI Diagnosis", "📚 About Leukemia", "🏥 Find Clinics"])
    
    with tab1:
        show_scan()
    with tab2:
        show_about_leukemia()
    with tab3:
        show_clinics()
    
    st.markdown("""
    <div class="footer">
        <div style="font-size: 1.3rem; font-weight: 700; color: #6b46c1; margin-bottom: 10px;">🩸 LeukemiaDetector</div>
        <p>AI-Powered Blood Cancer Detection for Nigeria</p>
        <p style="color: #6b46c1; font-weight: 600;">🇳🇬 Final Year Project | Deep Learning for Medical Diagnosis 🇳🇬</p>
    </div>
    """, unsafe_allow_html=True)

# ============== MAIN ==============
if not st.session_state.logged_in:
    if st.session_state.auth_page == "login":
        show_login_page()
    else:
        show_register_page()
else:
    show_dashboard()