# ============================================================================
# STREAMLIT DEPLOYMENT APP FOR DIABETIC RETINOPATHY CLASSIFICATION
# WITH PATIENT PROFILE BUILDER AND DATABASE INTEGRATION
# ============================================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import database and services
from database.db import DB
from services.llm import LLM
from services.json_gen import JSONGen
from services.pdf_gen import PDFGen

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="QSight - Diabetic Retinopathy Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #2E86AB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E86AB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 8px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .head-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    .stButton button {
        background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
    }
    .patient-form {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 15px 0;
    }
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
    }
    .step {
        text-align: center;
        flex: 1;
        padding: 10px;
        border-radius: 5px;
    }
    .step.active {
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .step.completed {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CONSTANTS AND CONFIGURATION
# ============================================================================
MODEL_DIR = "trained_model"
CLASS_COLORS = {
    'No_DR': '#4CAF50',      # Green
    'Mild': '#8BC34A',       # Light Green
    'Moderate': '#FFC107',   # Yellow
    'Severe': '#FF9800',     # Orange
    'Proliferate_DR': '#F44336', # Red
    0: '#4CAF50',
    1: '#8BC34A',
    2: '#FFC107',
    3: '#FF9800',
    4: '#F44336'
}

# Initialize services
@st.cache_resource
def init_services():
    """Initialize database and services"""
    try:
        db = DB()
        llm = LLM()
        jg = JSONGen()
        pg = PDFGen()
        return db, llm, jg, pg
    except Exception as e:
        st.error(f"❌ Error initializing services: {e}")
        return None, None, None, None

# ============================================================================
# 3. MODEL COMPONENTS
# ============================================================================
class VisionEncoder(nn.Module):
    def __init__(self, encoder_type='vit', pretrained=False):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'vit':
            # Updated for torchvision compatibility
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.encoder = models.vit_b_16(weights=weights)
            self.encoder.heads = nn.Identity()
            self.projection = nn.Linear(768, 2048)
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.encoder = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )
            self.projection = nn.Identity()

    def forward(self, x):
        features = self.encoder(x)
        if self.encoder_type == 'vit':
            features = self.projection(features)
        return features

class CompressionModule(nn.Module):
    def __init__(self, input_dim=2048, compressed_dim=30):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, compressed_dim)
        )

    def forward(self, x):
        return self.compressor(x)

class ClassicalHeadA(nn.Module):
    def __init__(self, input_dim=2048, num_classes=5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class ClassicalHeadB(nn.Module):
    def __init__(self, input_dim=30, num_classes=5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class QuantumSimulatedHead(nn.Module):
    def __init__(self, input_dim=30, num_classes=5, quantum_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        current_dim = input_dim
        
        for i in range(quantum_layers):
            next_dim = 64 if i < quantum_layers - 1 else 32
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.Tanh() if i < quantum_layers - 1 else nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = next_dim
        
        layers.append(nn.Linear(32, num_classes))
        
        self.quantum_sim = nn.Sequential(*layers)

    def forward(self, x):
        return self.quantum_sim(x)

class DynamicEnsemble(nn.Module):
    def __init__(self, num_heads=3, init_temp=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.base_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.uncertainty_scales = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, head_outputs, uncertainties=None):
        weights = F.softmax(self.base_weights / self.temperature, dim=0)
        
        if uncertainties is not None:
            if uncertainties.dim() == 1:
                uncertainties = uncertainties.unsqueeze(0)
            
            scaled_uncertainties = uncertainties * self.uncertainty_scales.unsqueeze(0)
            confidence = 1.0 / (scaled_uncertainties + 1e-8)
            batch_confidence = confidence.mean(dim=0)
            confidence_weights = F.softmax(batch_confidence, dim=0)
            
            with torch.no_grad():
                predictions = torch.stack([torch.argmax(out, dim=1) for out in head_outputs], dim=1)
                predictions_float = predictions.float()
                max_vals, _ = predictions_float.max(dim=1)
                min_vals, _ = predictions_float.min(dim=1)
                agreement_mask = (max_vals == min_vals).float()
                agreement = agreement_mask.mean()
            
            uncertainty_weight = 0.7 * (1 - agreement) + 0.3
            weights = (1 - uncertainty_weight) * weights + uncertainty_weight * confidence_weights
        
        weights = weights / weights.sum()
        
        if not self.training:
            weights = weights.detach()
        
        final_output = sum(w * out for w, out in zip(weights, head_outputs))
        return final_output, weights

# ============================================================================
# 4. HYBRID MODEL FOR DEPLOYMENT
# ============================================================================
class HybridDRModel(nn.Module):
    def __init__(self, model_info):
        super().__init__()
        
        self.classes = model_info.get('classes', ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'])
        self.compressed_dim = model_info.get('compressed_dim', 30)
        self.num_classes = len(self.classes)
        self.encoder_type = model_info.get('encoder_type', 'vit')
        self.ensemble_weights = model_info.get('ensemble_weights', [0.333, 0.333, 0.333])
        
        self.vision_encoder = VisionEncoder(encoder_type=self.encoder_type, pretrained=False)
        self.compression = CompressionModule(input_dim=2048, compressed_dim=self.compressed_dim)
        self.classical_head_a = ClassicalHeadA(input_dim=2048, num_classes=self.num_classes)
        self.classical_head_b = ClassicalHeadB(input_dim=self.compressed_dim, num_classes=self.num_classes)
        
        self.quantum_head = QuantumSimulatedHead(input_dim=self.compressed_dim, num_classes=self.num_classes)
        
        self.ensemble = DynamicEnsemble(num_heads=3)
        
        if 'ensemble_weights' in model_info:
            with torch.no_grad():
                self.ensemble.base_weights.data = torch.tensor(model_info['ensemble_weights'])
        
        self.uncertainty_a = nn.Sequential(
            nn.Linear(2048, 64), nn.ReLU(), 
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.uncertainty_b = nn.Sequential(
            nn.Linear(self.compressed_dim, 32), nn.ReLU(), 
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.uncertainty_c = nn.Sequential(
            nn.Linear(self.compressed_dim, 32), nn.ReLU(), 
            nn.Linear(32, 1), nn.Sigmoid()
        )
    
    def forward(self, x, return_all=True):
        latent_features = self.vision_encoder(x)
        compressed_features = self.compression(latent_features)
        
        output_a = self.classical_head_a(latent_features)
        output_b = self.classical_head_b(compressed_features)
        output_c = self.quantum_head(compressed_features)
        
        unc_a = self.uncertainty_a(latent_features)
        unc_b = self.uncertainty_b(compressed_features)
        unc_c = self.uncertainty_c(compressed_features)
        
        uncertainties = torch.cat([unc_a, unc_b, unc_c], dim=1).squeeze()
        
        head_outputs = [output_a, output_b, output_c]
        final_output, ensemble_weights = self.ensemble(head_outputs, uncertainties)
        
        if return_all:
            return {
                'output_a': output_a,
                'output_b': output_b,
                'output_c': output_c,
                'final_output': final_output,
                'ensemble_weights': ensemble_weights,
                'uncertainties': uncertainties,
                'probabilities': F.softmax(final_output, dim=1),
                'latent_features': latent_features,
                'compressed_features': compressed_features
            }
        else:
            return final_output

# ============================================================================
# 5. UTILITY FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained hybrid model"""
    try:
        model_path = os.path.join(MODEL_DIR, 'phase1_classical_model.pth')
        info_path = os.path.join(MODEL_DIR, 'model_info.pkl')
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Please copy 'phase1_classical_model.pth' from Colab to the 'trained_model' folder")
            return None, None, None
        
        if not os.path.exists(info_path):
            st.error(f"❌ Model info file not found at: {info_path}")
            st.info("Please copy 'model_info.pkl' from Colab to the 'trained_model' folder")
            return None, None, None
        
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        model = HybridDRModel(model_info)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        class_names = model_info.get('classes', ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'])
        ensemble_weights = model_info.get('ensemble_weights', [0.333, 0.333, 0.333])
        
        return model, class_names, ensemble_weights
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor):
    """Make prediction on a single image"""
    with torch.no_grad():
        outputs = model(image_tensor, return_all=True)
    
    probs = outputs['probabilities'].squeeze()
    prediction_idx = torch.argmax(probs).item()
    confidence = probs[prediction_idx].item()
    
    _, pred_a = torch.max(outputs['output_a'], 1)
    _, pred_b = torch.max(outputs['output_b'], 1)
    _, pred_c = torch.max(outputs['output_c'], 1)
    
    return {
        'final_prediction': prediction_idx,
        'final_confidence': confidence,
        'classical_a_pred': pred_a.item(),
        'classical_b_pred': pred_b.item(),
        'quantum_pred': pred_c.item(),
        'probabilities': probs.numpy(),
        'ensemble_weights': outputs['ensemble_weights'].detach().cpu().numpy(),
        'uncertainties': outputs['uncertainties'].detach().cpu().numpy()
    }

def create_patient_profile(db, patient_data):
    """Create patient profile in database"""
    try:
        patient = db.create_patient(patient_data)
        return patient
    except Exception as e:
        st.error(f"❌ Error creating patient profile: {e}")
        return None

def calculate_risk_score(patient, diagnosis_results):
    """Calculate risk score based on patient data and diagnosis"""
    base_risk = 0
    
    # Age factor
    if patient.age > 50:
        base_risk += 1
    if patient.age > 60:
        base_risk += 1
    
    # BMI factor
    if patient.bmi:
        if patient.bmi > 25:
            base_risk += 1
        if patient.bmi > 30:
            base_risk += 1
    
    # Insulin factor
    if patient.insulin > 10:
        base_risk += 1
    
    # Lifestyle factors
    if patient.smoker:
        base_risk += 2
    if patient.alcohol == "High":
        base_risk += 1
    
    # Vascular disease
    if patient.vascular:
        base_risk += 2
    
    # Diagnosis severity factor
    severity_map = {
        'No_DR': 0,
        'Mild': 1,
        'Moderate': 2,
        'Severe': 3,
        'Proliferate_DR': 4
    }
    
    left_severity = severity_map.get(diagnosis_results.get('retinopathy_left', 'No_DR'), 0)
    right_severity = severity_map.get(diagnosis_results.get('retinopathy_right', 'No_DR'), 0)
    
    severity_score = max(left_severity, right_severity)
    base_risk += severity_score
    
    # Normalize to 0-10 scale
    risk_score = min(10, max(0, base_risk))
    
    return risk_score

# ============================================================================
# 6. PATIENT PROFILE BUILDER COMPONENT
# ============================================================================
def patient_profile_builder():
    """Display patient profile form"""
    st.markdown('<div class="patient-form">', unsafe_allow_html=True)
    st.markdown("### 👤 Patient Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name", placeholder="John Doe", key="patient_name")
        age = st.number_input("Age", min_value=0, max_value=120, value=45, key="patient_age")
        sex = st.selectbox("Sex", ["Male", "Female", "Other"], key="patient_sex")
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=75.0, step=0.1, key="patient_weight")
        height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, value=175.0, step=0.1, key="patient_height")
    
    with col2:
        insulin = st.number_input("Insulin Level", min_value=0.0, value=12.0, step=0.1, key="patient_insulin")
        smoker = st.checkbox("Smoker", key="patient_smoker")
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"], key="patient_alcohol")
        vascular = st.checkbox("Vascular Disease", key="patient_vascular")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'name': name,
        'age': int(age),
        'sex': sex,
        'weight': float(weight),
        'height': float(height),
        'insulin': float(insulin),
        'smoker': bool(smoker),
        'alcohol': alcohol,
        'vascular': bool(vascular)
    }

# ============================================================================
# 7. EYE UPLOAD COMPONENT
# ============================================================================
def eye_upload_component():
    """Component for uploading left and right eye images"""
    st.markdown('<div class="patient-form">', unsafe_allow_html=True)
    st.markdown("### 👁️ Eye Image Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Left Eye")
        left_eye = st.file_uploader(
            "Upload Left Eye Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="left_eye_uploader"
        )
        if left_eye:
            left_image = Image.open(left_eye).convert('RGB')
            st.image(left_image, caption="Left Eye", width=300)
        else:
            left_image = None
    
    with col2:
        st.markdown("#### Right Eye")
        right_eye = st.file_uploader(
            "Upload Right Eye Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="right_eye_uploader"
        )
        if right_eye:
            right_image = Image.open(right_eye).convert('RGB')
            st.image(right_image, caption="Right Eye", width=300)
        else:
            right_image = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return left_image, right_image

# ============================================================================
# 8. STEP DISPLAY FUNCTIONS
# ============================================================================
def display_step_indicator(current_step):
    """Display step indicator"""
    steps = ["1. Patient Profile", "2. Upload Images", "3. Analyze", "4. View Report"]
    current_index = steps.index(current_step) if current_step in steps else 0
    
    html = '<div class="step-indicator">'
    for i, step in enumerate(steps):
        step_class = "step"
        if i == current_index:
            step_class += " active"
        elif i < current_index:
            step_class += " completed"
        html += f'<div class="{step_class}">{step}</div>'
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

def display_patient_profile_step(db):
    """Display patient profile step"""
    display_step_indicator("1. Patient Profile")
    st.markdown('<h3 class="sub-header">Step 1: Patient Information</h3>', unsafe_allow_html=True)
    
    # Check if patient already exists in session
    if st.session_state.current_patient:
        st.success(f"✅ Patient profile already created: {st.session_state.current_patient.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 View Current Patient", type="secondary", key="view_current_patient"):
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### 👤 Current Patient")
                st.write(f"**Name:** {st.session_state.current_patient.name}")
                st.write(f"**Age:** {st.session_state.current_patient.age}")
                st.write(f"**Sex:** {st.session_state.current_patient.sex}")
                if st.session_state.current_patient.bmi:
                    st.write(f"**BMI:** {st.session_state.current_patient.bmi:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("🔄 Create New Patient", type="primary", key="create_new_patient"):
                # Clear current patient
                st.session_state.current_patient = None
                st.session_state.patient_data = None
                st.rerun()
        
        # Navigation to next step
        st.markdown("---")
        st.markdown("### Next Step")
        st.info("Patient profile is complete. Proceed to Step 2 to upload eye images.")
        if st.button("➡️ Go to Step 2: Upload Images", type="primary", key="nav_to_step2"):
            st.session_state.workflow_step = "2. Upload Images"
            st.rerun()
        
        return
    
    # Get patient data from form
    st.info("Please fill in the patient information below:")
    patient_data = patient_profile_builder()
    
    # Display summary
    if patient_data['name']:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📋 Patient Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {patient_data['name']}")
            st.write(f"**Age:** {patient_data['age']}")
            st.write(f"**Sex:** {patient_data['sex']}")
        
        with col2:
            if patient_data['height'] > 0:
                bmi = round(patient_data['weight'] / ((patient_data['height'] / 100) ** 2), 2)
                st.write(f"**Weight:** {patient_data['weight']} kg")
                st.write(f"**Height:** {patient_data['height']} cm")
                st.write(f"**BMI:** {bmi}")
                bmi_status = "Normal" if 18.5 <= bmi <= 24.9 else "Abnormal"
                st.write(f"**Status:** {bmi_status}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation check
    is_valid = True
    validation_errors = []
    
    if not patient_data['name'] or patient_data['name'].strip() == "":
        validation_errors.append("Name is required")
        is_valid = False
    
    if patient_data['age'] <= 0:
        validation_errors.append("Age must be greater than 0")
        is_valid = False
    
    if patient_data['height'] <= 0:
        validation_errors.append("Height must be greater than 0")
        is_valid = False
    
    if patient_data['weight'] <= 0:
        validation_errors.append("Weight must be greater than 0")
        is_valid = False
    
    # Show validation errors
    if validation_errors:
        st.error("❌ Please fix the following errors:")
        for error in validation_errors:
            st.write(f"- {error}")
    
    # Save patient button
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("💾 Save Patient Profile", type="primary", disabled=not is_valid, key="save_patient"):
            with st.spinner("Saving patient profile..."):
                patient = create_patient_profile(db, patient_data)
                if patient:
                    st.session_state.current_patient = patient
                    st.session_state.patient_data = patient_data
                    st.success(f"✅ Patient '{patient.name}' saved successfully!")
                    
                    # Create a placeholder for navigation
                    nav_placeholder = st.empty()
                    with nav_placeholder.container():
                        st.info("✅ Patient profile saved! Click below to proceed.")
                        if st.button("➡️ Go to Step 2: Upload Images", key="nav_after_save"):
                            st.session_state.workflow_step = "2. Upload Images"
                            st.rerun()
                else:
                    st.error("❌ Failed to save patient profile")
    
    with col2:
        if st.button("🗑️ Clear Form", type="secondary", key="clear_form"):
            st.session_state.current_patient = None
            st.session_state.patient_data = None
            st.rerun()

def display_upload_images_step():
    """Display image upload step"""
    display_step_indicator("2. Upload Images")
    st.markdown('<h3 class="sub-header">Step 2: Upload Eye Images</h3>', unsafe_allow_html=True)
    
    # Display patient info
    if st.session_state.current_patient:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 👤 Current Patient")
        st.write(f"**Name:** {st.session_state.current_patient.name}")
        st.write(f"**Age:** {st.session_state.current_patient.age}")
        st.write(f"**Sex:** {st.session_state.current_patient.sex}")
        st.write(f"**Patient ID:** {st.session_state.current_patient.id}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload images
    left_image, right_image = eye_upload_component()
    
    # Store images in session state
    st.session_state.left_image = left_image
    st.session_state.right_image = right_image
    
    # Check if both images are uploaded
    images_uploaded = left_image is not None and right_image is not None
    
    if images_uploaded:
        st.success("✅ Both eye images uploaded successfully!")
        
        # Navigation to next step
        st.markdown("---")
        st.markdown("### Next Step")
        st.info("Eye images are ready. Proceed to Step 3 for AI analysis.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Patient Profile", type="secondary", key="back_to_profile"):
                st.session_state.workflow_step = "1. Patient Profile"
                st.rerun()
        
        with col2:
            if st.button("➡️ Go to Step 3: Analyze", type="primary", key="nav_to_analyze"):
                st.session_state.workflow_step = "3. Analyze"
                st.rerun()
    
    else:
        # Show which images are missing
        missing_images = []
        if left_image is None:
            missing_images.append("Left Eye")
        if right_image is None:
            missing_images.append("Right Eye")
        
        if missing_images:
            st.warning(f"⚠️ Please upload: {', '.join(missing_images)}")
        
        # Navigation back
        st.markdown("---")
        if st.button("⬅️ Back to Patient Profile", type="secondary", key="back_to_profile2"):
            st.session_state.workflow_step = "1. Patient Profile"
            st.rerun()

def display_analyze_step(db, llm):
    """Display analysis step"""
    display_step_indicator("3. Analyze")
    st.markdown('<h3 class="sub-header">Step 3: AI Analysis</h3>', unsafe_allow_html=True)
    
    # Check prerequisites
    if not st.session_state.model:
        st.warning("⚠️ Please load the AI model first (use sidebar button)")
        return
    
    # Display patient and image info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 👤 Patient")
        st.write(f"**Name:** {st.session_state.current_patient.name}")
        st.write(f"**ID:** {st.session_state.current_patient.id}")
        st.write(f"**Age:** {st.session_state.current_patient.age}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📸 Images")
        st.write("✅ Left eye image uploaded")
        st.write("✅ Right eye image uploaded")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if analysis already done
    if st.session_state.diagnosis_results:
        st.success("✅ Analysis already completed!")
        
        diagnosis = st.session_state.diagnosis_results['summary']
        
        # Show quick results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Left Eye", diagnosis['retinopathy_left'])
        with col2:
            st.metric("Right Eye", diagnosis['retinopathy_right'])
        
        # Navigation
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Images", type="secondary", key="back_to_images"):
                st.session_state.workflow_step = "2. Upload Images"
                st.rerun()
        with col2:
            if st.button("➡️ View Full Report", type="primary", key="view_report_from_analyze"):
                st.session_state.workflow_step = "4. View Report"
                st.rerun()
    
    else:
        # Analyze button
        if st.button("🔍 Start AI Analysis", type="primary", key="start_analysis"):
            with st.spinner("Analyzing images with AI..."):
                try:
                    # Analyze left eye
                    left_tensor = preprocess_image(st.session_state.left_image)
                    left_result = predict_image(st.session_state.model, left_tensor)
                    left_class = st.session_state.class_names[left_result['final_prediction']]
                    
                    # Analyze right eye
                    right_tensor = preprocess_image(st.session_state.right_image)
                    right_result = predict_image(st.session_state.model, right_tensor)
                    right_class = st.session_state.class_names[right_result['final_prediction']]
                    
                    # Calculate average confidence
                    avg_confidence = (left_result['final_confidence'] + right_result['final_confidence']) / 2
                    
                    # Calculate risk score
                    diagnosis_data = {
                        'retinopathy_left': left_class,
                        'retinopathy_right': right_class,
                        'confidence': float(avg_confidence * 100),
                        'left_confidence': float(left_result['final_confidence'] * 100),
                        'right_confidence': float(right_result['final_confidence'] * 100)
                    }
                    
                    risk_score = calculate_risk_score(st.session_state.current_patient, diagnosis_data)
                    diagnosis_data['risk'] = float(risk_score)
                    
                    # Store diagnosis results
                    st.session_state.diagnosis_results = {
                        'left': left_result,
                        'right': right_result,
                        'summary': diagnosis_data
                    }
                    
                    # Generate LLM report
                    with st.spinner("Generating comprehensive report with AI..."):
                        llm_report = llm.generate_report(st.session_state.current_patient, diagnosis_data)
                        if llm_report:
                            st.session_state.llm_report = llm_report
                        else:
                            st.session_state.llm_report = {
                                "condition_overview": "AI report generation failed. Using basic analysis.",
                                "patient_assessment": f"Patient {st.session_state.current_patient.name} diagnosed with {left_class} in left eye and {right_class} in right eye.",
                                "compliance_notice": "Consult with a healthcare professional for accurate diagnosis."
                            }
                    
                    # Save to database
                    try:
                        # Prepare diagnosis data for database
                        diag_data = {
                            'patient_id': st.session_state.current_patient.id,
                            'retinopathy_left': left_class,
                            'retinopathy_right': right_class,
                            'confidence': float(avg_confidence * 100),
                            'risk': float(risk_score),
                            'left_img': 'left_eye.jpg',
                            'right_img': 'right_eye.jpg',
                            'summary': str(diagnosis_data)
                        }
                        
                        # Create diagnosis record
                        diagnosis = db.create_diagnosis(diag_data)
                        st.session_state.diagnosis_id = diagnosis.id
                        
                        st.success("✅ Analysis complete!")
                        
                        # Auto-navigate to report
                        st.info("✅ Analysis complete! Proceed to Step 4 to view the report.")
                        
                        if st.button("➡️ Go to Report", type="primary", key="go_to_report_auto"):
                            st.session_state.workflow_step = "4. View Report"
                            st.rerun()
                        
                    except Exception as db_error:
                        st.warning(f"⚠️ Analysis complete but database error: {db_error}")
                        st.success("✅ Analysis complete! You can view the report.")
                        
                        if st.button("➡️ View Report", type="primary", key="view_report_manual"):
                            st.session_state.workflow_step = "4. View Report"
                            st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Navigation back
        st.markdown("---")
        if st.button("⬅️ Back to Images", type="secondary", key="back_to_images2"):
            st.session_state.workflow_step = "2. Upload Images"
            st.rerun()

# In the display_report_step function, update the report generation section:

def display_report_step(db, jg, pg):
    """Display report generation step"""
    display_step_indicator("4. View Report")
    st.markdown('<h3 class="sub-header">Step 4: Comprehensive Report</h3>', unsafe_allow_html=True)
    
    if not st.session_state.diagnosis_results:
        st.error("❌ No diagnosis results found. Please complete the analysis first.")
        if st.button("⬅️ Go to Analysis", type="primary"):
            st.session_state.workflow_step = "3. Analyze"
            st.rerun()
        return
    
    # Display diagnosis results
    diagnosis = st.session_state.diagnosis_results['summary']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### 👁️ Left Eye")
        left_color = CLASS_COLORS.get(diagnosis['retinopathy_left'], '#2E86AB')
        st.markdown(f"<h3 style='color: {left_color};'>{diagnosis['retinopathy_left']}</h3>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {diagnosis.get('left_confidence', 0):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### 👁️ Right Eye")
        right_color = CLASS_COLORS.get(diagnosis['retinopathy_right'], '#2E86AB')
        st.markdown(f"<h3 style='color: {right_color};'>{diagnosis['retinopathy_right']}</h3>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {diagnosis.get('right_confidence', 0):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Overall metrics
    st.markdown("### 📊 Overall Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Confidence", f"{diagnosis['confidence']:.1f}%")
    
    with col2:
        st.metric("Risk Score", f"{diagnosis['risk']:.1f}/10")
    
    with col3:
        severity_map = {
            'No_DR': 'Low',
            'Mild': 'Mild',
            'Moderate': 'Moderate',
            'Severe': 'High',
            'Proliferate_DR': 'Critical'
        }
        left_severity = severity_map.get(diagnosis['retinopathy_left'], 'Unknown')
        right_severity = severity_map.get(diagnosis['retinopathy_right'], 'Unknown')
        overall_severity = max(left_severity, right_severity)
        st.metric("Overall Severity", overall_severity)
    
    # Display LLM report if available
    if st.session_state.llm_report:
        st.markdown("### 📝 AI-Generated Report")
        
        with st.expander("View Detailed Report", expanded=True):
            report = st.session_state.llm_report
            
            sections = [
                ("Condition Overview", "condition_overview"),
                ("Patient Assessment", "patient_assessment"),
                ("Clinical Implications", "implications"),
                ("Treatment Plan", "treatment_plan"),
                ("Life Impact", "life_impact"),
                ("Financial Impact", "financial_impact"),
                ("Recovery Projection", "recovery_projection"),
                ("Additional Assessments", "additional_assessments"),
            ]
            
            for title, key in sections:
                if key in report and report[key]:
                    st.markdown(f"**{title}**")
                    
                    # Format the content properly
                    content = report[key]
                    
                    if isinstance(content, str):
                        # Clean up bullet formatting
                        lines = content.split('•')
                        if len(lines) > 1:
                            for line in lines:
                                line = line.strip()
                                if line:
                                    st.write(f"• {line}")
                        else:
                            st.write(content)
                    elif isinstance(content, list):
                        for item in content:
                            st.write(f"• {item}")
                    else:
                        st.write(str(content))
                    
                    st.markdown("---")
        
        # Compliance notice
        if "compliance_notice" in report:
            st.info(report["compliance_notice"])
    else:
        st.warning("⚠️ No AI-generated report available. Basic diagnosis shown above.")
    
    # Report generation section
    st.markdown("### 📄 Report Download")
    
    # Check if reports already exist
    import os
    import json
    
    json_path = os.path.join("json_outputs", f"diagnosis_{st.session_state.current_patient.id}.json")
    pdf_path = os.path.join("reports", f"report_{st.session_state.current_patient.id}.pdf")
    
    json_exists = os.path.exists(json_path)
    pdf_exists = os.path.exists(pdf_path)
    
    if json_exists and pdf_exists:
        st.success("✅ Reports already generated!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                with open(json_path, 'rb') as f:
                    json_data = f.read()
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_data,
                    file_name=f"diagnosis_{st.session_state.current_patient.id}.json",
                    mime="application/json",
                    key="download_json"
                )
            except Exception as e:
                st.error(f"Error reading JSON file: {e}")
        
        with col2:
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"report_{st.session_state.current_patient.id}.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
        
        # Show file info
        with st.expander("📁 File Information"):
            st.write(f"**JSON File:** `{json_path}`")
            st.write(f"**PDF File:** `{pdf_path}`")
            st.write(f"**Generated:** {datetime.fromtimestamp(os.path.getmtime(pdf_path)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        # Generate reports button
        if st.button("🔄 Generate Reports", type="primary", key="generate_reports"):
            with st.spinner("Generating reports..."):
                try:
                    # Prepare diagnosis data with all required fields
                    diagnosis_data = {
                        'retinopathy_left': diagnosis['retinopathy_left'],
                        'retinopathy_right': diagnosis['retinopathy_right'],
                        'confidence': diagnosis['confidence'],
                        'risk': diagnosis['risk'],
                        'left_img': 'left_eye.jpg',
                        'right_img': 'right_eye.jpg',
                        'left_confidence': diagnosis.get('left_confidence', 0),
                        'right_confidence': diagnosis.get('right_confidence', 0),
                        'llm': st.session_state.llm_report if st.session_state.llm_report else {}
                    }
                    
                    # Generate JSON
                    json_path = jg.create(st.session_state.current_patient, diagnosis_data)
                    
                    # Generate PDF
                    pdf_path = pg.create(st.session_state.current_patient, diagnosis_data, 
                                       st.session_state.llm_report if st.session_state.llm_report else {})
                    
                    st.success("✅ Reports generated successfully!")
                    
                    # Provide immediate download buttons
                    st.markdown("### 📥 Download Reports")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        try:
                            with open(json_path, 'rb') as f:
                                json_data = f.read()
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_data,
                                file_name=f"diagnosis_{st.session_state.current_patient.id}.json",
                                mime="application/json",
                                key="download_json_new"
                            )
                        except Exception as e:
                            st.error(f"Error reading JSON file: {e}")
                    
                    with col2:
                        try:
                            with open(pdf_path, 'rb') as f:
                                pdf_data = f.read()
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_data,
                                file_name=f"report_{st.session_state.current_patient.id}.pdf",
                                mime="application/pdf",
                                key="download_pdf_new"
                            )
                        except Exception as e:
                            st.error(f"Error reading PDF file: {e}")
                    
                    # Auto-rerun to update UI
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating reports: {str(e)}")
                    import traceback
                    with st.expander("🔧 Technical Details"):
                        st.code(traceback.format_exc())
    
    # Alternative: Direct PDF generation with single button
    st.markdown("---")
    st.markdown("### Quick PDF Download")
    
    if st.button("📄 Generate & Download PDF Only", type="secondary", key="quick_pdf"):
        with st.spinner("Creating PDF..."):
            try:
                # Prepare minimal data for PDF
                pdf_data = {
                    'retinopathy_left': diagnosis['retinopathy_left'],
                    'retinopathy_right': diagnosis['retinopathy_right'],
                    'confidence': diagnosis['confidence'],
                    'risk': diagnosis['risk'],
                    'left_img': 'left_eye.jpg',
                    'right_img': 'right_eye.jpg',
                    'left_confidence': diagnosis.get('left_confidence', 0),
                    'right_confidence': diagnosis.get('right_confidence', 0),
                    'llm': st.session_state.llm_report if st.session_state.llm_report else {
                        'condition_overview': f"Patient diagnosed with {diagnosis['retinopathy_left']} in left eye and {diagnosis['retinopathy_right']} in right eye.",
                        'patient_assessment': f"Risk score: {diagnosis['risk']}/10. Confidence: {diagnosis['confidence']:.1f}%",
                        'compliance_notice': 'This is a preliminary AI analysis. Consult with healthcare professionals for accurate diagnosis.'
                    }
                }
                
                # Generate PDF
                pdf_path = pg.create(st.session_state.current_patient, pdf_data, pdf_data['llm'])
                
                # Provide download
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                st.download_button(
                    label="⬇️ Click to Download PDF",
                    data=pdf_bytes,
                    file_name=f"qsight_report_{st.session_state.current_patient.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="direct_pdf_download"
                )
                
                st.success("✅ PDF ready for download!")
                
            except Exception as e:
                st.error(f"❌ Error creating PDF: {str(e)}")
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Analysis", type="secondary", key="back_to_analysis"):
            st.session_state.workflow_step = "3. Analyze"
            st.rerun()
    with col2:
        if st.button("🏠 Start Over", type="primary", key="start_over"):
            # Clear session state
            for key in ['current_patient', 'patient_data', 'diagnosis_results', 'llm_report', 'left_image', 'right_image']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.workflow_step = "1. Patient Profile"
            st.rerun()

# ============================================================================
# 9. MAIN STREAMLIT APP
# ============================================================================
def main():
    # Title and description
    st.markdown('<h1 class="main-header">👁️ QSight - Diabetic Retinopathy Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Hybrid Classical-Quantum Deep Learning Model with Patient Management System")
    
    # Initialize services
    db, llm, jg, pg = init_services()
    
    # Initialize session state for workflow
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = "1. Patient Profile"
    
    # Initialize other session states
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    if 'diagnosis_results' not in st.session_state:
        st.session_state.diagnosis_results = None
    if 'llm_report' not in st.session_state:
        st.session_state.llm_report = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🚀 Quick Start")
        
        # Load model button
        if st.button("🔄 Load AI Model", type="primary", use_container_width=True, key="load_model"):
            with st.spinner("Loading model..."):
                model, class_names, ensemble_weights = load_model()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.class_names = class_names
                    st.success(f"✅ Model loaded!")
                    st.info(f"Classes: {class_names}")
                else:
                    st.error("❌ Failed to load model")
        
        st.markdown("---")
        st.markdown("## 📊 Database Status")
        
        # Check database connection
        if db is not None:
            try:
                patient_count = len(db.list_patients())
                st.success(f"✅ Database connected")
                st.info(f"📋 Patients in database: {patient_count}")
            except Exception as e:
                st.error(f"❌ Database error: {e}")
        else:
            st.error("❌ Database not initialized")
        
        st.markdown("---")
        st.markdown("## 🏥 Workflow")
        
        # Simple workflow selector
        workflow_step = st.radio(
            "Select Step:",
            ["1. Patient Profile", "2. Upload Images", "3. Analyze", "4. View Report"],
            index=["1. Patient Profile", "2. Upload Images", "3. Analyze", "4. View Report"].index(st.session_state.workflow_step)
        )
        
        # Update workflow step if changed
        if workflow_step != st.session_state.workflow_step:
            st.session_state.workflow_step = workflow_step
            st.rerun()
        
        st.markdown("---")
        st.markdown("## 🛠️ Quick Actions")
        
        if st.button("🔄 Reset All", type="secondary", use_container_width=True, key="reset_all"):
            # Clear session state
            for key in ['current_patient', 'patient_data', 'diagnosis_results', 'llm_report', 'left_image', 'right_image']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.workflow_step = "1. Patient Profile"
            st.rerun()
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.info("""
        **QSight System Features:**
        - Patient profile management
        - Dual-eye analysis
        - Hybrid AI model (2 Classical + Quantum-inspired)
        - LLM-powered report generation
        - PDF and JSON output
        - Database storage
        
        **DR Severity Classes:**
        0: No DR (Normal)
        1: Mild DR
        2: Moderate DR
        3: Severe DR
        4: Proliferative DR
        """)
    
    # Main content based on workflow step
    if st.session_state.workflow_step == "1. Patient Profile":
        display_patient_profile_step(db)
    
    elif st.session_state.workflow_step == "2. Upload Images":
        if st.session_state.current_patient is None:
            st.warning("⚠️ Please complete Patient Profile first")
            st.session_state.workflow_step = "1. Patient Profile"
            st.rerun()
        else:
            display_upload_images_step()
    
    elif st.session_state.workflow_step == "3. Analyze":
        if not st.session_state.current_patient:
            st.warning("⚠️ Please complete Patient Profile first")
            st.session_state.workflow_step = "1. Patient Profile"
            st.rerun()
        elif not hasattr(st.session_state, 'left_image') or not st.session_state.left_image:
            st.warning("⚠️ Please upload eye images first")
            st.session_state.workflow_step = "2. Upload Images"
            st.rerun()
        else:
            display_analyze_step(db, llm)
    
    elif st.session_state.workflow_step == "4. View Report":
        display_report_step(db, jg, pg)

# ============================================================================
# 10. RUN THE APP
# ============================================================================
if __name__ == "__main__":
    # Create required directories
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        st.warning(f"📁 Created '{MODEL_DIR}' directory. Please place your model files here.")
    
    # Create output directories
    for dir_name in ['json_outputs', 'reports']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    main()