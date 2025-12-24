# ============================================================================
# STREAMLIT DEPLOYMENT APP FOR DIABETIC RETINOPATHY CLASSIFICATION
# COMPATIBLE WITH THE UPDATED MODEL FROM COLAB
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

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Diabetic Retinopathy Classifier",
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
        width: 100%;
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
    'Proliferate_DR': '#F443366', # Red
    0: '#4CAF50',
    1: '#8BC34A',
    2: '#FFC107',
    3: '#FF9800',
    4: '#F44336'
}

# ============================================================================
# 3. MODEL COMPONENTS (MUST MATCH COLAB EXACTLY!)
# ============================================================================
class VisionEncoder(nn.Module):
    def __init__(self, encoder_type='vit', pretrained=False):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'vit':
            self.encoder = models.vit_b_16(pretrained=pretrained)
            self.encoder.heads = nn.Identity()
            self.projection = nn.Linear(768, 2048)
        else:
            resnet = models.resnet50(pretrained=pretrained)
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

# CRITICAL: Must match EXACTLY the QuantumSimulatedHead from Colab
class QuantumSimulatedHead(nn.Module):
    def __init__(self, input_dim=30, num_classes=5, quantum_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build quantum-inspired layers (EXACT MATCH to Colab)
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
        
        # Final classification layer
        layers.append(nn.Linear(32, num_classes))
        
        # CRITICAL: Attribute name MUST be 'quantum_sim'
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
            
            # Calculate agreement between heads
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
        
        # Get configuration from model_info
        self.classes = model_info.get('classes', ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'])
        self.compressed_dim = model_info.get('compressed_dim', 30)
        self.num_classes = len(self.classes)
        self.encoder_type = model_info.get('encoder_type', 'vit')
        self.ensemble_weights = model_info.get('ensemble_weights', [0.333, 0.333, 0.333])
        
        # Initialize components
        self.vision_encoder = VisionEncoder(encoder_type=self.encoder_type, pretrained=False)
        self.compression = CompressionModule(input_dim=2048, compressed_dim=self.compressed_dim)
        self.classical_head_a = ClassicalHeadA(input_dim=2048, num_classes=self.num_classes)
        self.classical_head_b = ClassicalHeadB(input_dim=self.compressed_dim, num_classes=self.num_classes)
        
        # CRITICAL: QuantumSimulatedHead must match training architecture
        self.quantum_head = QuantumSimulatedHead(input_dim=self.compressed_dim, num_classes=self.num_classes)
        
        # Ensemble
        self.ensemble = DynamicEnsemble(num_heads=3)
        
        # Initialize with trained ensemble weights if available
        if 'ensemble_weights' in model_info:
            with torch.no_grad():
                self.ensemble.base_weights.data = torch.tensor(model_info['ensemble_weights'])
        
        # Uncertainty estimators
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
        # Encode and compress
        latent_features = self.vision_encoder(x)
        compressed_features = self.compression(latent_features)
        
        # Get predictions
        output_a = self.classical_head_a(latent_features)
        output_b = self.classical_head_b(compressed_features)
        output_c = self.quantum_head(compressed_features)
        
        # Estimate uncertainties
        unc_a = self.uncertainty_a(latent_features)
        unc_b = self.uncertainty_b(compressed_features)
        unc_c = self.uncertainty_c(compressed_features)
        
        uncertainties = torch.cat([unc_a, unc_b, unc_c], dim=1).squeeze()
        
        # Dynamic ensemble
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
        # Define paths
        model_path = os.path.join(MODEL_DIR, 'phase1_classical_model.pth')
        info_path = os.path.join(MODEL_DIR, 'model_info.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Please copy 'phase1_classical_model.pth' from Colab to the 'trained_model' folder")
            return None, None, None
        
        if not os.path.exists(info_path):
            st.error(f"❌ Model info file not found at: {info_path}")
            st.info("Please copy 'model_info.pkl' from Colab to the 'trained_model' folder")
            return None, None, None
        
        # Load model info
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # Create model
        model = HybridDRModel(model_info)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        
        # Set to evaluation mode
        model.eval()
        
        # Get class names and ensemble weights
        class_names = model_info.get('classes', ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'])
        ensemble_weights = model_info.get('ensemble_weights', [0.333, 0.333, 0.333])
        
        return model, class_names, ensemble_weights
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure you have the correct model files from Colab")
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
    
    # Get final prediction
    probs = outputs['probabilities'].squeeze()
    prediction_idx = torch.argmax(probs).item()
    confidence = probs[prediction_idx].item()
    
    # Get individual head predictions
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

# ============================================================================
# 6. MAIN STREAMLIT APP
# ============================================================================
def main():
    # Title and description
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Hybrid Classical-Quantum Deep Learning Model with Dynamic Ensemble")
    st.markdown("Upload a retinal fundus image for automated classification of diabetic retinopathy severity.")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'ensemble_weights' not in st.session_state:
        st.session_state.ensemble_weights = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🚀 Quick Start")
        
        # Load model button
        if st.button("🔄 Load Model", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                model, class_names, ensemble_weights = load_model()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.class_names = class_names
                    st.session_state.ensemble_weights = ensemble_weights
                    st.success(f"✅ Model loaded!")
                    st.info(f"Classes: {class_names}")
                    st.info(f"Ensemble Weights: {ensemble_weights}")
                else:
                    st.error("❌ Failed to load model")
        
        st.markdown("---")
        st.markdown("## 📁 File Status")
        
        # Check required files
        required_files = [
            ('phase1_classical_model.pth', 'Main Model'),
            ('model_info.pkl', 'Model Info'),
        ]
        
        all_files_exist = True
        for file_name, description in required_files:
            file_path = os.path.join(MODEL_DIR, file_name)
            if os.path.exists(file_path):
                st.success(f"✅ {description}")
            else:
                st.error(f"❌ {description}")
                all_files_exist = False
        
        if not all_files_exist:
            st.warning("""
            **Missing files detected!**
            
            Steps to fix:
            1. Train model in Colab
            2. Download model files
            3. Create 'trained_model' folder
            4. Copy files to the folder
            
            Required files:
            - phase1_classical_model.pth
            - model_info.pkl
            """)
        
        st.markdown("---")
        st.markdown("## 📸 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a retinal image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a retinal fundus image for classification",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.session_state.uploaded_image = image
                
                # Display preview
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Predict button
                if st.session_state.model is not None:
                    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                        with st.spinner("Analyzing image..."):
                            input_tensor = preprocess_image(image)
                            result = predict_image(st.session_state.model, input_tensor)
                            st.session_state.prediction_result = result
                            st.rerun()
                else:
                    st.warning("⚠️ Please load the model first.")
                    
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.info("""
        **Model Architecture:**
        - Vision Transformer Encoder
        - Feature Compression (2048D → 30D)
        - 3 Parallel Heads (2 Classical, 1 Quantum-inspired)
        - Dynamic Ensemble with Learnable Weights
        
        **Classes:**
        0: No DR (Normal)
        1: Mild DR
        2: Moderate DR
        3: Severe DR
        4: Proliferative DR
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">Input Image</h3>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_image is not None:
            st.image(st.session_state.uploaded_image, use_column_width=True)
            
            # Image info
            img_array = np.array(st.session_state.uploaded_image)
            st.caption(f"📐 Image size: {img_array.shape[1]}×{img_array.shape[0]} pixels")
        else:
            st.info("👈 Upload an image using the sidebar uploader")
    
    with col2:
        st.markdown('<h3 class="sub-header">Analysis Results</h3>', unsafe_allow_html=True)
        
        if st.session_state.model is None:
            st.warning("⚠️ Model not loaded. Click 'Load Model' in the sidebar.")
        
        elif st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            class_names = st.session_state.class_names
            
            # Get prediction
            pred_idx = result['final_prediction']
            pred_class = class_names[pred_idx]
            confidence = result['final_confidence']
            
            # Color for prediction box
            color = CLASS_COLORS.get(pred_idx, '#2E86AB')
            
            # Display final prediction
            st.markdown(f"""
            <div class="prediction-box" style="border-left-color: {color};">
                <h4 style="color: #2E86AB; margin-top: 0;">Final Prediction</h4>
                <h1 style="color: {color}; margin: 10px 0;">{pred_class}</h1>
                <p style="font-size: 1.1rem;">Confidence: <strong>{confidence:.2%}</strong></p>
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; margin: 10px 0;">
                    <div style="width: {confidence*100}%; background-color: {color}; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual head predictions
            st.markdown("### 🤖 Individual Head Predictions")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown('<div class="head-card">', unsafe_allow_html=True)
                st.markdown("**Classical Head A**")
                st.metric("Prediction", class_names[result['classical_a_pred']])
                st.caption(f"⚖️ Weight: {result['ensemble_weights'][0]:.3f}")
                st.caption(f"🎯 Uncertainty: {result['uncertainties'][0]:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="head-card">', unsafe_allow_html=True)
                st.markdown("**Classical Head B**")
                st.metric("Prediction", class_names[result['classical_b_pred']])
                st.caption(f"⚖️ Weight: {result['ensemble_weights'][1]:.3f}")
                st.caption(f"🎯 Uncertainty: {result['uncertainties'][1]:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="head-card">', unsafe_allow_html=True)
                st.markdown("**Quantum Head**")
                st.metric("Prediction", class_names[result['quantum_pred']])
                st.caption(f"⚖️ Weight: {result['ensemble_weights'][2]:.3f}")
                st.caption(f"🎯 Uncertainty: {result['uncertainties'][2]:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Probability distribution
            st.markdown("### 📊 Probability Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(class_names, result['probabilities'], 
                         color=[CLASS_COLORS.get(i, '#2E86AB') for i in range(len(class_names))])
            
            ax.set_ylabel('Probability')
            ax.set_ylim([0, 1])
            ax.set_title('Class Probabilities')
            ax.grid(True, alpha=0.3)
            
            # Add probability values on bars
            for bar, prob in zip(bars, result['probabilities']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            
            # Ensemble weights visualization
            st.markdown("### ⚖️ Ensemble Weight Distribution")
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            head_names = ['Classical A', 'Classical B', 'Quantum']
            colors = ['#2E86AB', '#1a5276', '#0d3b5c']
            
            ax2.pie(result['ensemble_weights'], labels=head_names, colors=colors,
                   autopct='%1.1f%%', startangle=90, shadow=True)
            ax2.axis('equal')
            
            st.pyplot(fig2)
            
            # Show trained ensemble weights from model info
            if st.session_state.ensemble_weights is not None:
                st.markdown("### 🎯 Trained Ensemble Weights")
                trained_weights = np.array(st.session_state.ensemble_weights)
                current_weights = result['ensemble_weights']
                
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                x = np.arange(len(head_names))
                width = 0.35
                
                ax3.bar(x - width/2, trained_weights, width, label='Trained Weights', color='#2E86AB')
                ax3.bar(x + width/2, current_weights, width, label='Current Weights', color='#4CAF50')
                
                ax3.set_ylabel('Weight')
                ax3.set_title('Trained vs Current Ensemble Weights')
                ax3.set_xticks(x)
                ax3.set_xticklabels(head_names)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                st.pyplot(fig3)
        
        elif st.session_state.model is not None and st.session_state.uploaded_image is not None:
            st.info("📸 Image uploaded. Click 'Analyze Image' to get predictions.")
        
        elif st.session_state.model is not None:
            st.success("✅ Model loaded and ready!")
            st.info("Upload an image to begin analysis.")
            
            # Show ensemble weights info
            if st.session_state.ensemble_weights is not None:
                weights = st.session_state.ensemble_weights
                st.markdown("### ⚖️ Trained Ensemble Weights")
                st.write(f"**Classical A:** {weights[0]:.3f}")
                st.write(f"**Classical B:** {weights[1]:.3f}")
                st.write(f"**Quantum:** {weights[2]:.3f}")
                
                # Check if weights are dynamic (not equal)
                if abs(weights[0] - weights[1]) > 0.1 or abs(weights[0] - weights[2]) > 0.1:
                    st.success("🎯 Dynamic ensemble is working! Weights are not equal.")
                else:
                    st.warning("⚠️ Weights are equal. The ensemble may not have learned properly.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>Hybrid Classical-Quantum Deep Learning Model for Diabetic Retinopathy Classification</strong></p>
        <p>⚠️ <em>This tool is for research and educational purposes only. Not for clinical diagnosis.</em></p>
        <p>Model uses dynamic ensemble learning with adaptive weight allocation.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 7. RUN THE APP
# ============================================================================
if __name__ == "__main__":
    # Check if trained_model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        st.warning(f"📁 Created '{MODEL_DIR}' directory. Please place your model files here.")
    
    main()