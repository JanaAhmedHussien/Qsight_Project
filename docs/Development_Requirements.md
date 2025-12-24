# **Development Requirements for Modular Quantum-Classical Medical Imaging Pipeline**

## **System Architecture Overview**

A hybrid vision AI system for diabetic retinopathy diagnosis featuring:
1. **High-resolution classical pathway** for maximum diagnostic accuracy
2. **Compressed quantum-compatible pathway** for quantum advantage exploration
3. **Modular design** enabling independent training and deployment

## **Core Technical Requirements**

### **1. Vision Encoding Module**
```
Component: VisionEncoder
Input: Retinal fundus images (512×512×3 RGB)
Output: 2048-dimensional latent representation
Architecture Options:
  - UNet Encoder (pretrained on medical imaging)
  - Vision Transformer (ViT-B/16)
  - Hybrid CNN-Transformer
Requirements:
  - Output fixed dimension (2048D) regardless of architecture
  - Support for both pretrained and trainable modes
  - Gradient checkpointing for memory efficiency
```

### **2. Feature Compression Module**
```
Component: QuantumCompressor
Input: 2048D features from VisionEncoder
Output: 30-dimensional quantum-compatible features
Architecture:
  - Multi-layer perceptron with tanh activation
  - Bottleneck structure: 2048→512→128→30
  - Reconstruction decoder: 30→128→512→2048
Loss Functions:
  - Reconstruction loss: MSE(original_2048D, reconstructed_2048D)
  - Distribution regularization: KL divergence for Gaussian prior
  - Entropy minimization for feature sparsity
```

### **3. Classification Heads (Modular)**

#### **3.1 Classical Full-Resolution Head**
```
Component: ClassicalFullHead
Input: 2048D features (direct from encoder)
Output: 5-class severity probabilities
Architecture:
  - 3-layer MLP: 2048→1024→512→5
  - Batch normalization + ReLU activations
  - 0.3 dropout for regularization
Training: Jointly with encoder for maximum accuracy
```

#### **3.2 Classical Compressed Head**
```
Component: ClassicalCompressedHead  
Input: 30D compressed features
Output: 5-class severity probabilities
Architecture:
  - 2-layer MLP: 30→64→5
  - ReLU activation, no dropout (limited capacity)
Purpose: Fair comparison baseline for quantum head
```

#### **3.3 Quantum Classification Head**
```
Component: QuantumHead
Input: 30D compressed features (normalized to [-π, π])
Output: 5-class severity probabilities
Implementation Options:
  Primary: PennyLane (for gradient-based training)
  Secondary: Qiskit (for hardware compatibility)
Circuit Architecture:
  - Feature encoding: Angle embedding (RY gates)
  - Variational layers: 3 layers of entangling blocks
  - Measurements: Pauli-Z expectation values
  - Post-processing: Linear layer for class mapping
Qubit Strategy: Use all 30 features or select top-20 via attention
```

### **4. Training Strategies (Multiple Approaches Required)**

#### **4.1 Strategy A: Classical-First Cascaded Training**
```
Phase 1: Vision encoder + ClassicalFullHead
  - Train end-to-end with cross-entropy loss
  - Optimize for maximum diagnostic accuracy
  - Freeze encoder after convergence

Phase 2: QuantumCompressor + ClassicalCompressedHead  
  - Train compressor with reconstruction + classification loss
  - Keep encoder frozen
  - λ_recon = 0.3, λ_class = 0.7

Phase 3: QuantumHead training
  - Train on frozen compressed features
  - Separate optimization loop
  - Option: Use parameter shift or finite difference gradients
```

#### **4.2 Strategy B: Joint End-to-End Training**
```
Unified Loss Function:
  L_total = λ1·L_full + λ2·L_recon + λ3·L_comp + λ4·L_quant
  
Where:
  L_full = CE(ClassicalFullHead(encoder(x)), y)
  L_recon = MSE(reconstruct(compress(encoder(x))), encoder(x))  
  L_comp = CE(ClassicalCompressedHead(compress(encoder(x))), y)
  L_quant = CE(QuantumHead(compress(encoder(x))), y)

Training Procedure:
  - Initialize with Strategy A Phase 1
  - Gradually introduce quantum loss (anneal λ4)
  - Use gradient checkpointing for quantum simulations
  - Implement gradient accumulation for stability
```

#### **4.3 Strategy C: Frozen-Feature Quantum Adaptation**
```
Rationale: Quantum circuits may learn different functions than classical MLPs
           even on the same features. Frozen features force quantum head to
           adapt to existing representation rather than co-adapt with encoder.

Procedure:
  1. Generate compressed feature bank from trained Strategy A
  2. Train QuantumHead on fixed features with:
     - Multiple random initializations
     - Different circuit architectures
     - Advanced optimizers (AdamW, L-BFGS)
  3. Analyze what quantum learns vs classical on identical inputs
```

### **5. Ensemble and Fusion Module**
```
Component: DynamicEnsemble
Input: Predictions from all three heads + features
Output: Final severity classification
Fusion Strategies:
  1. Learned weighting: g(features) → [w_full, w_comp, w_quant]
  2. Confidence-based: Use prediction entropy for weighting
  3. Stacking: Train meta-classifier on all head outputs
  4. Calibration: Temperature scaling per head before fusion
```

## **Technical Implementation Details**

### **Quantum Simulation Requirements**
```
Primary Development: PennyLane with PyTorch interface
  - Reason: Better gradient support, faster simulation
  - Backend: 'default.qubit' for exact simulation
  - Gradient method: 'parameter-shift' for true quantum gradients

Hardware Compatibility: Qiskit interface
  - Export trained circuits to OpenQASM 2.0
  - Support for IBM Quantum backends
  - Noise model integration for robustness testing

Performance Optimizations:
  - Batch circuit execution when possible
  - Circuit cutting for >30 qubit simulations
  - Approximation methods for gradients (adjoint, SPSA)
```

### **Memory and Computation Constraints**
```
Feature Storage:
  - Cache encoder outputs to avoid recomputation
  - Store compressed features for multiple training runs

Gradient Management:
  - Separate optimizers for different components
  - Gradient checkpointing for quantum simulations
  - Mixed precision training (FP16) where applicable

Batch Strategy:
  - Classical training: Batch size 32-64
  - Quantum training: Batch size 8-16 (simulation memory)
  - Gradient accumulation for effective larger batches
```

## **Experimental Evaluation Framework**

### **Metrics Per Component**
```
1. Encoder Quality:
   - Reconstruction error (MSE, SSIM)
   - Feature separability (t-SNE visualization)
   - Transfer learning performance

2. Compression Module:
   - Reconstruction fidelity
   - Information bottleneck analysis
   - Quantum gate compatibility score

3. Classical Heads:
   - Accuracy, Precision, Recall, F1-score
   - AUC-ROC for each severity class
   - Calibration curves (ECE)

4. Quantum Head:
   - Same metrics as classical (for fair comparison)
   - Circuit depth and gate count
   - Parameter efficiency (accuracy/parameter ratio)
   - Hardware feasibility score

5. Ensemble:
   - Improvement over best single head
   - Diversity metrics (Q-statistics, disagreement)
   - Robustness to noisy inputs
```

### **Ablation Studies Required**
```
Study 1: Feature Dimension Impact
  - Compare 10D, 20D, 30D, 40D compression
  - Measure accuracy vs quantum hardware feasibility

Study 2: Training Strategy Comparison
  - Strategy A vs B vs C
  - Training time vs final accuracy
  - Quantum head performance variance

Study 3: Circuit Architecture Search
  - Variational layers: 1, 2, 3, 4
  - Entanglement patterns: linear, circular, all-to-all
  - Measurement strategies: single qubit, parity, swap test
```

## **Modularity and Deployment Requirements**

### **Interface Specifications**
```
Encoder Interface:
  def encode(image: Tensor) -> Tensor[2048]

Compressor Interface:  
  def compress(features: Tensor[2048]) -> Tensor[30]
  def get_reconstruction_loss(features, compressed) -> float

Head Interfaces (unified):
  def predict(features: Tensor) -> Tensor[5]
  def get_confidence(features) -> Tensor[5]

Quantum-Specific Interface:
  def get_circuit_qasm() -> str
  def get_expected_hardware_time() -> float
  def estimate_hardware_accuracy(backend) -> float
```

### **Deployment Scenarios**
```
Scenario 1: Classical-Only Deployment
  - Use Encoder + ClassicalFullHead
  - Maximum accuracy, real-time inference

Scenario 2: Quantum-Enhanced Deployment  
  - Use Encoder + Compressor + Ensemble
  - Quantum head contributes when available
  - Fallback to classical when quantum offline

Scenario 3: Pure Quantum Deployment
  - Use Compressor + QuantumHead
  - For quantum hardware demonstrations
  - Lower accuracy but showcases quantum advantage
```

## **Research Questions to Address**

### **Primary Research Questions**
```
1. Does quantum processing offer advantages over classical MLPs
   when both operate on identical compressed features?

2. Can quantum circuits learn different decision functions than
   classical models from the same feature representation?

3. Does end-to-end training improve quantum performance compared to
   training on frozen features?

4. What is the optimal compression dimension balancing:
   - Information preservation
   - Quantum hardware constraints
   - Classification accuracy
```

### **Hypotheses to Test**
```
H1: Quantum heads will show higher parameter efficiency
     (better accuracy per parameter) than classical MLPs

H2: Quantum and classical heads will make different errors
     suggesting complementary decision patterns

H3: End-to-end training will improve quantum performance by
     allowing feature representation to adapt to quantum processing

H4: There exists a "quantum sweet spot" compression dimension
     (20-30D) that maximizes quantum advantage
```

## **Implementation Priority Order**

### **Phase 1: Foundation (Week 1)**
```
Priority 1: Vision encoder + ClassicalFullHead
  - Train to state-of-the-art accuracy
  - Establish performance baseline
  - Create reproducible training pipeline

Priority 2: Feature compression module
  - Implement reconstruction-based training
  - Validate feature preservation metrics
  - Ensure proper normalization for quantum

Priority 3: ClassicalCompressedHead
  - Train on compressed features
  - Compare with full-resolution baseline
  - Analyze information loss
```

### **Phase 2: Quantum Integration (Week 2)**
```
Priority 4: QuantumHead with frozen features (Strategy C)
  - Quick implementation, immediate results
  - Baseline quantum performance
  - Circuit architecture experiments

Priority 5: Joint training experiments (Strategy B)
  - If time permits and simulations are fast enough
  - Test quantum gradient methods
  - Compare with frozen feature approach

Priority 6: Ensemble and fusion
  - Implement multiple fusion strategies
  - Measure ensemble improvements
  - Create demo visualization
```

### **Phase 3: Advanced Research (Post-Hackathon)**
```
If quantum hardware access available:
  - Deploy circuits to real quantum processors
  - Measure noise impact and error mitigation
  - Compare simulated vs real hardware results

If additional compute available:
  - Larger scale architecture search
  - More aggressive joint training
  - Multi-modal integration (clinical data + images)
```

## **Key Technical Decisions to Document**

1. **Encoder Architecture Choice**: Justify UNet vs ViT based on validation performance
2. **Compression Dimension**: Document why 30D was chosen over alternatives
3. **Quantum Circuit Design**: Explain layer count, entanglement pattern, measurement strategy
4. **Training Strategy Selection**: Justify cascaded vs joint training based on results
5. **Fusion Method**: Document why particular ensemble method was selected

## **Success Criteria**

### **Minimum Viable Success**
```
- ClassicalFullHead achieves >95% accuracy on validation set
- QuantumHead trains successfully on frozen features
- Compressed features preserve >80% of original information
- System demonstrates end-to-end functionality
```

### **Target Success**  
```
- QuantumHead achieves >90% of ClassicalCompressedHead accuracy
- Ensemble outperforms any single head
- Clear visualization of quantum decision boundaries
- Modular design enables multiple deployment scenarios
```

### **Stretch Goals**
```
- QuantumHead matches ClassicalCompressedHead accuracy
- End-to-end training improves quantum performance
- Successful deployment simulation on quantum hardware
- Novel insights about quantum vs classical learning patterns
```

This requirements document provides a comprehensive technical foundation while maintaining the modular, research-focused approach you requested. The emphasis on multiple training strategies addresses your concern about quantum adaptation to classically-optimized features, and the phased implementation ensures progress even with quantum simulation constraints.