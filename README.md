##  The Problem: Preventing Diabetic Blindness

**Diabetic Retinopathy (DR)** is the leading cause of blindness among working-age adults worldwide. Early detection can prevent 95% of severe vision loss, but current challenges include:

- **Limited screening access**: 50% of diabetics don't get regular eye exams
- **Diagnostic delays**: Manual review by specialists takes days to weeks
- **Resource inequality**: 80% of ophthalmologists serve only 20% of the population
- **Progression risk**: DR progresses silently until vision loss occurs

##  Our Vision: Quantum-Accelerated Screening

**Q-Sight** aims to revolutionize diabetic retinopathy screening through quantum-enhanced artificial intelligence. Our vision is to create an accessible system that provides hospital-grade accuracy anywhere, anytime.

### **Core Innovation Goals**
- **Quantum-optimized feature selection**: Intelligently reduce 512+ image features to critical biomarkers
- **Hybrid quantum-classical architecture**: Leverage quantum advantages while maintaining classical reliability
- **Accessible deployment**: Cloud-based solution requiring minimal local infrastructure

##  Project Objectives

### **Primary Goals**
1. **Accuracy Target**: Achieve 92-95% classification accuracy on the APTOS 2019 dataset
2. **Quantum Advantage**: Demonstrate measurable improvement over classical-only approaches
3. **Clinical Utility**: Create an intuitive interface for healthcare professionals

### **Technical Milestones**
- Phase 1: Classical baseline establishment (Days 1-3)
- Phase 2: Quantum integration prototyping (Days 4-8)
- Phase 3: System integration & validation (Days 9-12)
- Phase 4: Presentation preparation (Days 13-14)

##  Technology Strategy

| Component | Technology Choice | Rationale |
|-----------|------------------|-----------|
| **Quantum Framework** | Qiskit (IBM Quantum) | Industry standard with excellent simulator support |
| **Machine Learning** | PyTorch | Flexibility for custom quantum-classical integration |
| **Dashboard** | Streamlit | Rapid prototyping for medical interfaces |
| **Deployment** | Docker | Ensures reproducibility across environments |

## Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Classification Accuracy** | >92% | 5-fold cross validation |
| **Quantum Advantage** | >5% improvement | Comparison with classical baseline |
| **Processing Speed** | <5 minutes per image | End-to-end timing benchmarks |
| **Clinical Relevance** | High sensitivity (>90%) | Medical validation metrics |

##  Team Structure & Responsibilities

| Role | Team Member(s) | Key Focus Areas |
|------|----------------|----------------|
| **Quantum AI Engineer** | Noor and Jana | Quantum circuit design, hybrid integration |
| **ML Engineer** | Bryan Gracia and Jana Ahmed | Classical pipelines, model optimization |
| **Full-Stack Developer** | Bryan Gracia | Dashboard development, deployment strategy |

##  Research Approach

Our methodology follows these key principles:

1. **Progressive Integration**: Start classical, add quantum enhancements incrementally
2. **Comparative Validation**: Maintain parallel baselines for performance comparison
3. **Medical Validation**: Prioritize clinically relevant metrics over pure accuracy
4. **Accessibility Focus**: Design for deployment in resource-limited settings

##  Repository Structure (Proposed)

Qsight_Project/
├── docs/ # Documentation & planning
├── notebooks/ # Experimental notebooks
│ ├── exploration/ # Data exploration
│ ├── classical/ # Classical model experiments
│ └── quantum/ # Quantum integration tests
├── src/ # Source code (to be developed)
├── data/ # Dataset storage
├── requirements.txt # Dependencies
└── README.md # This file


##  Key Resources

### **Primary Dataset**
- **APTOS 2019 Blindness Detection**: ~3,662 retinal images with DR severity grades

### **Research References**
- Quantum transfer learning for medical imaging
- Quantum-enhanced optimization techniques
- Diabetic retinopathy classification benchmarks

### **Development Tools**
- IBM Quantum Experience (free tier)
- GitHub for version control

##  Contributing

This project is currently in the planning phase for a hackathon. We welcome:
- Feedback on our technical approach
- Suggestions for quantum optimization techniques
- Ideas for clinical validation metrics
