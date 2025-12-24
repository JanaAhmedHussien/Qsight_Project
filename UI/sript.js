// Diabetic Retinopathy Detection UI - JavaScript
// TODO: Replace mock functions with actual backend integration

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const themeToggle = document.getElementById('themeToggle');
    const imageInput = document.getElementById('imageInput');
    const uploadArea = document.getElementById('uploadArea');
    const browseBtn = document.getElementById('browseBtn');
    const previewImage = document.getElementById('previewImage');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const previewContainer = document.getElementById('previewContainer');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileDimensions = document.getElementById('fileDimensions');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const errorMessage = document.getElementById('errorMessage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const sampleBtn = document.getElementById('sampleBtn');
    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const loadingContainer = document.getElementById('loadingContainer');
    const loadingDetail = document.getElementById('loadingDetail');
    const resultsContainer = document.getElementById('resultsContainer');
    const noResults = document.getElementById('noResults');
    const resultsTemplate = document.getElementById('resultsTemplate');
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityValue = document.getElementById('opacityValue');
    const segmentationOverlay = document.getElementById('segmentationOverlay');
    const originalImage = document.getElementById('originalImage');
    const segmentedImage = document.getElementById('segmentedImage');
    const originalImagePlaceholder = document.getElementById('originalImagePlaceholder');
    const segmentedImagePlaceholder = document.getElementById('segmentedImagePlaceholder');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const emailInput = document.getElementById('emailInput');
    const emailError = document.getElementById('emailError');
    const sendReportBtn = document.getElementById('sendReportBtn');
    const toastContainer = document.getElementById('toastContainer');
    const menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    const startUploadBtn = document.getElementById('startUploadBtn');
    
    // State variables
    let currentImage = null;
    let analysisInProgress = false;
    let currentTheme = localStorage.getItem('theme') || 'light';
    let currentResults = null;
    
    // Initialize the application
    function init() {
        // Set theme
        document.documentElement.setAttribute('data-theme', currentTheme);
        
        // Set current date in report
        document.getElementById('reportDate').textContent = new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        
        document.getElementById('screeningDate').textContent = new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        
        // Event listeners
        setupEventListeners();
        
        // Initialize the opacity slider
        updateSegmentationOpacity();
    }
    
    // Set up all event listeners
    function setupEventListeners() {
        // Theme toggle
        themeToggle.addEventListener('click', toggleTheme);
        
        // Image upload
        browseBtn.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', handleImageUpload);
        
        // Drag and drop
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        
        // Remove image
        removeImageBtn.addEventListener('click', removeImage);
        
        // Analysis controls
        analyzeBtn.addEventListener('click', startAnalysis);
        resetBtn.addEventListener('click', resetAnalysis);
        sampleBtn.addEventListener('click', downloadSampleImage);
        
        // Segmentation opacity control
        if (opacitySlider) {
            opacitySlider.addEventListener('input', updateSegmentationOpacity);
        }
        
        // XAI tabs
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => switchTab(btn.dataset.tab));
        });
        
        // Email validation
        emailInput.addEventListener('input', validateEmail);
        sendReportBtn.addEventListener('click', sendReport);
        
        // Mobile menu toggle
        menuToggle.addEventListener('click', toggleMobileMenu);
        
        // Start upload button
        startUploadBtn.addEventListener('click', () => {
            document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
            // Simulate click on browse button after a short delay
            setTimeout(() => browseBtn.click(), 300);
        });
        
        // Learn more button
        document.getElementById('learnMoreBtn').addEventListener('click', () => {
            showToast('Info', 'This demo shows the UI for a Diabetic Retinopathy Detection system. Backend AI integration is under development.', 'info');
        });
        
        // Close mobile menu when clicking on a link
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('show');
            });
        });
    }
    
    // Theme toggle functionality
    function toggleTheme() {
        currentTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', currentTheme);
        localStorage.setItem('theme', currentTheme);
    }
    
    // Handle image upload via file input
    function handleImageUpload(e) {
        const file = e.target.files[0];
        if (file) {
            validateAndLoadImage(file);
        }
    }
    
    // Handle drag over event
    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    }
    
    // Handle drag leave event
    function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    }
    
    // Handle drop event
    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file) {
            validateAndLoadImage(file);
        }
    }
    
    // Validate and load image file
    function validateAndLoadImage(file) {
        // Reset error message
        hideError();
        
        // Check file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            showError('Invalid file type. Please upload a JPG, JPEG, or PNG image.');
            return;
        }
        
        // Check file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            showError('File size exceeds 10MB limit. Please upload a smaller image.');
            return;
        }
        
        // Load and preview image
        const reader = new FileReader();
        reader.onload = function(e) {
            currentImage = {
                src: e.target.result,
                file: file,
                name: file.name,
                size: formatFileSize(file.size)
            };
            
            // Display preview
            previewImage.src = currentImage.src;
            previewImage.style.display = 'block';
            previewPlaceholder.style.display = 'none';
            
            // Display file info
            fileName.textContent = currentImage.name;
            fileSize.textContent = currentImage.size;
            
            // Get image dimensions
            const img = new Image();
            img.onload = function() {
                fileDimensions.textContent = `${img.width} × ${img.height}px`;
                currentImage.width = img.width;
                currentImage.height = img.height;
                
                // Store for use in results
                originalImage.src = currentImage.src;
                segmentedImage.src = currentImage.src;
            };
            img.src = currentImage.src;
            
            // Enable analyze button
            analyzeBtn.disabled = false;
            
            // Scroll to preview section
            previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };
        
        reader.readAsDataURL(file);
    }
    
    // Remove uploaded image
    function removeImage() {
        currentImage = null;
        previewImage.src = '';
        previewImage.style.display = 'none';
        previewPlaceholder.style.display = 'flex';
        fileName.textContent = '-';
        fileSize.textContent = '-';
        fileDimensions.textContent = '-';
        analyzeBtn.disabled = true;
        
        // Clear results if any
        resetAnalysis();
    }
    
    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.add('show');
        
        // Auto-hide error after 5 seconds
        setTimeout(hideError, 5000);
    }
    
    // Hide error message
    function hideError() {
        errorMessage.classList.remove('show');
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Start analysis process
    function startAnalysis() {
        if (!currentImage || analysisInProgress) return;
        
        analysisInProgress = true;
        
        // Disable analyze button during analysis
        analyzeBtn.disabled = true;
        
        // Show progress section and loading
        progressSection.style.display = 'block';
        loadingContainer.classList.add('show');
        
        // Reset progress
        updateProgress(0);
        
        // Update progress steps
        updateProgressSteps(0);
        
        // Simulate analysis process with different stages
        simulateAnalysisProgress();
    }
    
    // Simulate analysis progress
    function simulateAnalysisProgress() {
        // Stage 1: Upload and preprocessing (0-15%)
        loadingDetail.textContent = 'Uploading and preprocessing image...';
        simulateProgress(15, 1000);
        
        // Stage 2: Quantum ML inference (15-45%)
        setTimeout(() => {
            loadingDetail.textContent = 'Running Quantum Neural Network inference...';
            simulateProgress(45, 2000);
        }, 1200);
        
        // Stage 3: Image segmentation (45-70%)
        setTimeout(() => {
            loadingDetail.textContent = 'Segmenting retinal lesions...';
            simulateProgress(70, 1500);
        }, 3400);
        
        // Stage 4: XAI analysis (70-90%)
        setTimeout(() => {
            loadingDetail.textContent = 'Generating explainable AI insights...';
            simulateProgress(90, 1800);
        }, 5200);
        
        // Stage 5: Finalizing (90-100%)
        setTimeout(() => {
            loadingDetail.textContent = 'Finalizing results and generating report...';
            simulateProgress(100, 1000);
            
            // Show results
            setTimeout(() => {
                completeAnalysis();
            }, 1200);
        }, 7200);
    }
    
    // Simulate progress increment
    function simulateProgress(targetPercent, duration) {
        const startPercent = parseInt(progressFill.style.width) || 0;
        const increment = (targetPercent - startPercent) / (duration / 50); // Update every 50ms
        let current = startPercent;
        
        const interval = setInterval(() => {
            current += increment;
            if (current >= targetPercent) {
                current = targetPercent;
                clearInterval(interval);
            }
            updateProgress(current);
        }, 50);
    }
    
    // Update progress bar and percentage
    function updateProgress(percent) {
        const roundedPercent = Math.round(percent);
        progressFill.style.width = `${roundedPercent}%`;
        progressPercent.textContent = `${roundedPercent}%`;
        
        // Update progress steps
        updateProgressSteps(roundedPercent);
    }
    
    // Update which progress step is active
    function updateProgressSteps(percent) {
        const steps = document.querySelectorAll('.step');
        let activeIndex = 0;
        
        if (percent >= 15) activeIndex = 1;
        if (percent >= 45) activeIndex = 2;
        if (percent >= 70) activeIndex = 3;
        if (percent >= 90) activeIndex = 4;
        
        steps.forEach((step, index) => {
            if (index <= activeIndex) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });
    }
    
    // Complete analysis and show results
    function completeAnalysis() {
        analysisInProgress = false;
        
        // Hide loading
        loadingContainer.classList.remove('show');
        
        // Generate mock results
        generateMockResults();
        
        // Show results section
        noResults.style.display = 'none';
        
        // Clone and display results template
        const resultsClone = resultsTemplate.cloneNode(true);
        resultsClone.id = 'resultsDisplay';
        resultsClone.hidden = false;
        resultsContainer.appendChild(resultsClone);
        
        // Scroll to results
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        
        // Set up results interaction
        setupResultsInteractions();
        
        // Re-enable analyze button
        analyzeBtn.disabled = false;
    }
    
    // Generate mock results for demonstration
    function generateMockResults() {
        // Randomly determine if DR is detected (30% chance)
        const hasDR = Math.random() < 0.3;
        
        // Generate confidence score (higher if no DR)
        const confidence = hasDR ? 
            Math.floor(Math.random() * 30) + 70 : // 70-99% if DR detected
            Math.floor(Math.random() * 25) + 75;  // 75-99% if no DR
        
        // Store results for use in report
        currentResults = {
            hasDR: hasDR,
            confidence: confidence,
            diagnosis: hasDR ? 'Diabetic Retinopathy Detected' : 'No Diabetic Retinopathy',
            severity: hasDR ? (confidence > 85 ? 'Moderate' : 'Mild') : 'None'
        };
    }
    
    // Set up interactions for results section
    function setupResultsInteractions() {
        // Get elements from the cloned results
        const resultBadge = document.querySelector('#resultsDisplay #resultBadge');
        const confidenceValue = document.querySelector('#resultsDisplay #confidenceValue');
        const meterFill = document.querySelector('#resultsDisplay #meterFill');
        const reportDiagnosis = document.getElementById('reportDiagnosis');
        
        if (currentResults) {
            // Update classification badge
            if (currentResults.hasDR) {
                resultBadge.classList.add('detected');
                resultBadge.querySelector('.badge-label').textContent = 'Diabetic Retinopathy Detected';
                resultBadge.querySelector('.badge-confidence').textContent = `Confidence: ${currentResults.confidence}%`;
                
                // Update report diagnosis
                if (reportDiagnosis) {
                    reportDiagnosis.classList.add('detected');
                    reportDiagnosis.querySelector('.badge-label').textContent = 'Diabetic Retinopathy Detected';
                    reportDiagnosis.querySelector('.badge-confidence').textContent = `Confidence: ${currentResults.confidence}%`;
                }
            } else {
                resultBadge.querySelector('.badge-label').textContent = 'No Diabetic Retinopathy';
                resultBadge.querySelector('.badge-confidence').textContent = `Confidence: ${currentResults.confidence}%`;
                
                // Update report diagnosis
                if (reportDiagnosis) {
                    reportDiagnosis.querySelector('.badge-label').textContent = 'No Diabetic Retinopathy';
                    reportDiagnosis.querySelector('.badge-confidence').textContent = `Confidence: ${currentResults.confidence}%`;
                }
            }
            
            // Update confidence display
            confidenceValue.textContent = `${currentResults.confidence}%`;
            
            // Update confidence meter with animation
            setTimeout(() => {
                meterFill.style.width = `${currentResults.confidence}%`;
            }, 500);
            
            // Update report diagnosis details
            const diagnosisDetails = document.querySelector('.diagnosis-details');
            if (diagnosisDetails && currentResults.hasDR) {
                diagnosisDetails.innerHTML = `
                    <p>Signs of diabetic retinopathy detected with ${currentResults.confidence}% confidence. The model identified microvascular abnormalities consistent with ${currentResults.severity.toLowerCase()} diabetic retinopathy.</p>
                    <p>Recommendation: Refer to ophthalmologist for comprehensive dilated eye examination and further evaluation. Consider frequency of follow-up based on severity.</p>
                `;
            }
            
            // Show segmentation images
            if (originalImage && segmentedImage) {
                originalImage.style.display = 'block';
                segmentedImage.style.display = 'block';
                segmentationOverlay.style.display = 'block';
                originalImagePlaceholder.style.display = 'none';
                segmentedImagePlaceholder.style.display = 'none';
            }
        }
        
        // Set up opacity slider in results
        const resultsOpacitySlider = document.querySelector('#resultsDisplay #opacitySlider');
        const resultsOpacityValue = document.querySelector('#resultsDisplay #opacityValue');
        
        if (resultsOpacitySlider) {
            resultsOpacitySlider.addEventListener('input', function() {
                resultsOpacityValue.textContent = this.value;
                segmentationOverlay.style.opacity = this.value / 100;
            });
        }
        
        // Update segmentation opacity
        updateSegmentationOpacity();
    }
    
    // Update segmentation overlay opacity
    function updateSegmentationOpacity() {
        if (opacitySlider && opacityValue && segmentationOverlay) {
            opacityValue.textContent = opacitySlider.value;
            segmentationOverlay.style.opacity = opacitySlider.value / 100;
        }
    }
    
    // Switch XAI tabs
    function switchTab(tabId) {
        // Update tab buttons
        tabBtns.forEach(btn => {
            if (btn.dataset.tab === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        // Update tab contents
        tabContents.forEach(content => {
            if (content.id === tabId) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
        
        // Animate feature importance bars when feature tab is selected
        if (tabId === 'feature') {
            setTimeout(() => {
                const bars = document.querySelectorAll('.chart-bar');
                bars.forEach(bar => {
                    const height = bar.getAttribute('data-value');
                    bar.style.height = `${height}%`;
                });
            }, 100);
        }
    }
    
    // Validate email address
    function validateEmail() {
        const email = emailInput.value.trim();
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        
        if (email === '') {
            emailError.textContent = 'Email address is required';
            emailError.style.display = 'block';
            return false;
        } else if (!emailRegex.test(email)) {
            emailError.textContent = 'Please enter a valid email address';
            emailError.style.display = 'block';
            return false;
        } else {
            emailError.style.display = 'none';
            return true;
        }
    }
    
    // Send report (UI only - demo)
    function sendReport() {
        if (!validateEmail()) return;
        
        if (!currentResults) {
            showToast('Error', 'Please analyze an image before sending a report.', 'error');
            return;
        }
        
        // Show sending state
        const originalText = sendReportBtn.innerHTML;
        sendReportBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
        sendReportBtn.disabled = true;
        
        // Simulate sending process
        setTimeout(() => {
            // Reset button
            sendReportBtn.innerHTML = originalText;
            sendReportBtn.disabled = false;
            
            // Show success toast
            showToast('Success', `Report sent successfully to ${emailInput.value}`, 'success');
            
            // Clear email field
            emailInput.value = '';
        }, 2000);
    }
    
    // Reset analysis and results
    function resetAnalysis() {
        // Reset state
        analysisInProgress = false;
        currentResults = null;
        
        // Hide progress and loading
        progressSection.style.display = 'none';
        loadingContainer.classList.remove('show');
        updateProgress(0);
        
        // Remove results display if exists
        const resultsDisplay = document.getElementById('resultsDisplay');
        if (resultsDisplay) {
            resultsDisplay.remove();
        }
        
        // Show no results message
        noResults.style.display = 'block';
        
        // Enable analyze button if image is uploaded
        analyzeBtn.disabled = !currentImage;
        
        // Reset email error
        emailError.style.display = 'none';
        
        // Reset segmentation placeholders
        if (originalImagePlaceholder && segmentedImagePlaceholder) {
            originalImagePlaceholder.style.display = 'flex';
            segmentedImagePlaceholder.style.display = 'flex';
        }
        
        if (originalImage && segmentedImage) {
            originalImage.style.display = 'none';
            segmentedImage.style.display = 'none';
        }
        
        if (segmentationOverlay) {
            segmentationOverlay.style.display = 'none';
        }
    }
    
    // Download sample image
    function downloadSampleImage() {
        // Create a sample image (in real app, this would be a real retina image)
        // For demo, we'll create a simple canvas image
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 800;
        const ctx = canvas.getContext('2d');
        
        // Draw a simple retina-like image
        // Background
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, 800, 800);
        
        // Optic disc
        ctx.beginPath();
        ctx.arc(400, 400, 100, 0, Math.PI * 2);
        ctx.fillStyle = '#f0f0f0';
        ctx.fill();
        
        // Blood vessels
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 8;
        
        // Draw some vessel-like lines
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            const x1 = 400 + Math.cos(angle) * 100;
            const y1 = 400 + Math.sin(angle) * 100;
            const length = 150 + Math.random() * 100;
            const x2 = 400 + Math.cos(angle) * length;
            const y2 = 400 + Math.sin(angle) * length;
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        
        // Convert to data URL and trigger download
        const dataUrl = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = 'sample_retina_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Show toast notification
        showToast('Sample Downloaded', 'A sample retina image has been downloaded. You can upload it for analysis.', 'success');
    }
    
    // Toggle mobile menu
    function toggleMobileMenu() {
        navLinks.classList.toggle('show');
    }
    
    // Show toast notification
    function showToast(title, message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        // Show toast
        setTimeout(() => {
            toast.classList.add('show');
        }, 10);
        
        // Auto-remove toast after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 5000);
    }
    
    // Initialize the application
    init();
    
    // Log initialization for debugging
    console.log('Diabetic Retinopathy Detection UI initialized');
    console.log('This is a UI demonstration prototype. Backend AI integration points are marked with TODO comments.');
});