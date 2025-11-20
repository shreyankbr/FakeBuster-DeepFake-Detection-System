// --- Global State ---
let currentUser = null;
let analysisHistory = [];
let isAnalyzing = false;
let analysisChart = null;
let isTrialMode = false;

// --- App Initialization ---
document.addEventListener('DOMContentLoaded', initializeApp);

function initializeApp() {
    document.body.style.visibility = "hidden";

    // Check for trial mode in URL
    const urlParams = new URLSearchParams(window.location.search);
    isTrialMode = urlParams.get('mode') === 'trial';

    if (isTrialMode) {
        // Trial mode - no user data loading
        currentUser = null;
        analysisHistory = [];
        document.body.style.visibility = "visible";
        setupEventListeners();
        initAnimations();
        updatePageUI();
        showTrialModeIndicator();
        return;
    }

    const cachedUser = JSON.parse(localStorage.getItem("fb_currentUser"));

    if (cachedUser) {
        currentUser = cachedUser;
        updateNavigation();
    }

    const auth = firebase.auth();
    const firestore = firebase.firestore();

    auth.onAuthStateChanged((user) => {
        document.body.style.visibility = "visible";

        if (user) {
            currentUser = {
                uid: user.uid,
                email: user.email,
                name: user.displayName || user.email.split('@')[0]
            };
            localStorage.setItem("fb_currentUser", JSON.stringify(currentUser));
            loadUserHistory();
        } else {
            currentUser = null;
            localStorage.removeItem("fb_currentUser");
            analysisHistory = [];
        }
        updatePageUI();
        checkAuthState();
    });

    setupEventListeners();
    initAnimations();
    updatePageUI();
}

function showTrialModeIndicator() {
    const indicator = document.getElementById('trialModeIndicator');
    if (indicator) {
        indicator.style.display = 'block';
    }

    updateNavigation();
}

// Firestore functions for permanent history storage
async function loadUserHistory() {
    if (!currentUser) return;

    console.log('🔄 Loading user history for:', currentUser.uid);

    const cached = localStorage.getItem(`analysisHistory_${currentUser.uid}`);
    if (cached) {
        try {
            analysisHistory = JSON.parse(cached);
            console.log('📁 Loaded from localStorage:', analysisHistory.length, 'items');
            updatePageUI();
        } catch (e) {
            console.error('Error parsing cached history:', e);
            analysisHistory = [];
        }
    }

    try {
        const snapshot = await firebase.firestore()
            .collection('userAnalyses')
            .doc(currentUser.uid)
            .collection('analyses')
            .orderBy('timestamp', 'desc')
            .limit(100)
            .get();

        if (!snapshot.empty) {
            const firestoreHistory = snapshot.docs.map(doc => {
                const data = doc.data();
                console.log('📄 Firestore doc data:', data);

                // Ensure all required fields are present
                return {
                    id: doc.id,
                    filename: data.filename || 'Unknown File',
                    timestamp: data.timestamp || data.savedAt?.toDate?.()?.toISOString() || new Date().toISOString(),
                    type: data.type || 'unknown',
                    isReal: data.isReal !== undefined ? data.isReal : true,
                    confidence: data.confidence || 0,
                    probability: data.probability || 0,
                    heatmap_base64: data.heatmap_base64 || null,
                    frame_count: data.frame_count || 1
                };
            });

            console.log('🔥 Loaded from Firestore:', firestoreHistory.length, 'items');

            analysisHistory = firestoreHistory;
            localStorage.setItem(`analysisHistory_${currentUser.uid}`, JSON.stringify(firestoreHistory));
            updatePageUI();
        } else {
            console.log('📭 No history found in Firestore');
        }
    } catch (error) {
        console.error("❌ Firestore load failed:", error);
    }
}

async function saveUserHistory() {
    if (isTrialMode) {
        console.log('Trial mode: Skipping history save');
        return;
    }

    if (!currentUser) {
        console.log('No current user, skipping history save');
        return;
    }

    if (!analysisHistory.length) {
        console.log('No analysis history to save');
        return;
    }

    try {
        // Save the MOST RECENT analysis
        const latestAnalysis = analysisHistory[0];

        // Ensure the analysis has all required fields
        const analysisToSave = {
            ...latestAnalysis,
            // Ensure timestamp is in the correct format for Firestore querying
            timestamp: latestAnalysis.timestamp || new Date().toISOString(),
            savedAt: firebase.firestore.FieldValue.serverTimestamp()
        };

        console.log('💾 Saving analysis to Firestore:', analysisToSave);

        await firebase.firestore()
            .collection('userAnalyses')
            .doc(currentUser.uid)
            .collection('analyses')
            .doc(latestAnalysis.id.toString())
            .set(analysisToSave);

        console.log('✅ Analysis saved successfully to Firestore');

        // Update localStorage with current state
        localStorage.setItem(`analysisHistory_${currentUser.uid}`, JSON.stringify(analysisHistory));

    } catch (error) {
        console.error('❌ Error saving analysis to Firestore:', error);
        // Fallback to localStorage only
        localStorage.setItem(`analysisHistory_${currentUser.uid}`, JSON.stringify(analysisHistory));
    }
}

// --- Event Listeners Setup ---
function setupEventListeners() {
    document.querySelector('.nav-toggle')?.addEventListener('click', () => {
        const navLinks = document.querySelector('.nav-links');
        if (navLinks) {
            navLinks.classList.toggle('active');
            // Add mobile-specific styles
            if (window.innerWidth <= 768) {
                if (navLinks.classList.contains('active')) {
                    navLinks.style.display = 'flex';
                    navLinks.style.flexDirection = 'column';
                    navLinks.style.position = 'absolute';
                    navLinks.style.top = '70px';
                    navLinks.style.left = '0';
                    navLinks.style.right = '0';
                    navLinks.style.background = 'var(--surface-color)';
                    navLinks.style.padding = '20px';
                    navLinks.style.boxShadow = 'var(--shadow-lg)';
                } else {
                    navLinks.style.display = 'none';
                }
            }
        }
    });
    document.querySelector('.btn-logout')?.addEventListener('click', handleLogout);
    document.getElementById('loginForm')?.addEventListener('submit', handleLoginSubmit);
    document.getElementById('signupForm')?.addEventListener('submit', handleSignupSubmit);
    document.querySelectorAll('.password-toggle').forEach(toggle => {
        toggle.addEventListener('click', function () {
            const input = this.previousElementSibling;
            input.type = input.type === 'password' ? 'text' : 'password';
            this.textContent = input.type === 'password' ? '👁️' : '🙈';
        });
    });
    document.querySelectorAll('.tab-button').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));
    document.querySelector('.analyze-btn')?.addEventListener('click', startAnalysis);
    document.querySelector('.new-analysis-btn')?.addEventListener('click', resetAnalysis);
    document.getElementById('clearFiltersBtn')?.addEventListener('click', clearHistoryFilters);
    document.getElementById('typeFilter')?.addEventListener('change', filterHistory);
    document.getElementById('resultFilter')?.addEventListener('change', filterHistory);
    setupFileUploads();
}

function setupFileUploads() {
    const uploadVideoArea = document.getElementById('uploadVideoArea');
    const uploadBatchArea = document.getElementById('uploadBatchArea');

    const uploadFileInput = document.getElementById('uploadFileInput');
    const batchFileInput = document.getElementById('batchFileInput');

    setupUploadArea(uploadVideoArea, uploadFileInput);
    setupUploadArea(uploadBatchArea, batchFileInput);
}

function setupUploadArea(uploadArea, fileInput) {
    if (!uploadArea || !fileInput) return;

    // Prevent duplicate setup
    if (uploadArea._setupInitialized) return;
    uploadArea._setupInitialized = true;

    // Mobile-optimized file input setup
    fileInput.style.position = 'absolute';
    fileInput.style.top = '0';
    fileInput.style.left = '0';
    fileInput.style.width = '100%';
    fileInput.style.height = '100%';
    fileInput.style.opacity = '0';
    fileInput.style.zIndex = '10';
    fileInput.style.cursor = 'pointer';

    // Ensure file input is inside upload area
    if (!uploadArea.contains(fileInput)) {
        uploadArea.appendChild(fileInput);
    }

    let isProcessingClick = false;

    const handleAreaClick = (e) => {
        if (isProcessingClick) return;
        isProcessingClick = true;

        if (e.target !== fileInput && !fileInput.contains(e.target)) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            fileInput.click();
        }

        setTimeout(() => { isProcessingClick = false; }, 500);
    };

    // Remove any existing listeners and add new ones
    uploadArea.removeEventListener('click', handleAreaClick);
    uploadArea.removeEventListener('touchend', handleAreaClick);

    uploadArea.addEventListener('click', handleAreaClick, true);
    uploadArea.addEventListener('touchend', handleAreaClick, { passive: false, capture: true });

    // Drag/drop for desktop
    const onDragOver = (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    };
    const onDragLeave = () => uploadArea.classList.remove('dragover');
    const onDrop = (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer && e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelection(e.dataTransfer.files, fileInput);
        }
    };

    uploadArea.addEventListener('dragover', onDragOver);
    uploadArea.addEventListener('dragleave', onDragLeave);
    uploadArea.addEventListener('drop', onDrop);

    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length) {
            handleFileSelection(e.target.files, fileInput);
        }
    });

    // Ensure proper styling for mobile
    uploadArea.style.position = 'relative';
    uploadArea.style.overflow = 'hidden';
    uploadArea.style.cursor = 'pointer';
}

// --- Authentication ---
async function handleLoginSubmit(e) {
    e.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    if (!email || !password) {
        showNotification('Please fill in all fields', 'error');
        return;
    }
    try {
        showLoading('Signing in...');
        await firebase.auth().signInWithEmailAndPassword(email, password);
        hideLoading();
        showNotification('Login successful!', 'success');
        setTimeout(() => { window.location.href = '/dashboard'; }, 1000);
    } catch (error) {
        hideLoading();
        showNotification(getFirebaseErrorMessage(error), 'error');
    }
}

async function handleSignupSubmit(e) {
    e.preventDefault();
    const name = document.getElementById('signupName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    if (!name || !email || !password || !confirmPassword) {
        showNotification('Please fill in all fields', 'error');
        return;
    }
    if (password !== confirmPassword) {
        showNotification('Passwords do not match', 'error');
        return;
    }
    if (password.length < 6) {
        showNotification('Password must be at least 6 characters', 'error');
        return;
    }
    try {
        showLoading('Creating account...');
        const userCredential = await firebase.auth().createUserWithEmailAndPassword(email, password);
        await userCredential.user.updateProfile({ displayName: name });
        hideLoading();
        showNotification('Account created successfully!', 'success');
        setTimeout(() => { window.location.href = '/dashboard'; }, 1000);
    } catch (error) {
        hideLoading();
        showNotification(getFirebaseErrorMessage(error), 'error');
    }
}

async function handleLogout() {
    try {
        await firebase.auth().signOut();
        showNotification('Logged out successfully', 'success');
        setTimeout(() => { window.location.href = '/'; }, 1000);
    } catch (error) {
        showNotification('Logout failed. Please try again.', 'error');
    }
}

// --- File and Analysis Logic ---
function handleFileSelection(files, fileInput) {
    if (!files || !files.length) return;
    const isBatch = fileInput.multiple;
    const uploadArea = fileInput.parentElement;

    // Prevent duplicate input elements inside the uploadArea
    const presentInputs = Array.from(uploadArea.querySelectorAll('input[type="file"]'));
    presentInputs.forEach(el => {
        if (el !== fileInput) el.remove();
    });

    // Keep fileInput hidden but attached
    fileInput.style.display = 'none';

    let successHTML = '';
    if (isBatch) {
        successHTML = `<div class="upload-success"><i class="fas fa-check-circle" style="color: var(--success-color); font-size: 3rem;"></i><h3>${files.length} Files Selected</h3><p>Ready for batch analysis.</p></div>`;
    } else {
        const file = files[0];
        successHTML = `<div class="upload-success"><i class="fas fa-check-circle" style="color: var(--success-color); font-size: 3rem;"></i><h3>File Selected</h3><p>${file.name}</p><p class="text-sm">${formatFileSize(file.size)}</p></div>`;
    }

    // Replace only the visible content portion, keep fileInput attached
    let contentContainer = uploadArea.querySelector('.upload-content');
    if (!contentContainer) {
        // create a content container and move existing non-input children into it
        contentContainer = document.createElement('div');
        contentContainer.className = 'upload-content';
        // move all existing children except file inputs into the content container
        Array.from(uploadArea.childNodes).forEach(child => {
            if (child.nodeType === Node.ELEMENT_NODE && child.tagName.toLowerCase() === 'input') return;
            contentContainer.appendChild(child);
        });
        uploadArea.insertBefore(contentContainer, fileInput);
    }
    contentContainer.innerHTML = successHTML;

    // Re-enable analyze button - ensure it's properly enabled
    const analyzeBtn = document.querySelector('.analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        // Force re-bind the click event to ensure it works
        analyzeBtn.onclick = startAnalysis;
    }
}

async function startAnalysis() {
    if (isAnalyzing) return;
    const activeTab = document.querySelector('.tab-content.active');
    if (!activeTab) return;

    let fileInput, apiUrl, isBatch = false;
    if (activeTab.id === 'uploadTab') {
        fileInput = document.getElementById('uploadFileInput');
        apiUrl = '/api/analyze/video';
    } else if (activeTab.id === 'batchTab') {
        fileInput = document.getElementById('batchFileInput');
        apiUrl = '/api/analyze/batch';
        isBatch = true;
    } else {
        return;
    }

    if (!fileInput || !fileInput.files.length) {
        showNotification('Please select a file first.', 'error');
        return;
    }

    const formData = new FormData();
    if (isBatch) {
        for (let i = 0; i < fileInput.files.length; i++) {
            formData.append('files', fileInput.files[i]);
        }
    } else {
        formData.append('file', fileInput.files[0]);
    }

    isAnalyzing = true;
    showLoading('Uploading and analyzing, please wait... (Mobile uploads may take longer)');

    try {
        const response = await fetch(apiUrl, { method: 'POST', body: formData });
        let result;
        try {
            result = await response.json();
        } catch (err) {
            throw new Error('Invalid response from server. Please try again.');
        }

        if (!response.ok || result.error) {
            throw new Error(result?.error || 'Server error');
        }

        if (isBatch) {
            completeBatchAnalysis(result);
        } else {
            completeSingleAnalysis(fileInput.files[0], result);
        }
    } catch (error) {
        console.error('Analysis failed:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
        hideLoading();
        isAnalyzing = false;
    }
}

function completeSingleAnalysis(file, result) {
    console.log('🔍 Raw API response:', result);

    hideLoading();
    isAnalyzing = false;

    const analysisResult = {
        id: Date.now().toString(),
        filename: file.name,
        timestamp: new Date().toISOString(),
        type: file.type.includes('video') ? 'video' : 'image',
        isReal: result.isReal,
        confidence: parseFloat(result.confidence) * 100,
        probability: result.probability,
        heatmap_base64: result.heatmap_base64,
        frame_count: result.frame_count || 1
    };

    console.log('💫 Created analysis result:', analysisResult);

    // Only add to history and save if not in trial mode
    if (!isTrialMode) {
        analysisHistory.unshift(analysisResult);
        console.log('📝 Added to analysisHistory, new length:', analysisHistory.length);
        saveUserHistory();
    }

    showAnalysisResults(analysisResult, file);
    showNotification('Analysis complete!', 'success');
}

function completeBatchAnalysis(result) {
    hideLoading();
    isAnalyzing = false;

    const analysisResult = {
        id: Date.now(),
        filename: `${result.frame_count || 'Multiple'} Frames (Video)`,
        timestamp: new Date().toISOString(),
        type: 'video',
        isReal: result.isReal,
        heatmap_base64: result.heatmap_base64,
        frame_count: result.frame_count || 1
    };

    // Only add to history and save if not in trial mode
    if (!isTrialMode) {
        analysisHistory.unshift(analysisResult);
        saveUserHistory();
    }

    showAnalysisResults(analysisResult, new File([], `batch_analysis_${Date.now()}.mp4`));
    showNotification(`Batch analysis complete! Analyzed ${result.frame_count || 'multiple'} frames.`, 'success');
}

// --- UI Rendering ---
function showAnalysisResults(result, file) {
    hideLoading();
    isAnalyzing = false;

    const header = document.querySelector('.analysis-header');
    if (header) header.style.display = 'none';

    const resultsSection = document.querySelector('.results-section');
    const analysisTab = document.querySelector('.analysis-tabs');
    const analyzeActions = document.querySelector('.analyze-actions');

    resultsSection.style.display = 'block';
    analysisTab.style.display = 'none';
    analyzeActions.style.display = 'none';

    const isReal = result.isReal;

    document.getElementById('predictionStatus').textContent = isReal ? 'REAL CONTENT' : 'DEEPFAKE DETECTED';
    document.getElementById('predictionStatus').className = `prediction-status ${isReal ? 'real' : 'fake'}`;

    // Set result description
    const resultDescription = document.getElementById('resultDescription');
    if (isReal) {
        resultDescription.textContent = 'The content appears to be authentic and not AI-generated.';
        resultDescription.style.color = 'var(--success-color)';
    } else {
        resultDescription.textContent = 'AI-generated manipulation detected in the content.';
        resultDescription.style.color = 'var(--danger-color)';
    }

    const vizContainer = document.querySelector('.visualization-container');
    const mediaURL = URL.createObjectURL(file);
    const mediaEl = file.type && file.type.startsWith('image/')
        ? `<img src="${mediaURL}" alt="Analyzed Media">`
        : `<video src="${mediaURL}" controls></video>`;

    // Heatmap display with styling
    const heatmapEl = result.heatmap_base64
        ? `<div class="heatmap-container">
             <img src="data:image/jpeg;base64,${result.heatmap_base64}" alt="AI Attention Heatmap" class="heatmap-image">
             <div class="heatmap-overlay">AI Focus Areas</div>
           </div>`
        : '<div class="no-visualization"><i class="fas fa-image" style="font-size: 3rem; color: var(--text-secondary);"></i><p>Visualization not available</p></div>';

    vizContainer.innerHTML = `
        <div class="visualization-item">
            <div class="visualization-label">Original Media</div>
            <div class="media-container">${mediaEl}</div>
        </div>
        <div class="visualization-item">
            <div class="visualization-label">AI Analysis Heatmap</div>
            <div class="heatmap-wrapper">${heatmapEl}</div>
            <p class="heatmap-caption">Red areas indicate where the AI detected potential manipulation</p>
        </div>
    `;

    document.querySelector('.result-actions').innerHTML = `
        <button class="btn btn-primary new-analysis-btn"><i class="fas fa-redo"></i> New Analysis</button>
    `;
    const newBtn = document.querySelector('.new-analysis-btn');
    if (newBtn) newBtn.addEventListener('click', resetAnalysis);

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function resetAnalysis() {
    console.log('Resetting analysis... Trial mode:', isTrialMode);

    const resultsSection = document.querySelector('.results-section');
    const analysisTab = document.querySelector('.analysis-tabs');
    const analyzeActions = document.querySelector('.analyze-actions');

    if (resultsSection) resultsSection.style.display = 'none';
    if (analysisTab) analysisTab.style.display = 'block';
    if (analyzeActions) analyzeActions.style.display = 'block';

    if (analysisChart) {
        analysisChart.destroy();
        analysisChart = null;
    }

    const header = document.querySelector('.analysis-header');
    if (header) header.style.display = 'block';

    // Show trial mode indicator again if in trial mode
    if (isTrialMode) {
        showTrialModeIndicator();
    }

    // --- Rebuild upload areas ---
    replaceUploadArea('uploadVideoArea', 'Upload Video', 'Drag & drop your video file here or click to browse', 'Supports: MP4, AVI, MOV (Max 100MB)', 'uploadFileInput', '.mp4,.avi,.mov', false);
    replaceUploadArea('uploadBatchArea', 'Upload Video Frames', 'Drag & drop multiple image files extracted from a video or click to browse', 'Upload frames extracted from videos for more accurate analysis', 'batchFileInput', '.jpg,.jpeg,.png', true);

    // --- REINITIALIZE Upload Event Bindings ---
    const uploadVideoArea = document.getElementById('uploadVideoArea');
    const uploadBatchArea = document.getElementById('uploadBatchArea');

    const uploadFileInput = document.getElementById('uploadFileInput');
    const batchFileInput = document.getElementById('batchFileInput');

    setupUploadArea(uploadVideoArea, uploadFileInput);
    setupUploadArea(uploadBatchArea, batchFileInput);

    // --- Reset analyze button ---
    const analyzeBtn = document.querySelector('.analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        // Remove any existing event listeners and re-add
        analyzeBtn.replaceWith(analyzeBtn.cloneNode(true));
        document.querySelector('.analyze-btn').addEventListener('click', startAnalysis);
    }

    isAnalyzing = false;

    // --- Force reinitialization of current tab ---
    const activeTab = document.querySelector('.tab-button.active');
    if (activeTab) {
        // Remove and re-add active class to trigger any tab-specific initialization
        activeTab.classList.remove('active');
        setTimeout(() => {
            activeTab.classList.add('active');
            switchTab(activeTab.dataset.tab);

            // Force reset analyze button state after tab switch
            setTimeout(() => {
                const analyzeBtn = document.querySelector('.analyze-btn');
                if (analyzeBtn) analyzeBtn.disabled = true;
            }, 50);
        }, 10);
    } else {
        switchTab('upload');

        // Force reset analyze button state
        setTimeout(() => {
            const analyzeBtn = document.querySelector('.analyze-btn');
            if (analyzeBtn) analyzeBtn.disabled = true;
        }, 50);
    }

    console.log('Analysis reset complete and upload areas reinitialized');
}

function replaceUploadArea(areaId, title, description, formats, inputId, accept, multiple) {
    const oldArea = document.getElementById(areaId);
    if (!oldArea) {
        console.log('Upload area not found:', areaId);
        return;
    }

    // Create new area node from scratch
    const newArea = document.createElement('div');
    newArea.className = 'upload-area';
    newArea.id = areaId;

    const multipleAttr = multiple ? 'multiple' : '';
    newArea.innerHTML = `
        <i class="fas fa-cloud-upload-alt upload-icon"></i>
        <h3>${title}</h3>
        <p>${description}</p>
        <p class="text-sm">${formats}</p>
        <input type="file" id="${inputId}" accept="${accept}" ${multipleAttr} style="display: none;">
    `;

    oldArea.parentNode.replaceChild(newArea, oldArea);

    // Attach file input and re-setup upload area
    const fileInput = document.getElementById(inputId);
    setupUploadArea(newArea, fileInput);
}

// --- History Page Logic ---
function loadHistoryPage() {
    const historyList = document.getElementById('historyList');
    if (!historyList) {
        console.log('❌ History list element not found');
        return;
    }

    console.log('📖 Loading history page with', analysisHistory.length, 'items');

    if (analysisHistory.length === 0) {
        historyList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-history"></i>
                <h3>No Analysis History</h3>
                <p>Your analyzed files will appear here.</p>
                <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 10px;">
                    Analyses from your dashboard should appear here automatically.
                </p>
            </div>
        `;
        return;
    }

    renderHistoryItems(analysisHistory);
}

function renderHistoryItems(items) {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = items.map(item => `
        <div class="activity-item" data-type="${item.type}" data-result="${item.isReal ? 'real' : 'fake'}" onclick="showHistoryDetails(${JSON.stringify(item).replace(/"/g, '&quot;')})" style="cursor: pointer;">
            <div class="activity-icon ${item.isReal ? 'real' : 'fake'}"><i class="fas fa-${item.type === 'video' ? 'video' : 'image'}"></i></div>
            <div class="activity-details">
                <div class="activity-title">${item.filename}</div>
                <div class="activity-meta"><span>${formatDate(item.timestamp)}</span><span>•</span><span>${item.type}</span></div>
            </div>
            <div class="activity-result ${item.isReal ? 'real' : 'fake'}">
                ${item.isReal ? 'Real' : 'Fake'} (${item.confidence.toFixed(1)}%)
            </div>
        </div>
    `).join('');
}

function showHistoryDetails(historyItem) {
    // If we're on the dashboard page, redirect to analysis page with history data
    if (window.location.pathname.includes('/dashboard') || window.location.pathname.includes('/history')) {
        // Store the history item to show on analysis page
        sessionStorage.setItem('viewHistoryItem', JSON.stringify(historyItem));
        window.location.href = '/analysis?view=history';
        return;
    }

    // If we're already on analysis page, show the history details
    const resultsSection = document.querySelector('.results-section');
    const analysisTab = document.querySelector('.analysis-tabs');
    const analyzeActions = document.querySelector('.analyze-actions');

    resultsSection.style.display = 'block';
    analysisTab.style.display = 'none';
    analyzeActions.style.display = 'none';

    const isReal = historyItem.isReal;

    document.getElementById('predictionStatus').textContent = isReal ? 'REAL CONTENT' : 'DEEPFAKE DETECTED';
    document.getElementById('predictionStatus').className = `prediction-status ${isReal ? 'real' : 'fake'}`;

    // Set result description
    const resultDescription = document.getElementById('resultDescription');
    if (isReal) {
        resultDescription.textContent = 'The content appears to be authentic and not AI-generated.';
        resultDescription.style.color = 'var(--success-color)';
    } else {
        resultDescription.textContent = 'AI-generated manipulation detected in the content.';
        resultDescription.style.color = 'var(--danger-color)';
    }

    const vizContainer = document.querySelector('.visualization-container');

    // History visualization
    let mediaHTML = '';
    if (historyItem.heatmap_base64) {
        mediaHTML = `
            <div class="visualization-item">
                <div class="visualization-label">Analysis Results</div>
                <div class="no-visualization">
                    <i class="fas fa-${historyItem.type === 'video' ? 'video' : 'image'}" style="font-size: 3rem; color: var(--text-secondary);"></i>
                    <p>Original media not stored for privacy</p>
                </div>
            </div>
            <div class="visualization-item">
                <div class="visualization-label">AI Analysis Heatmap</div>
                <div class="heatmap-wrapper">
                    <div class="heatmap-container">
                        <img src="data:image/jpeg;base64,${historyItem.heatmap_base64}" alt="AI Attention Heatmap" class="heatmap-image">
                        <div class="heatmap-overlay">AI Focus Areas</div>
                    </div>
                </div>
                <p class="heatmap-caption">Red areas indicate where the AI detected potential manipulation</p>
            </div>
        `;
    } else {
        mediaHTML = `
            <div class="visualization-item">
                <div class="visualization-label">Analysis Results</div>
                <div class="no-visualization">
                    <i class="fas fa-${historyItem.type === 'video' ? 'video' : 'image'}" style="font-size: 3rem; color: var(--text-secondary);"></i>
                    <p>Visualization not available for this analysis</p>
                </div>
            </div>
        `;
    }

    vizContainer.innerHTML = mediaHTML;

    // Update result actions
    document.querySelector('.result-actions').innerHTML = `
        <button class="btn btn-primary new-analysis-btn"><i class="fas fa-redo"></i> New Analysis</button>
    `;

    // Re-add event listener
    const newBtn = document.querySelector('.new-analysis-btn');
    if (newBtn) newBtn.addEventListener('click', resetAnalysis);

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function filterHistory() {
    const resultFilter = document.getElementById('resultFilter').value;
    const items = document.querySelectorAll('#historyList .activity-item');
    items.forEach(item => {
        const resultMatch = resultFilter === 'all' || item.dataset.result === resultFilter;
        item.style.display = resultMatch ? 'flex' : 'none';
    });
}

function clearHistoryFilters() {
    document.getElementById('resultFilter').value = 'all';
    filterHistory();
}

// --- UI Helpers & Page Management ---
function updatePageUI() {
    console.log('🔄 Updating page UI, current user:', currentUser ? currentUser.email : 'none');
    console.log('📊 History length:', analysisHistory.length);
    console.log('📍 Current path:', window.location.pathname);

    updateNavigation();

    if (window.location.pathname.includes('/dashboard')) {
        console.log('🏠 On dashboard, updating...');
        updateDashboard();
    }

    if (window.location.pathname.includes('/history')) {
        console.log('📚 On history page, loading history...');
        loadHistoryPage();
    }

    if (window.location.pathname.includes('/analysis')) {
        const params = new URLSearchParams(window.location.search);
        const tabParam = params.get('tab');

        if (tabParam) {
            switchTab(tabParam);
        }

        if (params.get('view') === 'history') {
            document.querySelector('.analysis-tabs')?.style.setProperty('display', 'none', 'important');
            document.querySelector('.analyze-actions')?.style.setProperty('display', 'none', 'important');
            document.querySelector('.results-section')?.style.setProperty('display', 'block', 'important');

            const header = document.querySelector('.analysis-header');
            if (header) header.style.display = 'none';
        }

        if (window.location.search.includes('view=history')) {
            const historyItem = sessionStorage.getItem('viewHistoryItem');
            if (historyItem) {
                showHistoryDetails(JSON.parse(historyItem));
                sessionStorage.removeItem('viewHistoryItem');
            }
        }

        // Show trial mode indicator if in trial mode
        if (isTrialMode) {
            showTrialModeIndicator();
        }
    }
}

function updateNavigation() {
    const authSection = document.querySelector('.auth-section');
    const userSection = document.querySelector('.user-menu');

    if (authSection && userSection) {
        if (currentUser) {
            authSection.style.display = 'none';
            userSection.style.display = 'flex';
            document.querySelectorAll('.user-name').forEach(el => el.textContent = currentUser.name || 'User');
        } else {
            if (isTrialMode) {
                authSection.style.display = 'flex';
                userSection.style.display = 'none';

                const trialNavIndicator = document.createElement('div');
                trialNavIndicator.className = 'trial-nav-indicator';
                trialNavIndicator.innerHTML = `<span style="color: var(--warning-color); font-weight: 600;"><i class="fas fa-flask"></i> Trial Mode</span>`;

                const existingIndicator = document.querySelector('.trial-nav-indicator');
                if (existingIndicator) {
                    existingIndicator.remove();
                }

                authSection.parentNode.insertBefore(trialNavIndicator, authSection);
            } else {
                authSection.style.display = 'flex';
                userSection.style.display = 'none';

                const existingIndicator = document.querySelector('.trial-nav-indicator');
                if (existingIndicator) {
                    existingIndicator.remove();
                }
            }
        }
    }

    const getStartedBtn = document.querySelector('.btn-primary.get-started');
    if (getStartedBtn) {
        getStartedBtn.style.display = currentUser ? 'none' : 'inline-block';
    }
}

function checkAuthState() {
    const isAuthPage = window.location.pathname.includes('/login') || window.location.pathname.includes('/signup');
    const isProtectedPage = ['/dashboard', '/analysis', '/history'].some(p => window.location.pathname.includes(p));

    // Allow trial mode access to analysis page
    if (isProtectedPage && !currentUser && !isTrialMode) {
        if (window.location.pathname.includes('/analysis')) {
            // Redirect to trial mode if trying to access analysis without auth
            window.location.href = '/analysis?mode=trial';
            return;
        } else {
            window.location.href = '/login';
        }
    }

    if (isAuthPage && currentUser) window.location.href = '/dashboard';
}

function updateDashboard() {
    if (!currentUser) {
        console.log('No user for dashboard update');
        return;
    }

    console.log('Updating dashboard with history:', analysisHistory.length, 'items');

    document.getElementById('userDisplayName').textContent = currentUser.name || 'User';
    const total = analysisHistory.length;
    const real = analysisHistory.filter(item => item.isReal).length;
    const fake = total - real;
    const avgConfidence = total > 0 ? (analysisHistory.reduce((sum, item) => sum + item.confidence, 0) / total).toFixed(1) : '0.0';

    const statNumbers = document.querySelectorAll('.stat-number');
    if (statNumbers.length >= 4) {
        statNumbers[0].textContent = total;
        statNumbers[1].textContent = real;
        statNumbers[2].textContent = fake;
        statNumbers[3].textContent = `${avgConfidence}%`;
    }

    updateRecentActivity();
}

function updateRecentActivity() {
    const activityList = document.querySelector('.activity-list');
    if (!activityList) {
        console.log('Activity list not found');
        return;
    }

    console.log('Updating recent activity with', analysisHistory.length, 'items');

    if (analysisHistory.length === 0) {
        activityList.innerHTML = `<div class="empty-state"><i class="fas fa-search"></i><h3>No Analysis Yet</h3><p>Your recent activity will appear here.</p></div>`;
        return;
    }

    const recentItems = analysisHistory.slice(0, 5);
    activityList.innerHTML = recentItems.map(item => `
        <div class="activity-item" data-result="${item.isReal ? 'real' : 'fake'}" style="cursor: pointer;">
            <div class="activity-icon ${item.isReal ? 'real' : 'fake'}"><i class="fas fa-${item.type === 'video' ? 'video' : 'image'}"></i></div>
            <div class="activity-details">
                <div class="activity-title">${item.filename}</div>
                <div class="activity-meta"><span>${formatDate(item.timestamp)}</span><span>•</span><span>${item.type}</span></div>
            </div>
            <div class="activity-result ${item.isReal ? 'real' : 'fake'}">
                ${item.isReal ? 'Real' : 'Fake'}
            </div>
        </div>
    `).join('');

    // Add click event listeners to all activity items
    document.querySelectorAll('.activity-item').forEach((item, index) => {
        item.addEventListener('click', () => {
            showHistoryDetails(recentItems[index]);
        });
    });
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-button').forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === `${tabName}Tab`));

    // Reset analyze button for the new tab
    const analyzeBtn = document.querySelector('.analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        // Clear any file selection state
        const activeTab = document.querySelector('.tab-content.active');
        if (activeTab) {
            const fileInput = activeTab.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.value = '';
            }
        }
    }

    // Reinitialize upload areas for the active tab
    const activeTab = document.querySelector('.tab-content.active');
    if (activeTab) {
        const uploadArea = activeTab.querySelector('.upload-area');
        const fileInput = activeTab.querySelector('input[type="file"]');
        if (uploadArea && fileInput) {
            // Remove any existing setup flag and reinitialize
            uploadArea._setupInitialized = false;
            setupUploadArea(uploadArea, fileInput);
        }
    }
}

function showNotification(message, type = 'info') {
    document.querySelector('.notification')?.remove();
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    const icons = { success: 'check-circle', error: 'exclamation-circle', info: 'info-circle', warning: 'exclamation-triangle' };
    notification.innerHTML = `<i class="fas fa-${icons[type]}"></i><span>${message}</span>`;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 5000);
}

function showLoading(message) {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `<div class="loading-content"><div class="loading-spinner"></div><h3>${message}</h3></div>`;
    document.body.appendChild(overlay);
}

function hideLoading() {
    document.querySelector('.loading-overlay')?.remove();
}

function getFirebaseErrorMessage(error) {
    switch (error.code) {
        case 'auth/invalid-email': return 'Invalid email address.';
        case 'auth/user-not-found': return 'No account found with this email.';
        case 'auth/wrong-password': return 'Incorrect password.';
        case 'auth/email-already-in-use': return 'An account with this email already exists.';
        case 'auth/weak-password': return 'Password is too weak (min. 6 characters).';
        default: return 'An unknown error occurred. Please try again.';
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${parseFloat((bytes / Math.pow(1024, i)).toFixed(2))} ${['Bytes', 'KB', 'MB', 'GB'][i]}`;
}

function formatDate(dateString) { return new Date(dateString).toLocaleDateString(); }

function initAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('.feature-card, .stat-card, .activity-item').forEach(el => observer.observe(el));
}