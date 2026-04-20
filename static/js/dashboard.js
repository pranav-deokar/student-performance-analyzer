// Student Performance AI - Dashboard JavaScript

let currentPrediction = null;
let currentStudentData = null;
let batchResults = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    initializePredictionForm();
    initializeBatchUpload();
});

// Tab Management
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            this.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
}

// Single Prediction Form
function initializePredictionForm() {
    const form = document.getElementById('prediction-form');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            if (value !== '') {
                data[key] = parseFloat(value);
            }
        }
        
        // Store current student data
        currentStudentData = data;
        
        // Show loading
        showLoading();
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (result.success) {
                currentPrediction = result.data;
                displayResults(result.data);
                
                // Scroll to results
                document.getElementById('results-section').scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            alert('Error making prediction: ' + error.message);
        } finally {
            hideLoading();
        }
    });
    
    // PDF Download
    document.getElementById('download-pdf').addEventListener('click', async function() {
        if (!currentPrediction || !currentStudentData) {
            alert('No prediction available');
            return;
        }
        
        showLoading();
        
        try {
            const response = await fetch('/api/generate_pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prediction: currentPrediction,
                    student: currentStudentData
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `student_performance_report_${Date.now()}.pdf`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                alert('Error generating PDF');
            }
        } catch (error) {
            alert('Error downloading PDF: ' + error.message);
        } finally {
            hideLoading();
        }
    });
}

// Display Prediction Results
function displayResults(data) {
    // Show results section
    document.getElementById('results-section').style.display = 'block';
    
    // Update result cards
    document.getElementById('predicted-grade').textContent = data.predicted_grade;
    document.getElementById('confidence').textContent = data.confidence.toFixed(2);
    document.getElementById('risk-score').textContent = data.risk_score + '/100';
    document.getElementById('cluster').textContent = data.cluster;
    
    // Update risk bar
    const riskFill = document.getElementById('risk-fill');
    riskFill.style.width = data.risk_score + '%';
    
    // Display grade distribution
    displayGradeDistribution(data.all_probabilities);
    
    // Display recommendations
    displayRecommendations(data.recommendations);
    
    // Display feature importance
    displayFeatureImportance(data.feature_importance);
}

// Grade Distribution Chart
function displayGradeDistribution(probabilities) {
    const container = document.getElementById('grade-distribution');
    container.innerHTML = '';
    
    // Sort grades
    const grades = ['A', 'B', 'C', 'D'];
    
    grades.forEach(grade => {
        if (probabilities[grade] !== undefined) {
            const barItem = document.createElement('div');
            barItem.className = 'grade-bar-item';
            
            const label = document.createElement('div');
            label.className = 'grade-label';
            label.textContent = `Grade ${grade}`;
            
            const barBg = document.createElement('div');
            barBg.className = 'grade-bar-bg';
            
            const barFill = document.createElement('div');
            barFill.className = 'grade-bar-fill';
            barFill.textContent = probabilities[grade].toFixed(1) + '%';
            
            // Animate width
            setTimeout(() => {
                barFill.style.width = probabilities[grade] + '%';
            }, 100);
            
            barBg.appendChild(barFill);
            barItem.appendChild(label);
            barItem.appendChild(barBg);
            container.appendChild(barItem);
        }
    });
}

// Display Recommendations
function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations-list');
    container.innerHTML = '';
    
    if (recommendations.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--success-color);">✅ Great job! No specific recommendations at this time.</p>';
        return;
    }
    
    recommendations.forEach(rec => {
        const recItem = document.createElement('div');
        recItem.className = 'recommendation-item';
        
        const header = document.createElement('div');
        header.className = 'rec-header';
        
        const category = document.createElement('div');
        category.className = 'rec-category';
        category.textContent = `${rec.icon} ${rec.category}`;
        
        const priority = document.createElement('div');
        priority.className = `rec-priority priority-${rec.priority.toLowerCase()}`;
        priority.textContent = rec.priority;
        
        header.appendChild(category);
        header.appendChild(priority);
        
        const action = document.createElement('div');
        action.className = 'rec-action';
        action.textContent = rec.action;
        
        recItem.appendChild(header);
        recItem.appendChild(action);
        container.appendChild(recItem);
    });
}

// Display Feature Importance
function displayFeatureImportance(features) {
    const container = document.getElementById('feature-importance');
    container.innerHTML = '';
    
    features.forEach(feat => {
        const featItem = document.createElement('div');
        featItem.className = 'feature-item';
        
        const name = document.createElement('div');
        name.className = 'feature-name';
        name.textContent = feat.feature;
        
        const details = document.createElement('div');
        details.className = 'feature-details';
        details.innerHTML = `
            <span>Importance: ${feat.importance}</span>
            <span>Value: ${feat.value}</span>
        `;
        
        featItem.appendChild(name);
        featItem.appendChild(details);
        container.appendChild(featItem);
    });
}

// Batch Upload
function initializeBatchUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('csv-file');
    const browseBtn = document.getElementById('browse-btn');
    const fileInfo = document.getElementById('file-info');
    const processBtn = document.getElementById('process-batch');
    const downloadCsvBtn = document.getElementById('download-batch-csv');
    
    let selectedFile = null;
    
    // Browse button
    browseBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        fileInput.click();
    });
    
    // Upload area click
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFileSelect(this.files[0]);
        }
    });
    
    function handleFileSelect(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please select a CSV file');
            return;
        }
        
        selectedFile = file;
        document.getElementById('file-name').textContent = file.name;
        fileInfo.style.display = 'block';
    }
    
    // Process batch
    processBtn.addEventListener('click', async function() {
        if (!selectedFile) {
            alert('Please select a file first');
            return;
        }
        
        showLoading();
        
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        try {
            const response = await fetch('/api/predict_batch', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                batchResults = result.data;
                displayBatchResults(result.data);
                
                // Scroll to results
                document.getElementById('batch-results-section').scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            alert('Error processing batch: ' + error.message);
        } finally {
            hideLoading();
        }
    });
    
    // Download CSV
    downloadCsvBtn.addEventListener('click', async function() {
        if (!batchResults) {
            alert('No batch results available');
            return;
        }
        
        showLoading();
        
        try {
            const response = await fetch('/api/export_batch_csv', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ results: batchResults })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `batch_predictions_${Date.now()}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                alert('Error exporting CSV');
            }
        } catch (error) {
            alert('Error downloading CSV: ' + error.message);
        } finally {
            hideLoading();
        }
    });
}

// Display Batch Results
function displayBatchResults(results) {
    document.getElementById('batch-results-section').style.display = 'block';
    
    // Calculate statistics
    const totalStudents = results.length;
    const highRisk = results.filter(r => r.risk_score && r.risk_score > 70).length;
    const atRisk = results.filter(r => r.cluster === 'At Risk').length;
    
    document.getElementById('total-students').textContent = totalStudents;
    document.getElementById('high-risk-count').textContent = highRisk;
    document.getElementById('at-risk-cluster').textContent = atRisk;
    
    // Populate table
    const tbody = document.getElementById('batch-results-body');
    tbody.innerHTML = '';
    
    results.forEach(result => {
        const row = document.createElement('tr');
        
        if (result.error) {
            row.innerHTML = `
                <td>${result.student_id}</td>
                <td colspan="5" style="color: var(--danger-color);">Error: ${result.error}</td>
            `;
        } else {
            const riskColor = result.risk_score > 70 ? 'var(--danger-color)' : 
                            result.risk_score > 40 ? 'var(--warning-color)' : 
                            'var(--success-color)';
            
            row.innerHTML = `
                <td>${result.student_id}</td>
                <td><strong>${result.predicted_grade}</strong></td>
                <td>${result.confidence.toFixed(1)}%</td>
                <td style="color: ${riskColor}; font-weight: bold;">${result.risk_score}/100</td>
                <td>${result.cluster}</td>
                <td>${result.recommendation_count} items</td>
            `;
        }
        
        tbody.appendChild(row);
    });
}

// Loading Overlay
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}
