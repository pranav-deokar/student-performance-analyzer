# 🎓 Student Performance AI - Complete System

An AI-powered system for predicting student performance, identifying at-risk students, and providing personalized recommendations using machine learning.

## 📋 Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Model Details](#model-details)
- [API Documentation](#api-documentation)
- [Dataset Format](#dataset-format)

---

## ✨ Features

### Core Functionality

1. **Grade Prediction**
   - Predicts student grades (A/B/C/D) using Random Forest classifier
   - Provides confidence scores for predictions
   - Shows probability distribution across all grades

2. **Risk Assessment**
   - Calculates risk scores (0-100) for each student
   - Higher scores indicate higher risk of poor performance
   - Based on model predictions and key performance indicators

3. **Student Segmentation**
   - Clusters students using K-Means algorithm
   - Automatically determines optimal number of clusters
   - Assigns meaningful labels:
     - High Performer
     - Average Performer
     - At Risk

4. **Personalized Recommendations**
   - Generates actionable suggestions based on student data
   - Prioritizes recommendations (Critical/High/Medium/Low)
   - Covers multiple areas:
     - Attendance improvement
     - Study time optimization
     - Assignment submission
     - LMS engagement
     - Resource utilization

5. **Explainability**
   - Shows feature importance
   - Displays key factors influencing predictions
   - Helps understand model decisions

### User Interface Features

6. **Single Student Prediction**
   - Clean, intuitive input form
   - Required fields: attendance, study hours, assignment rate
   - Optional fields with intelligent defaults
   - Real-time prediction results

7. **Batch Processing**
   - CSV file upload with drag-and-drop
   - Process multiple students simultaneously
   - Batch statistics and insights
   - Sortable results table

8. **Export Functionality**
   - **PDF Reports**: Detailed individual student reports
   - **CSV Export**: Batch prediction results
   - Professional formatting with charts and tables

9. **Modern UI**
   - Brown and beige color theme
   - Smooth animations and transitions
   - Responsive design (mobile-friendly)
   - Interactive visualizations

---

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **scikit-learn** - Machine learning models
- **pandas** - Data processing
- **NumPy** - Numerical computing
- **joblib** - Model serialization

### Frontend
- **HTML5/CSS3** - Structure and styling
- **JavaScript** - Interactive features
- **Vanilla JS** - No framework dependencies

### Data Science
- **Random Forest** - Grade prediction (supervised learning)
- **K-Means** - Student clustering (unsupervised learning)
- **StandardScaler** - Feature normalization

### Reporting
- **ReportLab** - PDF generation

---

## 📁 Project Structure

```
student_performance_ai/
│
├── data/                                      # Dataset folder
│   └── student_performance_dataset_20000.csv  # Training data (place here)
│
├── models/                                    # Trained models (auto-generated)
│   ├── random_forest_model.pkl
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── metadata.pkl
│
├── static/                                    # Static assets
│   ├── css/
│   │   └── style.css                         # Main stylesheet
│   └── js/
│       ├── main.js                           # Landing page scripts
│       └── dashboard.js                      # Dashboard functionality
│
├── templates/                                 # HTML templates
│   ├── index.html                            # Landing page
│   └── dashboard.html                        # Dashboard page
│
├── app.py                                     # Flask application
├── train.py                                   # Model training script
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum
- Dataset CSV file

### Step 1: Extract Project

```bash
unzip student_performance_ai.zip
cd student_performance_ai
```

### Step 2: Place Dataset

Place your dataset file in the `data/` folder:
```
data/student_performance_dataset_20000.csv
```

**Important**: The dataset filename must match exactly, or update the path in `train.py`.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will install:
- Flask
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn
- reportlab

### Step 4: Train Models

Run the training script **once**:

```bash
python train.py
```

**Expected Output**:
```
==============================================================
🎓 STUDENT PERFORMANCE AI - TRAINING PIPELINE
==============================================================

📂 Loading dataset...
✅ Loaded 20000 records
   Columns: [list of columns]

🔧 Preprocessing data...
✅ Features: 10
   Target classes: ['A', 'B', 'C', 'D']

🤖 Training supervised model (Random Forest)...
✅ Model trained successfully
   Accuracy: XX.XX%

📊 Classification Report:
[detailed metrics]

🎯 Training unsupervised model (K-Means)...
   Optimal clusters: 3 (silhouette score: X.XXX)
✅ Clustering completed

   Cluster Assignments:
   • Cluster 0 → High Performer
   • Cluster 1 → Average Performer
   • Cluster 2 → At Risk

💾 Saving models and preprocessors...
✅ All models saved successfully

==============================================================
✅ TRAINING COMPLETED SUCCESSFULLY!
==============================================================
```

### Step 5: Run Application

```bash
python app.py
```

**Expected Output**:
```
==============================================================
🎓 STUDENT PERFORMANCE AI - STARTING APPLICATION
==============================================================

✅ Models loaded successfully
✅ Application ready!
📍 URL: http://localhost:5000

==============================================================
```

### Step 6: Access Application

Open your web browser and navigate to:
```
http://localhost:5000
```

---

## 📖 How to Use

### Single Student Prediction

1. **Navigate to Dashboard**
   - Click "Dashboard" in the navigation bar
   - Or click "Get Started" on the home page

2. **Enter Student Information**
   - **Required fields** (must be filled):
     - Attendance Percentage (0-100)
     - Weekly Self-Study Hours
     - Assignment Submission Rate (0-1)
   
   - **Optional fields** (auto-filled with defaults if empty):
     - Class Participation (1-5)
     - LMS Login Frequency
     - Videos Watched
     - Quiz Attempts
     - Forum Activity
     - Late Submissions Count
     - Previous Grade (0-100)

3. **Get Prediction**
   - Click "Predict Performance"
   - View comprehensive results:
     - Predicted grade with confidence
     - Risk score (0-100)
     - Student cluster
     - Grade probability distribution
     - Personalized recommendations
     - Key influencing factors

4. **Download Report**
   - Click "Download PDF Report"
   - Professional PDF with all details

### Batch Processing

1. **Switch to Batch Tab**
   - Click "Batch Processing" tab

2. **Upload CSV File**
   - Drag and drop CSV file
   - Or click "Browse Files"
   - File must be in CSV format

3. **Process Data**
   - Click "Process Batch"
   - Wait for processing to complete

4. **View Results**
   - See batch statistics
   - Review all predictions in table
   - Filter and sort results

5. **Export Results**
   - Click "Download CSV"
   - Get complete results spreadsheet

---

## 🤖 Model Details

### Supervised Learning: Random Forest

**Purpose**: Grade prediction (A/B/C/D)

**Configuration**:
- Estimators: 200 trees
- Max depth: 15
- Min samples split: 5
- Min samples leaf: 2

**Features Used** (10 total):
1. weekly_self_study_hours
2. attendance_percentage
3. class_participation
4. lms_login_frequency
5. videos_watched
6. assignment_submission_rate
7. quiz_attempts
8. forum_activity
9. late_submissions_count
10. previous_grade

**Performance**:
- Trained on 80% of data
- Tested on 20% of data
- Stratified sampling for balanced classes

### Unsupervised Learning: K-Means

**Purpose**: Student segmentation

**Configuration**:
- Optimal clusters: Determined automatically using silhouette score
- Initialization: k-means++
- Max iterations: 300

**Cluster Interpretation**:
Clusters are automatically labeled based on:
- Average attendance
- Average study hours
- Average assignment submission rate

**Labels**:
- **High Performer**: Above-average engagement
- **Average Performer**: Moderate engagement
- **At Risk**: Below-average engagement

### Risk Score Calculation

**Formula Components**:
1. **Base Risk** (from grade prediction):
   - Grade A: 10
   - Grade B: 30
   - Grade C: 60
   - Grade D/F: 85

2. **Adjustments** (added to base):
   - Attendance < 70%: +10
   - Study hours < 3: +8
   - Assignment rate < 70%: +12
   - Late submissions > 3: +7
   - LMS logins < 5: +8

3. **Final Score**: min(100, base + adjustments)

---

## 🔌 API Documentation

### 1. Single Prediction

**Endpoint**: `/api/predict`  
**Method**: POST  
**Content-Type**: application/json

**Request Body**:
```json
{
  "attendance_percentage": 85.5,
  "weekly_self_study_hours": 10.0,
  "assignment_submission_rate": 0.9,
  "class_participation": 4.0,
  "lms_login_frequency": 15,
  "videos_watched": 20,
  "quiz_attempts": 8,
  "forum_activity": 3.0,
  "late_submissions_count": 1,
  "previous_grade": 82.0
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "predicted_grade": "A",
    "confidence": 87.5,
    "risk_score": 15,
    "cluster": "High Performer",
    "recommendations": [...],
    "feature_importance": [...],
    "all_probabilities": {
      "A": 87.5,
      "B": 10.2,
      "C": 2.1,
      "D": 0.2
    }
  }
}
```

### 2. Batch Prediction

**Endpoint**: `/api/predict_batch`  
**Method**: POST  
**Content-Type**: multipart/form-data

**Request**: Upload CSV file with student data

**Response**:
```json
{
  "success": true,
  "count": 100,
  "data": [
    {
      "student_id": 1,
      "predicted_grade": "B",
      "confidence": 75.3,
      "risk_score": 35,
      "cluster": "Average Performer",
      "recommendation_count": 3
    },
    ...
  ]
}
```

### 3. Generate PDF

**Endpoint**: `/api/generate_pdf`  
**Method**: POST  
**Content-Type**: application/json

**Request Body**:
```json
{
  "prediction": { ... },
  "student": { ... }
}
```

**Response**: PDF file download

### 4. Export Batch CSV

**Endpoint**: `/api/export_batch_csv`  
**Method**: POST  
**Content-Type**: application/json

**Request Body**:
```json
{
  "results": [ ... ]
}
```

**Response**: CSV file download

---

## 📊 Dataset Format

### Training Dataset

**Filename**: `student_performance_dataset_20000.csv`

**Required Columns**:
```
student_id,weekly_self_study_hours,attendance_percentage,class_participation,
lms_login_frequency,videos_watched,assignment_submission_rate,quiz_attempts,
forum_activity,late_submissions_count,previous_grade,total_score,grade
```

**Column Descriptions**:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| student_id | int | - | Unique student identifier |
| weekly_self_study_hours | float | 0-40 | Hours of self-study per week |
| attendance_percentage | float | 0-100 | Class attendance percentage |
| class_participation | float | 1-5 | Participation level (1=low, 5=high) |
| lms_login_frequency | int | 0+ | LMS logins per week |
| videos_watched | int | 0+ | Number of lecture videos watched |
| assignment_submission_rate | float | 0-1 | Fraction of assignments submitted |
| quiz_attempts | int | 0+ | Number of quiz attempts |
| forum_activity | float | 0+ | Forum posts per week |
| late_submissions_count | int | 0+ | Number of late submissions |
| previous_grade | float | 0-100 | Previous course grade |
| total_score | float | 0-100 | Current total score |
| grade | string | A/B/C/D | Final grade (target variable) |

### Batch Prediction CSV

For batch predictions, you can upload a CSV with any subset of the features above. Missing features will be auto-filled with defaults.

**Minimum Required Columns**:
- student_id (recommended)
- attendance_percentage
- weekly_self_study_hours
- assignment_submission_rate

---

## 🎯 Key Features Explained

### Why These Features Matter

1. **Attendance Percentage**: Strong predictor of engagement and success
2. **Study Hours**: Direct correlation with learning depth
3. **Assignment Submission Rate**: Indicates responsibility and time management
4. **LMS Engagement**: Shows active learning behavior
5. **Previous Performance**: Historical indicator of capability

### Recommendation Engine Logic

The system analyzes each feature and provides specific, actionable advice:

- **Low attendance** → "Attend more classes to improve performance"
- **Few study hours** → "Increase self-study time to X hours"
- **Low submission rate** → "Submit all assignments on time"
- **Many late submissions** → "Improve time management skills"
- **Low LMS usage** → "Engage more with learning materials"

---

## 🔧 Troubleshooting

### Models Not Found

**Error**: `FileNotFoundError: models/random_forest_model.pkl`

**Solution**: Run `python train.py` first to generate models

### Dataset Not Found

**Error**: `FileNotFoundError: data/student_performance_dataset_20000.csv`

**Solution**: 
1. Create `data/` folder
2. Place CSV file in it
3. Ensure filename matches exactly

### Port Already in Use

**Error**: `Address already in use`

**Solution**: 
```bash
# Change port in app.py (last line)
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Package Installation Issues

**Error**: `ModuleNotFoundError`

**Solution**:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

---

## 📈 Future Enhancements

Potential additions:
- Dark mode toggle
- Historical tracking
- Comparison charts
- Email notifications for at-risk students
- Integration with LMS systems
- Mobile app version
- Advanced visualizations (D3.js)
- Real-time monitoring dashboard

---

## 📝 License

This project is created for educational and demonstration purposes.

---

## 👨‍💻 Support

For issues or questions:
1. Check this README thoroughly
2. Verify dataset format matches specifications
3. Ensure all dependencies are installed
4. Check console output for error messages

---

## 🎓 Conclusion

This system provides a complete, production-ready solution for student performance prediction and intervention. It combines machine learning, data science, and modern web development to create an intuitive, powerful tool for educational institutions.

**Remember**: 
1. Place dataset in `data/` folder
2. Run `python train.py` once
3. Run `python app.py` to start
4. Open `http://localhost:5000`

That's it! The system is ready to use.

---

**Built with ❤️ using Python, Flask, and scikit-learn**
