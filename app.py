"""
Student Performance AI - Flask Application
Main application with prediction, batch processing, and reporting features
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class StudentPerformancePredictor:
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        """Load all trained models and preprocessors"""
        try:
            self.rf_model = joblib.load('models/random_forest_model.pkl')
            self.kmeans_model = joblib.load('models/kmeans_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.metadata = joblib.load('models/metadata.pkl')
            
            self.feature_names = self.metadata['feature_names']
            self.cluster_labels = self.metadata['cluster_labels']
            self.grade_classes = self.metadata['grade_classes']
            
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise
    
    def prepare_features(self, data_dict):
        """Prepare features from input dictionary"""
        
        # Mapping for previous_grade
        grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # Default values
        defaults = {
            'weekly_self_study_hours': 5.0,
            'attendance_percentage': 75.0,
            'class_participation': 3.0,
            'lms_login_frequency': 10.0,
            'videos_watched': 15.0,
            'assignment_submission_rate': 0.8,
            'quiz_attempts': 5.0,
            'forum_activity': 2.0,
            'late_submissions_count': 1.0,
            'previous_grade': 2  # default = C
        }
        
        # Update values
        for key in data_dict:
            if key in defaults:
                if key == 'previous_grade':
                    # Convert A/B/C/D → number
                    defaults[key] = grade_mapping.get(data_dict[key], 2)
                else:
                    defaults[key] = float(data_dict[key])
        
        # Create feature vector
        features = [defaults[fname] for fname in self.feature_names]
        
        return np.array(features).reshape(1, -1)
    
    def calculate_risk_score(self, features, prediction_proba):
        """Calculate risk score (0-100, higher = more at risk)"""
        # Get probability of failing (lower grades)
        grade_idx = np.argmax(prediction_proba)
        predicted_grade = self.grade_classes[grade_idx]
        
        # Base risk from grade probability
        if predicted_grade == 'A':
            base_risk = 10
        elif predicted_grade == 'B':
            base_risk = 30
        elif predicted_grade == 'C':
            base_risk = 60
        else:  # D or F
            base_risk = 85
        
        # Adjust based on key features
        feature_dict = {name: features[0][i] for i, name in enumerate(self.feature_names)}
        
        risk_adjustments = 0
        
        # Attendance
        if feature_dict.get('attendance_percentage', 100) < 70:
            risk_adjustments += 10
        
        # Study hours
        if feature_dict.get('weekly_self_study_hours', 10) < 3:
            risk_adjustments += 8
        
        # Assignment submission
        if feature_dict.get('assignment_submission_rate', 1.0) < 0.7:
            risk_adjustments += 12
        
        # Late submissions
        if feature_dict.get('late_submissions_count', 0) > 3:
            risk_adjustments += 7
        
        # LMS engagement
        if feature_dict.get('lms_login_frequency', 20) < 5:
            risk_adjustments += 8
        
        final_risk = min(100, base_risk + risk_adjustments)
        return int(final_risk)
    
    def generate_recommendations(self, features, predicted_grade, cluster):
        """Generate personalized recommendations"""
        recommendations = []
        feature_dict = {name: features[0][i] for i, name in enumerate(self.feature_names)}
        
        # Attendance-based
        attendance = feature_dict.get('attendance_percentage', 100)
        if attendance < 75:
            recommendations.append({
                'category': 'Attendance',
                'priority': 'High',
                'action': f'Your attendance is {attendance:.1f}%. Aim for at least 85% to improve performance.',
                'icon': '📅'
            })
        
        # Study hours
        study_hours = feature_dict.get('weekly_self_study_hours', 10)
        if study_hours < 5:
            recommendations.append({
                'category': 'Study Time',
                'priority': 'High',
                'action': f'Increase weekly self-study from {study_hours:.1f} to at least 7-8 hours.',
                'icon': '📚'
            })
        
        # Assignments
        assignment_rate = feature_dict.get('assignment_submission_rate', 1.0)
        if assignment_rate < 0.85:
            recommendations.append({
                'category': 'Assignments',
                'priority': 'Critical',
                'action': f'Your submission rate is {assignment_rate:.0%}. Submit all assignments on time.',
                'icon': '✍️'
            })
        
        # Late submissions
        late_count = feature_dict.get('late_submissions_count', 0)
        if late_count > 2:
            recommendations.append({
                'category': 'Punctuality',
                'priority': 'Medium',
                'action': f'You have {int(late_count)} late submissions. Improve time management.',
                'icon': '⏰'
            })
        
        # LMS engagement
        lms_logins = feature_dict.get('lms_login_frequency', 20)
        if lms_logins < 10:
            recommendations.append({
                'category': 'Engagement',
                'priority': 'Medium',
                'action': f'Increase LMS engagement. Log in at least 3-4 times per week.',
                'icon': '💻'
            })
        
        # Videos watched
        videos = feature_dict.get('videos_watched', 20)
        if videos < 10:
            recommendations.append({
                'category': 'Resources',
                'priority': 'Medium',
                'action': 'Watch more lecture videos to strengthen understanding.',
                'icon': '🎥'
            })
        
        # Forum activity
        forum = feature_dict.get('forum_activity', 5)
        if forum < 3:
            recommendations.append({
                'category': 'Collaboration',
                'priority': 'Low',
                'action': 'Participate in forum discussions to enhance learning.',
                'icon': '💬'
            })
        
        # Add cluster-specific recommendations
        if cluster == "At Risk":
            recommendations.insert(0, {
                'category': 'Alert',
                'priority': 'Critical',
                'action': 'You are in the At Risk category. Consider meeting with an academic advisor.',
                'icon': '⚠️'
            })
        
        return recommendations
    
    def get_feature_importance(self, features):
        """Get feature contributions for explainability"""
        feature_importance = self.rf_model.feature_importances_
        
        # Get top 5 most important features
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        
        explanations = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            importance = feature_importance[idx]
            value = features[0][idx]
            
            # Format feature name
            display_name = feature_name.replace('_', ' ').title()
            
            explanations.append({
                'feature': display_name,
                'importance': f"{importance:.1%}",
                'value': f"{value:.2f}"
            })
        
        return explanations
    
    def predict_single(self, data_dict):
        """Make prediction for single student"""
        # Prepare features
        features = self.prepare_features(data_dict)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict grade
        grade_encoded = self.rf_model.predict(features_scaled)[0]
        predicted_grade = self.label_encoder.inverse_transform([grade_encoded])[0]
        prediction_proba = self.rf_model.predict_proba(features_scaled)[0]
        confidence = np.max(prediction_proba) * 100
        
        # Predict cluster
        cluster_id = self.kmeans_model.predict(features_scaled)[0]
        cluster_label = self.cluster_labels.get(int(cluster_id), "Unknown")
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(features, prediction_proba)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(features, predicted_grade, cluster_label)
        
        # Feature importance
        feature_importance = self.get_feature_importance(features)
        
        return {
            'predicted_grade': predicted_grade,
            'confidence': round(confidence, 2),
            'risk_score': risk_score,
            'cluster': cluster_label,
            'recommendations': recommendations,
            'feature_importance': feature_importance,
            'all_probabilities': {
                grade: round(prob * 100, 2) 
                for grade, prob in zip(self.grade_classes, prediction_proba)
            }
        }
    
    def predict_batch(self, df):
        """Make predictions for batch of students"""
        results = []
        
        for idx, row in df.iterrows():
            try:
                # Convert row to dict
                data_dict = row.to_dict()
                
                # Make prediction
                prediction = self.predict_single(data_dict)
                
                # Add student identifier if available
                result = {
                    'student_id': data_dict.get('student_id', idx + 1),
                    'predicted_grade': prediction['predicted_grade'],
                    'confidence': prediction['confidence'],
                    'risk_score': prediction['risk_score'],
                    'cluster': prediction['cluster'],
                    'recommendation_count': len(prediction['recommendations'])
                }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'student_id': data_dict.get('student_id', idx + 1),
                    'error': str(e)
                })
        
        return results

# Initialize predictor
predictor = StudentPerformancePredictor()

# Routes
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single student prediction API"""
    try:
        data = request.json
        result = predictor.predict_single(data)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction API"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Make predictions
        results = predictor.predict_batch(df)
        
        return jsonify({'success': True, 'data': results, 'count': len(results)})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/generate_pdf', methods=['POST'])
def generate_pdf():
    """Generate PDF report for single prediction"""
    try:
        data = request.json
        prediction_data = data.get('prediction')
        student_data = data.get('student')
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#6B4423'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#8B6434'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph("Student Performance Analysis Report", title_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Date
        date_text = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        elements.append(Paragraph(date_text, styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))
        
        # Student Information
        elements.append(Paragraph("Student Information", heading_style))
        student_info_data = []
        for key, value in student_data.items():
            display_key = key.replace('_', ' ').title()
            student_info_data.append([display_key, str(value)])
        
        student_table = Table(student_info_data, colWidths=[3*inch, 3*inch])
        student_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#D4C5B0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(student_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Prediction Results
        elements.append(Paragraph("Prediction Results", heading_style))
        prediction_info = [
            ["Predicted Grade", prediction_data['predicted_grade']],
            ["Confidence", f"{prediction_data['confidence']}%"],
            ["Risk Score", f"{prediction_data['risk_score']}/100"],
            ["Student Cluster", prediction_data['cluster']]
        ]
        
        prediction_table = Table(prediction_info, colWidths=[3*inch, 3*inch])
        prediction_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#D4C5B0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(prediction_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Recommendations
        elements.append(Paragraph("Personalized Recommendations", heading_style))
        
        for rec in prediction_data.get('recommendations', []):
            rec_text = f"<b>{rec['icon']} {rec['category']}</b> ({rec['priority']} Priority)<br/>{rec['action']}"
            elements.append(Paragraph(rec_text, styles['Normal']))
            elements.append(Spacer(1, 0.15 * inch))
        
        elements.append(Spacer(1, 0.2 * inch))
        
        # Feature Importance
        elements.append(Paragraph("Key Factors in Prediction", heading_style))
        feature_data = [['Feature', 'Importance', 'Your Value']]
        for feat in prediction_data.get('feature_importance', []):
            feature_data.append([feat['feature'], feat['importance'], feat['value']])
        
        feature_table = Table(feature_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B4423')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(feature_table)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'student_performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/export_batch_csv', methods=['POST'])
def export_batch_csv():
    """Export batch results as CSV"""
    try:
        data = request.json
        results = data.get('results', [])
        
        df = pd.DataFrame(results)
        
        # Create CSV in memory
        buffer = BytesIO()
        df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mimetype='text/csv'
        )
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 60)
    print("🎓 STUDENT PERFORMANCE AI - STARTING APPLICATION")
    print("=" * 60)
    print("\n✅ Application ready!")
    print("📍 URL: http://localhost:5000")
    print("\n" + "=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
