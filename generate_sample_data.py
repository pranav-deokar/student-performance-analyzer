"""
Sample Dataset Generator
Use this if you don't have the actual dataset
Generates realistic synthetic student performance data
"""

import pandas as pd
import numpy as np
import os

def generate_sample_dataset(n_samples=20000):
    """Generate synthetic student performance dataset"""
    
    print(f"Generating {n_samples} sample records...")
    
    np.random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # Determine student performance level
        performance_level = np.random.choice(['high', 'medium', 'low'], p=[0.25, 0.50, 0.25])
        
        if performance_level == 'high':
            attendance = np.random.uniform(85, 100)
            study_hours = np.random.uniform(8, 15)
            participation = np.random.uniform(4, 5)
            lms_logins = np.random.randint(15, 30)
            videos = np.random.randint(20, 40)
            assignment_rate = np.random.uniform(0.85, 1.0)
            quizzes = np.random.randint(8, 15)
            forum = np.random.uniform(3, 8)
            late_submissions = np.random.randint(0, 2)
            previous_grade = np.random.uniform(80, 95)
            
        elif performance_level == 'medium':
            attendance = np.random.uniform(70, 85)
            study_hours = np.random.uniform(4, 8)
            participation = np.random.uniform(3, 4)
            lms_logins = np.random.randint(8, 15)
            videos = np.random.randint(10, 20)
            assignment_rate = np.random.uniform(0.70, 0.85)
            quizzes = np.random.randint(5, 8)
            forum = np.random.uniform(1, 3)
            late_submissions = np.random.randint(1, 4)
            previous_grade = np.random.uniform(65, 80)
            
        else:  # low
            attendance = np.random.uniform(40, 70)
            study_hours = np.random.uniform(1, 4)
            participation = np.random.uniform(1, 3)
            lms_logins = np.random.randint(2, 8)
            videos = np.random.randint(3, 10)
            assignment_rate = np.random.uniform(0.40, 0.70)
            quizzes = np.random.randint(2, 5)
            forum = np.random.uniform(0, 1)
            late_submissions = np.random.randint(3, 8)
            previous_grade = np.random.uniform(45, 65)
        
        # Calculate total score (weighted average)
        total_score = (
            attendance * 0.15 +
            study_hours * 3.5 +
            participation * 8 +
            assignment_rate * 30 +
            previous_grade * 0.5
        )
        
        # Add some noise
        total_score += np.random.normal(0, 5)
        total_score = max(0, min(100, total_score))
        
        # Determine grade
        if total_score >= 85:
            grade = 'A'
        elif total_score >= 70:
            grade = 'B'
        elif total_score >= 55:
            grade = 'C'
        else:
            grade = 'D'
        
        record = {
            'student_id': i + 1,
            'weekly_self_study_hours': round(study_hours, 2),
            'attendance_percentage': round(attendance, 2),
            'class_participation': round(participation, 2),
            'lms_login_frequency': int(lms_logins),
            'videos_watched': int(videos),
            'assignment_submission_rate': round(assignment_rate, 3),
            'quiz_attempts': int(quizzes),
            'forum_activity': round(forum, 2),
            'late_submissions_count': int(late_submissions),
            'previous_grade': round(previous_grade, 2),
            'total_score': round(total_score, 2),
            'grade': grade
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/student_performance_dataset_20000.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset generated successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total records: {len(df)}")
    print(f"   Grade distribution:")
    print(df['grade'].value_counts().sort_index())
    print(f"\n   Sample records:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("📊 SAMPLE DATASET GENERATOR")
    print("=" * 60)
    print("\nThis will generate a synthetic dataset for testing.")
    print("Use this if you don't have the actual student data.\n")
    
    generate_sample_dataset(20000)
    
    print("\n" + "=" * 60)
    print("✅ DONE! You can now run: python train.py")
    print("=" * 60)
