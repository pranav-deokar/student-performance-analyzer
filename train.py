"""
Student Performance AI - Training Script
Trains supervised and unsupervised models on student performance data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceTrainer:
    def __init__(self, data_path='data/student_performance_dataset_20000.csv'):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = None
        self.label_encoder = None
        self.rf_model = None
        self.kmeans_model = None
        self.feature_names = None
        self.cluster_labels = {
            0: "High Performer",
            1: "Average Performer", 
            2: "At Risk"
        }
        
    def load_data(self):
        """Load and validate dataset"""
        print("📂 Loading dataset...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} records")
        print(f"   Columns: {list(self.df.columns)}")
        return self
    
    def preprocess_data(self):
        """Handle missing values and prepare features"""
        print("\n🔧 Preprocessing data...")
        
        # Drop student_id if exists
        if 'student_id' in self.df.columns:
            self.df = self.df.drop('student_id', axis=1)
        
        # Handle missing values
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Separate features and target
        if 'grade' in self.df.columns:
            self.y = self.df['grade']
            self.X = self.df.drop(['grade', 'total_score'], axis=1, errors='ignore')
        else:
            raise ValueError("Target column 'grade' not found")
        
        # Encode categorical column: previous_grade
        if 'previous_grade' in self.X.columns:
            le_prev = LabelEncoder()
            self.X['previous_grade'] = le_prev.fit_transform(self.X['previous_grade'])
        self.feature_names = list(self.X.columns)
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"✅ Features: {len(self.feature_names)}")
        print(f"   Target classes: {list(self.label_encoder.classes_)}")
        
        return self
    
    def train_supervised_model(self):
        """Train Random Forest classifier"""
        print("\n🤖 Training supervised model (Random Forest)...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"\n📊 Classification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        return self
    
    def train_unsupervised_model(self):
        """Train K-Means clustering"""
        print("\n🎯 Training unsupervised model (K-Means)...")
        
        # Use scaled features
        X_scaled = self.scaler.transform(self.X)
        
        # Find optimal clusters using silhouette score
        best_score = -1
        best_k = 3
        
        for k in range(2, 6):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans_temp.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"   Optimal clusters: {best_k} (silhouette score: {best_score:.3f})")
        
        # Train final K-Means
        self.kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.kmeans_model.fit(X_scaled)
        
        # Analyze clusters to assign meaningful labels
        clusters = self.kmeans_model.predict(X_scaled)
        self.df['cluster'] = clusters
        
        cluster_stats = []
        for i in range(best_k):
            cluster_data = self.df[self.df['cluster'] == i]
            avg_attendance = cluster_data['attendance_percentage'].mean()
            avg_study = cluster_data['weekly_self_study_hours'].mean()
            avg_assignments = cluster_data['assignment_submission_rate'].mean()
            
            # Calculate composite score
            composite = (avg_attendance + avg_study*10 + avg_assignments*100) / 3
            cluster_stats.append({
                'cluster': i,
                'composite': composite,
                'attendance': avg_attendance,
                'study_hours': avg_study,
                'assignments': avg_assignments
            })
        
        # Sort clusters by composite score
        cluster_stats.sort(key=lambda x: x['composite'], reverse=True)
        
        # Assign labels based on performance
        self.cluster_labels = {}
        labels_to_assign = ["High Performer", "Average Performer", "At Risk"]
        
        for idx, stat in enumerate(cluster_stats):
            label_idx = min(idx, len(labels_to_assign) - 1)
            self.cluster_labels[stat['cluster']] = labels_to_assign[label_idx]
        
        print(f"✅ Clustering completed")
        print(f"\n   Cluster Assignments:")
        for cluster_id, label in self.cluster_labels.items():
            stats = [s for s in cluster_stats if s['cluster'] == cluster_id][0]
            print(f"   • Cluster {cluster_id} → {label}")
            print(f"     - Avg Attendance: {stats['attendance']:.1f}%")
            print(f"     - Avg Study Hours: {stats['study_hours']:.1f}h/week")
            print(f"     - Avg Assignment Rate: {stats['assignments']:.1%}")
        
        return self
    
    def save_models(self):
        """Save all models and preprocessors"""
        print("\n💾 Saving models and preprocessors...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save models
        joblib.dump(self.rf_model, 'models/random_forest_model.pkl')
        joblib.dump(self.kmeans_model, 'models/kmeans_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'cluster_labels': self.cluster_labels,
            'grade_classes': list(self.label_encoder.classes_)
        }
        joblib.dump(metadata, 'models/metadata.pkl')
        
        print("✅ All models saved successfully:")
        print("   • random_forest_model.pkl")
        print("   • kmeans_model.pkl")
        print("   • scaler.pkl")
        print("   • label_encoder.pkl")
        print("   • metadata.pkl")
        
        return self
    
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        print("=" * 60)
        print("🎓 STUDENT PERFORMANCE AI - TRAINING PIPELINE")
        print("=" * 60)
        
        try:
            self.load_data()
            self.preprocess_data()
            self.train_supervised_model()
            self.train_unsupervised_model()
            self.save_models()
            
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\n📝 Next steps:")
            print("   1. Run: python app.py")
            print("   2. Open: http://localhost:5000")
            print("   3. Start making predictions!")
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error during training: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = StudentPerformanceTrainer()
    trainer.run_training_pipeline()
