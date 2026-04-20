"""
Automated Test Script
Tests the complete system functionality
"""

import os
import sys

def check_file_structure():
    """Verify all required files exist"""
    print("🔍 Checking file structure...")
    
    required_files = [
        'app.py',
        'train.py',
        'requirements.txt',
        'README.md',
        'generate_sample_data.py',
        'templates/index.html',
        'templates/dashboard.html',
        'static/css/style.css',
        'static/js/main.js',
        'static/js/dashboard.js'
    ]
    
    required_dirs = [
        'data',
        'templates',
        'static',
        'static/css',
        'static/js'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ Directory: {directory}")
        else:
            print(f"   ❌ Missing directory: {directory}")
            all_good = False
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ File: {file}")
        else:
            print(f"   ❌ Missing file: {file}")
            all_good = False
    
    return all_good

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'sklearn',
        'joblib',
        'reportlab'
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} not installed")
            all_installed = False
    
    return all_installed

def check_dataset():
    """Check if dataset exists"""
    print("\n📊 Checking dataset...")
    
    dataset_path = 'data/student_performance_dataset_20000.csv'
    
    if os.path.exists(dataset_path):
        print(f"   ✅ Dataset found: {dataset_path}")
        
        # Try to read it
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"   ✅ Dataset readable: {len(df)} records")
            
            required_columns = [
                'weekly_self_study_hours',
                'attendance_percentage',
                'assignment_submission_rate',
                'grade'
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
                return False
            else:
                print(f"   ✅ All required columns present")
                return True
                
        except Exception as e:
            print(f"   ❌ Error reading dataset: {e}")
            return False
    else:
        print(f"   ⚠️  Dataset not found")
        print(f"   💡 Run: python generate_sample_data.py")
        return False

def check_models():
    """Check if models are trained"""
    print("\n🤖 Checking trained models...")
    
    model_files = [
        'models/random_forest_model.pkl',
        'models/kmeans_model.pkl',
        'models/scaler.pkl',
        'models/label_encoder.pkl',
        'models/metadata.pkl'
    ]
    
    all_exist = True
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"   ✅ {model_file}")
        else:
            print(f"   ⚠️  {model_file} not found")
            all_exist = False
    
    if not all_exist:
        print(f"   💡 Run: python train.py")
    
    return all_exist

def run_system_test():
    """Run complete system test"""
    print("\n" + "=" * 60)
    print("🧪 STUDENT PERFORMANCE AI - SYSTEM TEST")
    print("=" * 60 + "\n")
    
    # Check file structure
    files_ok = check_file_structure()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check dataset
    data_ok = check_dataset()
    
    # Check models
    models_ok = check_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    print(f"\n   File Structure: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"   Dependencies:   {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"   Dataset:        {'✅ PASS' if data_ok else '⚠️  NOT READY'}")
    print(f"   Models:         {'✅ PASS' if models_ok else '⚠️  NOT TRAINED'}")
    
    print("\n" + "=" * 60)
    
    if files_ok and deps_ok and data_ok and models_ok:
        print("✅ SYSTEM READY!")
        print("\n🚀 You can now run: python app.py")
    elif files_ok and deps_ok:
        print("⚠️  SYSTEM NEEDS SETUP")
        if not data_ok:
            print("\n📝 Next step: python generate_sample_data.py")
        if not models_ok:
            print("\n📝 Next step: python train.py")
    else:
        print("❌ SYSTEM NOT READY")
        if not deps_ok:
            print("\n📝 Next step: pip install -r requirements.txt")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_system_test()
