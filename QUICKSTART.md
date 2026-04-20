# 🚀 QUICK START GUIDE

## Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data (if you don't have the dataset)
```bash
python generate_sample_data.py
```
**OR** place your own dataset in `data/student_performance_dataset_20000.csv`

### Step 3: Train Models
```bash
python train.py
```

### Step 4: Run Application
```bash
python app.py
```

### Step 5: Open Browser
Navigate to: **http://localhost:5000**

---

## 📝 What Each File Does

- `generate_sample_data.py` - Creates synthetic dataset for testing
- `train.py` - Trains ML models and saves them
- `app.py` - Runs the web application
- `requirements.txt` - Lists all Python dependencies

---

## ⚡ Testing the System

### Test Single Prediction

1. Go to Dashboard
2. Enter these values:
   - Attendance: `85`
   - Study Hours: `10`
   - Assignment Rate: `0.9`
3. Click "Predict Performance"

### Test Batch Processing

1. Click "Batch Processing" tab
2. Upload the file from `data/student_performance_dataset_20000.csv`
3. Click "Process Batch"

---

## 🎯 Expected Results

After training, you should see:
- ✅ Accuracy: ~85-95%
- ✅ 3 student clusters identified
- ✅ Models saved in `models/` folder

---

## ❓ Troubleshooting

**Problem**: "No module named 'flask'"
**Solution**: Run `pip install -r requirements.txt`

**Problem**: "Dataset not found"
**Solution**: Run `python generate_sample_data.py` first

**Problem**: "Models not found"
**Solution**: Run `python train.py` before `python app.py`

---

## 📞 Need Help?

Check the full README.md for detailed documentation.

---

**Enjoy the Student Performance AI System! 🎓**
