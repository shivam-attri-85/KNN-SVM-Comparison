# 🚀 QUICK START GUIDE

Follow these simple steps to run the KNN vs SVM Analyzer on your machine.

---

## Step 1: Open Terminal/Command Prompt

Navigate to the project directory:

```bash
cd "/home/shivam/Pictures/CA2 PROJECT/KNNSVMANALYZER"
```

Windows users: You can also right-click in the folder and select "Open in Terminal" or "Open PowerShell here"

---

## Step 2: Create Virtual Environment (Optional but Recommended)

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**What it does**: Installs Streamlit, pandas, scikit-learn, matplotlib, and seaborn

**Expected time**: 2-5 minutes (depending on your internet)

---

## Step 4: Run the Application

```bash
streamlit run app.py
```

**What happens**:
- A local web server starts
- Your default browser opens automatically
- App loads at: http://localhost:8501/

---

## Step 5: Upload Sample Dataset

1. Download or use the included `sample_iris.csv`
2. Click "Browse files" in the sidebar
3. Select the CSV file
4. Let the app load the data

---

## Step 6: Start the ML Pipeline

1. View data in Step 1
2. Analyze data in Step 2
3. Follow each step sequentially
4. Adjust parameters (K value, test size, encoding method)
5. View results and comparison

---

## First Run Checklist

- [ ] Python 3.8+ installed?
- [ ] In the correct project directory?
- [ ] Virtual environment activated?
- [ ] Dependencies installed without errors?
- [ ] Streamlit installed correctly?
- [ ] CSV file ready to upload?

---

## Troubleshooting

### "python: command not found" on macOS/Linux?
**Solution**: Use `python3` instead of `python`
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run app.py
```

### "ModuleNotFoundError: No module named 'streamlit'"?
**Solution**: Make sure virtual environment is activated and dependencies installed
```bash
pip install -r requirements.txt  # Run again
```

### "Port 8501 is already in use"?
**Solution**: Use a different port
```bash
streamlit run app.py --server.port 8502
```

### App not opening in browser?
**Solution**: Open manually
```
Go to: http://localhost:8501
```

### CSV not uploading?
**Solution**: Ensure CSV is valid
- File extension must be `.csv`
- Has headers/column names
- Use UTF-8 encoding
- Try sample_iris.csv first

---

## Data Format Requirements

Your CSV file should have:
- ✅ Column headers (first row)
- ✅ Numerical features (for most algorithms)
- ✅ One target column (what to predict)
- ✅ No special characters in headers

### Example Format:

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
...
```

---

## Test With Sample Data

A `sample_iris.csv` file is included. Use it to test:

1. Upload: `sample_iris.csv`
2. Select target: `species`
3. Run through all steps
4. Compare models

**Expected Result**: 
- KNN Accuracy: ~96-98%
- SVM Accuracy: ~95-97%

---

## Stop the App

Press `CTRL+C` in the terminal where you ran `streamlit run app.py`

---

## Next Steps

✅ Run with sample data first  
✅ Test with your own dataset  
✅ Adjust parameters and rerun  
✅ Compare different encodings/kernels  
✅ Download results  

---

## Need Help?

1. Check README.md for detailed documentation
2. Review app.py comments for implementation details
3. Ensure CSV format is correct
4. Check terminal output for error messages
5. Try with sample_iris.csv first

---

**You're ready to go!** 🎉

Run: `streamlit run app.py`
