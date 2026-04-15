# 🤖 KNN vs SVM Machine Learning Analyzer

A complete, step-by-step Machine Learning web application built with **Streamlit** that compares K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) algorithms on user-uploaded CSV datasets.

## 📋 Features

✅ **Complete ML Pipeline** - 10 sequential steps from data loading to model evaluation  
✅ **Interactive UI** - Simple, beginner-friendly Streamlit interface  
✅ **Data Visualization** - Correlation heatmaps, confusion matrices, accuracy charts  
✅ **Model Comparison** - Side-by-side accuracy metrics and performance analysis  
✅ **Flexible Configuration** - Adjustable K value, test size, encoding method, and SVM kernel  
✅ **Results Export** - Download analysis results as text file  
✅ **Viva-Ready Code** - Well-commented, clean, production-ready code  

## 🎯 ML Pipeline Steps

The app implements a complete 10-step ML pipeline:

1. **Data Loading** - Upload CSV and display basic info
2. **Exploratory Data Analysis (EDA)** - Statistics, correlation heatmap
3. **Missing Value Handling** - Fill with mean/median/mode
4. **Categorical Encoding** - Label Encoding or One-Hot Encoding
5. **Outlier Detection** - IQR method detection and capping
6. **Feature Scaling** - StandardScaler normalization
7. **Train-Test Split** - Configurable train/test split
8. **Model Training** - KNN (k=5) and SVM (linear kernel)
9. **Predictions** - Generate predictions on test data
10. **Model Evaluation** - Accuracy, confusion matrix, classification reports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation & Setup

#### Step 1: Clone/Navigate to Project Directory
```bash
cd "/home/shivam/Pictures/CA2 PROJECT/KNNSVMANALYZER"
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your default browser.

---

## 📊 How to Use

### 1. Upload Dataset
- Click on the file upload widget in the sidebar
- Select a CSV file with features and a target column
- The app will automatically load and validate the data

### 2. Explore Data
- View first N rows (adjustable with slider)
- Check dataset shape and data types
- Review basic statistics

### 3. Analyze Data
- View correlation heatmap for numerical features
- Check for missing values
- Understand data distribution

### 4. Configure ML Pipeline
- Choose encoding method (Label or One-Hot)
- Select target column (what to predict)
- Adjust K value for KNN algorithm
- Choose SVM kernel type
- Set test-train split ratio

### 5. Train Models
- Both KNN and SVM are trained automatically
- View training progress and confirmation

### 6. View Results
- See sample predictions
- Compare model accuracies
- Review confusion matrices
- Check classification reports
- Download results

---

## 📁 Sample Datasets

You can test the app with any classification dataset. Popular options:

### Example 1: Iris Dataset
```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('iris.csv', index=False)
```

### Example 2: Titanic Dataset
- Download from: https://kaggle.com/c/titanic
- Clean and prepare before using

### Example 3: Breast Cancer Dataset
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('breast_cancer.csv', index=False)
```

---

## 🔧 Configuration Options

### KNN Parameters
- **K Value**: Number of neighbors to consider (1-20)
  - Lower K: More sensitive to noise
  - Higher K: Smoother decision boundaries

### SVM Parameters
- **Kernel Type**: 
  - `linear`: Best for linearly separable data
  - `rbf`: Non-linear kernel, works for complex patterns
  - `poly`: Polynomial kernel for medium complexity

### Data Split
- **Test Size**: Percentage of data for testing (10%-50%)
- **Random State**: For reproducibility (default: 42)

### Encoding Methods
- **Label Encoding**: Convert categories to integers (0, 1, 2...)
  - Use when: Ordinal relationships exist
- **One-Hot Encoding**: Create binary columns for each category
  - Use when: No ordinal relationship

---

## 📊 Understanding Outputs

### Accuracy Metrics
- **Accuracy**: % of correct predictions
- **Precision**: % of positive predictions that were correct
- **Recall**: % of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall

### Confusion Matrix
Shows True Positives, True Negatives, False Positives, False Negatives

### Classification Report
Detailed metrics for each class in multi-class classification

---

## 💻 Code Structure

```
KNNSVMANALYZER/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Components in app.py

1. **Imports & Setup** - All necessary libraries
2. **Page Configuration** - Streamlit UI setup
3. **File Upload** - Sidebar file upload widget
4. **Step 1-10** - Complete ML pipeline with explanations
5. **Session State** - Data persistence across reruns

---

## 🐛 Troubleshooting

### Issue: "No module named 'streamlit'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Port 8501 already in use
**Solution**: Use different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: CSV file not loading
**Solution**: Ensure CSV format is correct
- File must be valid CSV format
- If upload fails, check file encoding (use UTF-8)
- Ensure file has proper headers

### Issue: Memory error with large datasets
**Solution**: 
- Use a smaller dataset
- Reduce number of features
- Sample the data before analysis

---

## 📚 Educational Value

**Perfect for College Viva!** This project demonstrates:

✅ Understanding of ML pipeline workflow  
✅ Data preprocessing techniques  
✅ Algorithm implementation (KNN & SVM)  
✅ Model evaluation and comparison  
✅ Web UI development with Streamlit  
✅ Data visualization and analysis  
✅ Python programming best practices  
✅ Clean, well-documented code  

### Interview Questions Covered:

1. What is the ML pipeline?
   - _Answer: 10-step process from data loading to model evaluation_

2. Why is feature scaling important?
   - _Answer: Ensures all features have similar impact on model_

3. What is the difference between KNN and SVM?
   - _KNN: Instance-based, finds k nearest neighbors_
   - _SVM: Finds optimal decision boundary/hyperplane_

4. How to handle missing values?
   - _Answer: Fill with mean/median for numerical, mode for categorical_

5. What does confusion matrix show?
   - _Answer: TP, TN, FP, FN for model prediction evaluation_

---

## 🎓 Key Concepts Explained

### K-Nearest Neighbors (KNN)
- **How it works**: Classifies based on k nearest training examples
- **Pros**: Simple, effective on small datasets
- **Cons**: Slow on large datasets, sensitive to feature scaling
- **Best for**: Non-linear decision boundaries, small-medium datasets

### Support Vector Machine (SVM)
- **How it works**: Finds optimal hyperplane to separate classes
- **Pros**: Effective on high-dimensional data, good with large datasets
- **Cons**: Slower on very large datasets, needs feature scaling
- **Best for**: Binary/multi-class classification, high-dimensional data

### Feature Scaling
- **Why**: Algorithms like KNN and SVM are distance-based
- **StandardScaler**: Transforms to mean=0, std=1
- **Formula**: (X - mean) / standard_deviation

### Train-Test Split
- **Why**: Evaluate on unseen data for fair assessment
- **Typical ratios**: 80-20, 70-30, 75-25
- **Test set size**: Should be large enough for reliable estimates

---

## 📝 Notes & Tips

- **Always explore data first** - EDA can reveal issues early
- **Handle missing values** - Don't ignore them
- **Scale features** - Essential for distance-based algorithms
- **Check class imbalance** - May affect model performance
- **Use cross-validation** - More robust than single train-test split
- **Try different K values** - Find optimal for your data
- **Tune SVM kernel** - Different kernels work best for different data

---

## 🔗 Additional Resources

- **Scikit-Learn Documentation**: https://scikit-learn.org/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **ML Best Practices**: https://machinelearningmastery.com/
- **Pandas Guide**: https://pandas.pydata.org/docs/

---

## 📄 License

This project is free to use for educational purposes.

---

## 👨‍💼 Author Notes

Created for college ML projects and viva preparations. The app emphasizes:
- Clean, readable code
- Step-by-step explanations
- Interactive visualization
- Best practices in data science

**Ready to impress your viva panel!** 🎓

---

## 📞 Support

If you encounter issues:
1. Check Python version (3.8+)
2. Verify all dependencies installed
3. Try with sample dataset first
4. Check error messages in terminal

---

**Last Updated**: April 2026  
**Python Version**: 3.8+  
**Status**: Production Ready ✅
