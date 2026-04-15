# 📊 SAMPLE DATASETS GUIDE

5 ready-to-use sample datasets included! Pick any one to test your ML pipeline.

---

## 📋 Available Datasets

### 1. **sample_iris.csv** ⭐ (Best for Beginners)
- **Classes**: 3 (species: setosa, versicolor, virginica)
- **Features**: 4 (sepal_length, sepal_width, petal_length, petal_width)
- **Samples**: 150
- **Target Column**: `species`
- **Expected KNN Accuracy**: 96-98%
- **Expected SVM Accuracy**: 95-97%
- **Use When**: Testing & learning, very clean data
- **⚠️ Note**: This is the easiest dataset - START HERE!

**Why good for learning?**
- ✅ Perfectly balanced classes
- ✅ No missing values
- ✅ High accuracy (easy to understand)
- ✅ Industry-standard dataset

---

### 2. **sample_titanic.csv** (Binary Classification)
- **Classes**: 2 (Survived: 0 or 1)
- **Features**: 5 (Pclass, Age, Fare, Sex, Embarked)
- **Samples**: 100
- **Target Column**: `Survived`
- **Expected KNN Accuracy**: 75-82%
- **Expected SVM Accuracy**: 77-84%
- **Use When**: Binary classification learning
- **⚠️ Note**: Realistic data with imbalance

**Real-world scenario**: Titanic survival prediction

---

### 3. **sample_breast_cancer.csv** (Medical Data)
- **Classes**: 2 (Malignant or Benign)
- **Features**: 11 (various tumor measurements)
- **Samples**: 100
- **Target Column**: `diagnosis`
- **Expected KNN Accuracy**: 85-92%
- **Expected SVM Accuracy**: 88-94%
- **Use When**: Medical/healthcare ML projects
- **⚠️ Note**: Important for precision!

**Real-world scenario**: Cancer diagnosis prediction

---

### 4. **sample_wine.csv** (Multi-class)
- **Classes**: 3 (wine_class: 1, 2, or 3)
- **Features**: 11 (alcohol, malic_acid, ash, etc.)
- **Samples**: 30
- **Target Column**: `wine_class`
- **Expected KNN Accuracy**: 80-90%
- **Expected SVM Accuracy**: 82-92%
- **Use When**: Multi-class learning
- **⚠️ Note**: Small dataset, higher variance

**Real-world scenario**: Wine quality classification

---

### 5. **sample_extended_iris.csv** (3-class)
- **Classes**: 3 (Class1, Class2, Class3)
- **Features**: 5 (custom feature1-5)
- **Samples**: 71
- **Target Column**: `classification`
- **Expected KNN Accuracy**: 93-97%
- **Expected SVM Accuracy**: 91-95%
- **Use When**: Multi-class non-binary problem
- **⚠️ Note**: Semi-real data, balanced classes

**Real-world scenario**: Generic 3-class classification

---

## 🎯 How to Choose Dataset?

### I'm a Beginner:
👉 **Use `sample_iris.csv`** - Easiest, best results!

### I want Binary Classification:
👉 **Use `sample_titanic.csv` or `sample_breast_cancer.csv`**

### I want 3+ Classes:
👉 **Use `sample_wine.csv` or `sample_extended_iris.csv`**

### I have My Own Dataset:
👉 **Upload via sidebar** - App auto-detects target!

---

## 🚀 Quick Test (2 minutes)

### Step 1: Open App
```bash
streamlit run app.py
```

### Step 2: Upload Dataset
- Click "Browse files" in sidebar
- Select any sample_*.csv file

### Step 3: Run Pipeline
- The app will **auto-select target column**
- Scroll through all 10 steps
- View results at end

### Step 4: Try All Datasets
- Each will show different accuracy
- Iris = Highest accuracy
- Titanic = Realistic scenario
- Breast Cancer = Medical importance
- Wine = Multi-class learning
- Extended Iris = Generic 3-class

---

## 📊 Comparison Table

| Dataset | Classes | Difficulty | Target | Best Accuracy | Use Case |
|---------|---------|-----------|--------|---------------|----------|
| Iris | 3 | ★☆☆ | species | 96-98% | Learning |
| Titanic | 2 | ★★☆ | Survived | 77-84% | Binary |
| Breast Cancer | 2 | ★★☆ | diagnosis | 88-94% | Medical |
| Wine | 3 | ★★★ | wine_class | 80-90% | Multi-class |
| Extended Iris | 3 | ★★☆ | classification | 93-97% | 3-class |

---

## 💡 Pro Tips

### 1. Start Simple
- Begin with `iris` (highest accuracy)
- Build confidence
- Then try harder datasets

### 2. Compare Models
- Try different K values (5, 7, 9)
- Try different SVM kernels (linear, rbf)
- See which works best per dataset

### 3. Try Preprocessing
- **With outlier handling**: See if accuracy goes up/down
- **Different encodings**: Label vs One-Hot
- **Different test sizes**: 20% vs 30%

### 4. Observe Patterns
- Iris: Always high accuracy
- Titanic: More realistic results
- Breast Cancer: Medical importance
- Wine: Feature engineering matters

---

## 🎓 For Viva Preparation

**Use these datasets to demonstrate:**

1. **Iris** → "I achieved 97% accuracy with KNN"
2. **Titanic** → "Binary classification with 80% accuracy"
3. **Breast Cancer** → "Medical ML application" 
4. **Wine** → "Multi-class classification"
5. **Your Own** → "Real-world data preprocessing"

---

## 📝 Dataset Features Explained

### sample_iris.csv
```
sepal_length, sepal_width  → Flower measurements
petal_length, petal_width  → Flower petal size
species                     → Target (3 classes)
```

### sample_titanic.csv
```
Pclass    → Ticket class (1st, 2nd, 3rd)
Age       → Passenger age
Fare      → Ticket fare
Sex       → Male/Female (auto-encoded)
Embarked  → Port (S, C, Q - auto-encoded)
Survived  → Target (0=No, 1=Yes)
```

### sample_breast_cancer.csv
```
radius_mean         → Tumor radius
texture_mean        → Texture feature
perimeter_mean      → Perimeter
area_mean           → Area measurement
smoothness_mean     → Surface smoothness
diagnosis           → Target (Malignant/Benign)
```

### sample_wine.csv
```
alcohol             → Alcohol percentage
malic_acid          → Acidity level
ash, alkalinity     → Ash content
magnesium           → Mineral content
phenols, etc        → Chemical properties
wine_class          → Target (1, 2, or 3)
```

### sample_extended_iris.csv
```
feature1-5          → 5 custom features
classification      → Target (Class1, Class2, Class3)
```

---

## ✅ Testing Checklist

- [ ] Upload `sample_iris.csv` - Get 95%+ accuracy
- [ ] Upload `sample_titanic.csv` - Get 75%+ accuracy
- [ ] Try `sample_breast_cancer.csv` - Medical data
- [ ] Try `sample_wine.csv` - 3-class problem
- [ ] Try `sample_extended_iris.csv` - Alternative format
- [ ] Try different K values (5, 7, 9, 15)
- [ ] Try different SVM kernels (linear, rbf, poly)
- [ ] Try with/without outlier handling
- [ ] Try Label vs One-Hot encoding
- [ ] Download results from one run

---

## 🎯 Expected Outputs

### With sample_iris.csv:
```
✅ KNN Accuracy: 96-98%
✅ SVM Accuracy: 95-97%
✅ Both models perform well
✅ Few misclassifications
```

### With sample_titanic.csv:
```
✅ KNN Accuracy: 75-82%
✅ SVM Accuracy: 77-84%
✅ More realistic results
✅ Some imbalance visible
```

### With sample_breast_cancer.csv:
```
✅ KNN Accuracy: 85-92%
✅ SVM Accuracy: 88-94%
✅ High importance for accuracy
✅ Medical significance
```

---

## 🚀 Next Steps

1. **Open app**: `streamlit run app.py`
2. **Pick dataset**: Use any sample_*.csv
3. **Auto-select target**: App will find it automatically
4. **Run pipeline**: Scroll through all steps
5. **View results**: Compare accuracies
6. **Try another**: Upload different dataset
7. **Compare**: See how different data affects results

---

**Ready to start?** Pick any dataset and run the app! 🎉

All 5 datasets are ready-to-use. No cleaning needed. Just upload and run! 💪
