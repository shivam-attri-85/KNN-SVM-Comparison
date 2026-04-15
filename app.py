"""
KNN vs SVM Analyzer - Complete Machine Learning Pipeline
A step-by-step ML web app using Streamlit to compare KNN and SVM algorithms
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="KNN vs SVM Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom title and description
st.title("🤖 KNN vs SVM Machine Learning Analyzer")
st.markdown("""
    **Compare K-Nearest Neighbors and Support Vector Machine algorithms** on your custom dataset.
    Follow the step-by-step ML pipeline for complete data analysis and model evaluation.
""")

# ============================================================================
# SIDEBAR - FILE UPLOAD
# ============================================================================

st.sidebar.header("📁Upload Dataset")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file with features and a target column"
)

# Store data in session state for persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error reading file: {e}")

# ============================================================================
# MAIN APP - CHECK IF DATA IS LOADED
# ============================================================================

if st.session_state.df is None:
    st.warning("⚠️ Please upload a CSV file to get started!")
    st.info("""
        **How to use this app:**
        1. Upload a CSV file from the sidebar
        2. Follow each step of the ML pipeline
        3. View results and compare models
        
        **Sample CSV Format:**
        - Include numerical features and one target column (for classification)
        - Example: Iris.csv, Titanic.csv, or any classification dataset
    """)
else:
    df = st.session_state.df
    
    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    
    st.header("📥 DataSet Loading")
    st.markdown("Display the uploaded dataset and basic information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Shape")
        st.metric(label="Rows", value=df.shape[0])
        st.metric(label="Columns", value=df.shape[1])
    
    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)
    
    # Display first N rows
    n_rows = st.slider("Number of rows to display:", min_value=5, max_value=50, value=10)
    st.subheader("First N Rows")
    st.dataframe(df.head(n_rows), use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # STEP 2: DATA ANALYSIS (EDA)
    # ========================================================================
    
    st.header("📊Exploratory Data Analysis (EDA)")
    st.markdown("Understand the structure and distributions of your data")
    
    # Basic statistics
    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values
    st.subheader("❓ Missing Values Detection")
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_data, use_container_width=True)
    
    # Correlation heatmap (for numerical columns)
    st.subheader("🔗 Correlation Heatmap")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ Not enough numerical columns for correlation analysis")
    
    st.markdown("---")
    
    # ========================================================================
    # STEP 3: DATA PREPROCESSING
    # ========================================================================
    
    st.header("🔧 Data Preprocessing")
    
    # Step 3a: Handle Missing Values
    st.subheader("Handling Missing Values")
    
    df_processed = df.copy()
    
    # Find columns with missing values
    cols_with_missing = df_processed.isnull().sum()
    cols_with_missing = cols_with_missing[cols_with_missing > 0].index.tolist()
    
    if len(cols_with_missing) > 0:
        st.warning(f"⚠️ Missing values detected in: {cols_with_missing}")
        
        for col in cols_with_missing:
            if df_processed[col].dtype in [np.float64, np.int64]:
                # Fill using mean for numerical columns
                mean_val = df_processed[col].mean()
                df_processed[col].fillna(mean_val, inplace=True)
                st.success(f"✅ Filled missing values in '{col}' with mean: {mean_val:.2f}")
            else:
                # Fill using mode for categorical columns
                mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                df_processed[col].fillna(mode_val, inplace=True)
                st.success(f"✅ Filled missing values in '{col}' with mode: {mode_val}")
    else:
        st.info("✅ No missing values found!")
    
    st.markdown("")
    
    # Step 3b: Encoding categorical variables
    st.subheader("Encoding Categorical Variables")
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) > 0:
        st.write(f"Categorical columns found: {categorical_cols}")
        
        encoding_method = st.radio(
            "Choose encoding method:",
            ["Label Encoding", "One-Hot Encoding"],
            key="encoding_method",
            horizontal=True
        )
        
        if encoding_method == "Label Encoding":
            st.info("Using Label Encoding: Convert categories to integers (0, 1, 2, ...)")
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
                st.success(f"✅ Label encoded '{col}'")
        
        else:  # One-Hot Encoding
            st.info("Using One-Hot Encoding: Create binary columns for each category")
            df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
            st.success(f"✅ One-hot encoded {len(categorical_cols)} categorical column(s)")
    else:
        st.info("✅ No categorical columns found!")
    
    st.markdown("")
    
    # Drop any remaining rows with NaN values
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna()
    rows_after_dropna = len(df_processed)
    
    if initial_rows > rows_after_dropna:
        st.warning(f"⚠️ Dropped {initial_rows - rows_after_dropna} rows with remaining missing values")
    
    st.markdown("")
    
    # Step 3c: Outlier Detection and Handling
    st.subheader("Outlier Detection and Handling (IQR Method)")
    
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        outliers_detected = False
        rows_before = len(df_processed)
        
        for col in numerical_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            
            if outlier_count > 0:
                outliers_detected = True
                st.warning(f"⚠️ Found {outlier_count} outliers in '{col}'")
                
                # Cap outliers instead of removing
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                st.success(f"✅ Capped outliers in '{col}'")
        
        if not outliers_detected:
            st.info("✅ No outliers detected!")
        
        rows_after = len(df_processed)
        st.success(f"Dataset rows: {rows_before} → {rows_after}")
    
    st.markdown("---")
    
    # ========================================================================
    # STEP 4: FEATURE SCALING
    # ========================================================================
    
    st.header("⚖️ Feature Scaling")
    st.markdown("Normalize features using StandardScaler for better model performance")
    
    # Separate features and target
    st.subheader("Select Target Column (for classification)")
    st.info("⚠️ Choose a column with limited classes (e.g., 2-10 unique values for classification)")
    
    # Step 1: Find suitable columns for classification
    suitable_cols = []
    for col in df_processed.columns:
        unique_count = df_processed[col].nunique()
        # Look for columns with 2-10 unique values (good for classification)
        if 2 <= unique_count <= 10:
            suitable_cols.append(col)
    
    # Step 2: Auto-select best column (prioritize common names)
    priority_names = ['target', 'class', 'label', 'num', 'species', 'Species', 'Target', 'Class', 'Label']
    auto_selected = None
    
    for name in priority_names:
        if name in suitable_cols:
            auto_selected = name
            break
    
    # If no priority match, use first suitable column
    if auto_selected is None and len(suitable_cols) > 0:
        auto_selected = suitable_cols[0]
    
    # If still none, allow user to select from any column
    if auto_selected is None:
        st.warning("⚠️ No suitable target columns found. Showing all columns.")
        suitable_cols = df_processed.columns.tolist()
        auto_selected = suitable_cols[0] if len(suitable_cols) > 0 else None
    
    # Show auto-selected or let user override
    col1, col2 = st.columns([2, 1])
    with col1:
        target_col = st.selectbox(
            "Choose the target column (what you want to predict):",
            suitable_cols if len(suitable_cols) > 0 else df_processed.columns.tolist(),
            index=suitable_cols.index(auto_selected) if auto_selected in suitable_cols else 0,
            key="target_selection",
            help="Auto-selected the best option. You can override if needed."
        )
    with col2:
        if st.button("🔄 Auto-Detect", help="Auto-select best target"):
            st.rerun()
    
    # Show auto-detection info
    if auto_selected:
        st.success(f"✅ Auto-detected: **{auto_selected}** as target (2-10 classes)")
    
    # Validate target column
    unique_classes = df_processed[target_col].nunique()
    
    if unique_classes > 100:
        st.error(f"❌ Target column '{target_col}' has {unique_classes} unique values. This is not suitable for classification!")
        st.warning("Please select a column with fewer unique values (2-10 classes recommended)")
        st.stop()
    
    # Prepare features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Remove any rows where target is NaN
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    
    st.success(f"✅ Features shape: {X.shape}")
    st.success(f"✅ Target shape: {y.shape}")
    st.success(f"✅ Target classes: {y.nunique()}")
    st.info(f"Class distribution: {dict(y.value_counts())}")
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Handle any NaN values that might have been created during scaling
    X_scaled = X_scaled.fillna(X_scaled.mean())
    
    st.info("✅ Applied StandardScaler to all features")
    st.write("Scaled features preview:")
    st.dataframe(X_scaled.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # STEP 5: TRAIN-TEST SPLIT
    # ========================================================================
    
    st.header("🔀Train-Test Split")
    st.markdown("Split data into training and testing sets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test set percentage:",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="test_size"
        )
    
    with col2:
        random_state = st.number_input(
            "Random state (for reproducibility):",
            value=42,
            step=1,
            key="random_state"
        )
    
    # Reset indices to ensure alignment
    X_scaled = X_scaled.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Double-check for any NaN values
    if X_scaled.isnull().sum().sum() > 0:
        st.warning("⚠️ Removing remaining NaN values...")
        mask = X_scaled.isnull().any(axis=1) | y.isnull()
        X_scaled = X_scaled[~mask].reset_index(drop=True)
        y = y[~mask].reset_index(drop=True)
    
    # Check if stratification is possible (all classes need at least 2 members)
    stratify_possible = True
    class_counts = y.value_counts()
    if (class_counts < 2).any():
        stratify_possible = False
        st.warning("⚠️ Some classes have only 1 member. Using regular split instead of stratified split.")
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y if (len(np.unique(y)) > 1 and stratify_possible) else None
        )
    except ValueError:
        # Fallback to non-stratified split
        st.warning("⚠️ Could not use stratified split. Using regular split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=int(random_state)
        )
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training set size", len(X_train))
    col2.metric("Test set size", len(X_test))
    col3.metric("Total samples", len(X_train) + len(X_test))
    col4.metric("Test percentage", f"{test_size*100:.1f}%")
    
    st.markdown("---")
    
    # ========================================================================
    # STEP 6: MODEL TRAINING
    # ========================================================================
    
    st.header("🤖Model Training")
    st.markdown("Train KNN and SVM models on the training data")
    
    col1, col2 = st.columns(2)
    
    # KNN Model
    with col1:
        st.subheader("K-Nearest Neighbors (KNN)")
        k_value = st.slider(
            "Select K value:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="k_value"
        )
        
        # Train KNN
        knn_model = KNeighborsClassifier(n_neighbors=int(k_value))
        knn_model.fit(X_train, y_train)
        st.success(f"✅ KNN model trained with k={k_value}")
    
    # SVM Model
    with col2:
        st.subheader("Support Vector Machine (SVM)")
        svm_kernel = st.selectbox(
            "Choose SVM kernel:",
            ["linear", "rbf", "poly"],
            index=0,
            key="svm_kernel"
        )
        
        # Train SVM
        svm_model = SVC(kernel=svm_kernel, random_state=42)
        svm_model.fit(X_train, y_train)
        st.success(f"✅ SVM model trained with {svm_kernel} kernel")
    
    st.markdown("---")
    
    # ========================================================================
    # STEP 7: PREDICTIONS
    # ========================================================================
    
    st.header("🎯Making Predictions")
    st.markdown("Predict on test data using both models")
    
    # Make predictions
    knn_pred = knn_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)
    
    st.success(f"✅ KNN generated {len(knn_pred)} predictions")
    st.success(f"✅ SVM generated {len(svm_pred)} predictions")
    
    # Display sample predictions
    st.subheader("Sample Predictions (first 10 samples)")
    predictions_df = pd.DataFrame({
        'Actual': y_test.values[:10],
        'KNN Prediction': knn_pred[:10],
        'SVM Prediction': svm_pred[:10],
        'KNN Match': (knn_pred[:10] == y_test.values[:10]),
        'SVM Match': (svm_pred[:10] == y_test.values[:10])
    })
    st.dataframe(predictions_df, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # STEP 8: MODEL EVALUATION
    # ========================================================================
    
    st.header("📈 Model Evaluation")
    st.markdown("Compare accuracy and performance of both models")
    
    # Calculate accuracy
    knn_accuracy = accuracy_score(y_test, knn_pred)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    # Display accuracy metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("KNN Accuracy", f"{knn_accuracy*100:.2f}%", delta_color="off")
    
    with col2:
        st.metric("SVM Accuracy", f"{svm_accuracy*100:.2f}%", delta_color="off")
    
    with col3:
        accuracy_diff = abs(knn_accuracy - svm_accuracy)
        better_model = "KNN" if knn_accuracy > svm_accuracy else "SVM" if svm_accuracy > knn_accuracy else "Tie"
        st.metric("Difference", f"{accuracy_diff*100:.2f}%", 
                 delta=f"Winner: {better_model}", delta_color="off")
    
    st.markdown("")
    
    # Confusion Matrices
    st.subheader("Confusion Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**KNN Confusion Matrix**")
        knn_cm = confusion_matrix(y_test, knn_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(knn_cm, display_labels=np.unique(y)).plot(ax=ax, cmap='Blues')
        plt.title(f"KNN Confusion Matrix\nAccuracy: {knn_accuracy*100:.2f}%", 
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("**SVM Confusion Matrix**")
        svm_cm = confusion_matrix(y_test, svm_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(svm_cm, display_labels=np.unique(y)).plot(ax=ax, cmap='Greens')
        plt.title(f"SVM Confusion Matrix\nAccuracy: {svm_accuracy*100:.2f}%", 
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("")
    
    # Classification Reports
    st.subheader("Classification Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**KNN Classification Report**")
        knn_report = classification_report(y_test, knn_pred, output_dict=True)
        knn_report_df = pd.DataFrame(knn_report).transpose()
        st.dataframe(knn_report_df, use_container_width=True)
    
    with col2:
        st.write("**SVM Classification Report**")
        svm_report = classification_report(y_test, svm_pred, output_dict=True)
        svm_report_df = pd.DataFrame(svm_report).transpose()
        st.dataframe(svm_report_df, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # STEP 9: COMPARISON & VISUALIZATION
    # ========================================================================
    
    st.header("📊Model Comparison & Visualization")
    st.markdown("Visual comparison of KNN and SVM performance")
    
    # Accuracy comparison bar chart
    st.subheader("Accuracy Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['KNN', 'SVM']
    accuracies = [knn_accuracy * 100, svm_accuracy * 100]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{accuracy:.2f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Summary table
    st.subheader("Performance Summary")
    
    summary_data = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        'KNN': [
            f"{knn_accuracy*100:.2f}%",
            f"{knn_report['macro avg']['precision']*100:.2f}%",
            f"{knn_report['macro avg']['recall']*100:.2f}%",
            f"{knn_report['macro avg']['f1-score']*100:.2f}%"
        ],
        'SVM': [
            f"{svm_accuracy*100:.2f}%",
            f"{svm_report['macro avg']['precision']*100:.2f}%",
            f"{svm_report['macro avg']['recall']*100:.2f}%",
            f"{svm_report['macro avg']['f1-score']*100:.2f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Winner announcement
    
    # ========================================================================
    # FOOTER & DOWNLOAD
    # ========================================================================
    
    st.header("💾 Download Results")
    st.markdown("Export your analysis results")
    
    # Create results summary
    results_text = f"""
# ML Pipeline Analysis Results

## Dataset Information
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Target Column: {target_col}
- Test Size: {test_size*100:.1f}%

## Model Performance
- KNN (k={k_value}) Accuracy: {knn_accuracy*100:.2f}%
- SVM ({svm_kernel} kernel) Accuracy: {svm_accuracy*100:.2f}%
- Best Model: {'KNN' if knn_accuracy > svm_accuracy else 'SVM' if svm_accuracy > knn_accuracy else 'Tie'}

## Model Details
### KNN Classification Report
{classification_report(y_test, knn_pred)}

### SVM Classification Report
{classification_report(y_test, svm_pred)}
"""
    
    st.download_button(
        label="📄 Download Results as Text",
        data=results_text,
        file_name="ml_analysis_results.txt",
        mime="text/plain"
    )
    
    st.markdown("---")
