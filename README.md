# 🏥 IoT-Based Health Monitoring System using Machine Learning

An intelligent health monitoring system that predicts patient conditions (Low, Medium, High Risk) based on vital signs collected from IoT sensors. The system analyzes ECG data, temperature, blood pressure, and patient ID to provide real-time health risk assessment.

## 📋 Project Overview

This project implements a complete machine learning pipeline for health monitoring, featuring:
- IoT sensor data collection and analysis
- Multiple ML algorithm comparison
- Interactive web-based prediction interface using Streamlit
- Comprehensive data visualization and analysis

## 🗂️ Project Structure

```
health-monitoring/
├── iot_dataset.csv                    # IoT sensor dataset (152 records)
├── IotFile.ipynb                      # Jupyter notebook with ML pipeline
├── patient_condition_streamlit.py     # Streamlit web application
└── README.md                          # Project documentation
```

## 📊 Dataset Features

The dataset contains the following features:
- **Sl.No**: Serial number
- **Patient ID**: Unique patient identifier
- **Temperature Data**: Body temperature readings (°C)
- **ECG Data**: Electrocardiogram readings
- **Pressure Data**: Blood pressure measurements
- **Target**: Patient condition (0=Low Risk, 1=Moderate Risk, 2=High Risk)

## 🔬 Project Modules

### 1. Data Collection
IoT sensors collect real-time patient vital signs including:
- ECG readings
- Body temperature
- Blood pressure
- Patient identification

### 2. Data Preprocessing
- Data loading and inspection
- Handling missing values
- Data shape and type analysis
- Target variable distribution analysis

### 3. Data Visualization
Comprehensive exploratory data analysis (EDA) including:
- **Bar Charts**: Patient ID vs Temperature, ECG, Pressure, and Target
- **Histograms**: Distribution analysis of all features
- **Count Plots**: Frequency analysis using Seaborn
- **KDE Plots**: Kernel density estimation for continuous variables
- **Correlation Heatmap**: Feature correlation analysis

### 4. Model Implementation
Two machine learning algorithms implemented and compared:

#### Naive Bayes Algorithm
- Algorithm: MultinomialNB
- Train-test split: 80-20
- Evaluation metrics: Accuracy, Classification Report, Confusion Matrix

#### Decision Tree Algorithm
- Algorithm: DecisionTreeClassifier
- Train-test split: 80-20
- Evaluation metrics: Accuracy, Classification Report, Confusion Matrix
- **Selected as final model** for deployment

### 5. Classification and Prediction
- Model performance comparison
- Classification reports (Precision, Recall, F1-Score)
- Confusion matrix visualization
- Real-time prediction on new patient data
- Risk categorization: Low (0), Moderate (1), High (2)

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Running the Jupyter Notebook
```bash
jupyter notebook IotFile.ipynb
```

### Running the Streamlit Web Application
```bash
streamlit run patient_condition_streamlit.py
```

## 💻 Web Application Features

The Streamlit application provides:
- **Interactive UI**: User-friendly interface for inputting patient data
- **Real-time Prediction**: Instant health risk assessment
- **Visual Feedback**: Color-coded risk levels (Green/Yellow/Red)
- **Input Validation**: Ensures data quality and safety
- **Responsive Design**: Works on desktop and mobile devices

### Input Parameters:
- Patient ID
- Temperature (°C) - Range: 35.0 to 42.0
- ECG Reading
- Blood Pressure

### Output:
- 🟢 **LOW RISK**: Patient's vital signs are within normal ranges
- 🟡 **MODERATE RISK**: Patient requires regular monitoring
- 🔴 **HIGH RISK**: Immediate medical attention recommended

## 📈 Model Performance

The Decision Tree algorithm was selected as the final model based on its performance metrics including accuracy, precision, recall, and F1-score. Detailed performance metrics are available in the Jupyter notebook.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and metrics
- **Streamlit**: Web application framework
- **Jupyter Notebook**: Interactive development environment

## 👨‍💻 Author

**Allan John**

## 📝 License

This project is open-source and available for educational and research purposes.

---

*Last Updated: October 2025*
