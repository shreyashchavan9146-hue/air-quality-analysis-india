# 🌍 Air Quality Analysis of India

### Transforming Environmental Data into Actionable Insights with Python & Machine Learning
---

## 🚀 Live Project & Repository

| Resource                | Link                                                                                                                                                                                                                                                              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🌐 Live Dashboard       | [Explore Interactive Dashb](YOUR_STREAMLIT_APP_URL)[https://shreyashchavan9146-hue-air-quality-analysis-dashboardapp-riycpj.streamlit.app/](https://shreyashchavan9146-hue-air-quality-analysis-dashboardapp-riycpj.streamlit.app/)[oard](YOUR_STREAMLIT_APP_URL) |
| 💻 GitHub Repository    | [View Source Code](https://github.com/shreyashchavan9146-hue/air-quality-analysis-india)                                                                                                                                                                          |
| 👨‍💻 Developer Profile    | [Shreyash Chavan](https://github.com/shreyashchavan9146-hue)                                                                                                                                                                                                      |
---

## 📌 Project Overview

Air Quality Analysis of India is an end-to-end Data Analytics and Machine Learning project that explores air pollution patterns across India.

The project uses Python-based data processing, statistical analysis, data visualization, and machine learning algorithms to identify pollution patterns, compare pollutant concentrations, and generate estimated AQI predictions.

## 🎯 Project Objectives

- Analyze air pollution data across different Indian states.
- Identify patterns in major air pollutants.
- Compare state-wise estimated AQI levels.
- Understand relationships between pollutants.
- Apply machine learning for AQI estimation.
- Build an interactive dashboard for data exploration.

## ✨ Key Features

### 📊 1. Data Analysis

- Data loading and preprocessing.
- Missing-value treatment.
- Duplicate removal.
- Statistical summaries.

### 🌫️ 2. Pollutant Analysis

- SO₂ (Sulphur Dioxide)
- NO₂ (Nitrogen Dioxide)
- RSPM (Respirable Suspended Particulate Matter)
- SPM (Suspended Particulate Matter)

### 📍 3. State-wise Analysis

- State-level AQI comparison.
- Top 10 states by estimated AQI.
- Lowest 10 states by estimated AQI.

### 📈 4. Exploratory Data Analysis

- Histograms and boxplots.
- Correlation heatmaps.
- Pollution distribution.
- Trend analysis where date data is available.

### 🤖 5. Machine Learning

- Linear Regression
- Ridge Regression
- Lasso Regression
- Model comparison
- Model evaluation
- AQI prediction

### 🌐 6. Interactive Streamlit Dashboard

- National overview and KPIs.
- Interactive pollutant charts.
- State-wise comparisons.
- AQI analytics.
- Machine learning prediction.
- Dataset exploration and CSV download.

## 🛠️ Technology Stack & Useful Links

| Technology   | Purpose                   | Official Resource                           |
| ------------ | ------------------------- | ------------------------------------------- |
| Python       | Programming               | [Python](https://www.python.org/)           |
| Pandas       | Data manipulation         | [Pandas](https://pandas.pydata.org/)        |
| NumPy        | Numerical computing       | [NumPy](https://numpy.org/)                 |
| Matplotlib   | Data visualization        | [Matplotlib](https://matplotlib.org/)       |
| Seaborn      | Statistical visualization | [Seaborn](https://seaborn.pydata.org/)      |
| Plotly       | Interactive charts        | [Plotly](https://plotly.com/python/)        |
| Scikit-learn | Machine learning          | [Scikit-learn](https://scikit-learn.org/)   |
| Statsmodels  | Statistical analysis      | [Statsmodels](https://www.statsmodels.org/) |
| SciPy        | Scientific computing      | [SciPy](https://scipy.org/)                 |
| Joblib       | Model serialization       | [Joblib](https://joblib.readthedocs.io/)    |
| Streamlit    | Dashboard development     | [Streamlit](https://docs.streamlit.io/)     |
| Git          | Version control           | [Git](https://git-scm.com/)                 |
| GitHub       | Project hosting           | [GitHub](https://github.com/)               |

## 📂 Project Structure

```text
Air-Quality-Analysis-of-India/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── aqi_calculation.py
│   ├── state_analysis.py
│   ├── regression_analysis.py
│   ├── model_evaluation.py
│   ├── vif_analysis.py
│   ├── interaction_analysis.py
│   ├── regularization.py
│   ├── model_comparison.py
│   └── model_prediction.py
│
├── dashboard/
│   └── app.py
│
├── outputs/
│   ├── graphs/
│   ├── models/
│   └── reports/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/shreyashchavan9146-hue/air-quality-analysis-india.git
```

**Step 2: Navigate to the project**

```bash
cd air-quality-analysis-india
```

**Step 3: Create a virtual environment**

```bash
python -m venv venv
```

**Step 4: Activate the environment (Windows)**

```bash
venv\Scripts\activate
```

**Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 6: Run the Streamlit dashboard**

```bash
python -m streamlit run dashboard/app.py
```

## 📊 Machine Learning Models

This project compares three regression algorithms:

- Linear Regression
- Ridge Regression
- Lasso Regression

Model performance is evaluated using regression metrics, including R², MAE, and RMSE where applicable.

## 🔮 Future Enhancements

- Integrate real-time air quality data through APIs.
- Add interactive geographical pollution maps.
- Include additional machine learning algorithms.
- Improve prediction accuracy through feature engineering.
- Add time-series forecasting.
- Compare estimates against official CPCB AQI measurements.

## 👨‍💻 About the Developer

**Shreyash Chavan**

Aspiring Data Analyst | Python | SQL | Data Analytics | Machine Learning

- GitHub: [@shreyashchavan9146-hue](https://github.com/shreyashchavan9146-hue)
- Project Repository: [Air Quality Analysis of India](https://github.com/shreyashchavan9146-hue/air-quality-analysis-india)

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!

**Disclaimer:** The AQI calculation used in this project is a project-specific estimate and is not an official CPCB AQI calculation. Predictions are intended for educational and analytical purposes only.
