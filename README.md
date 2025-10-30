# 🏠 Advanced House Price Prediction (Lightweight Model)

This repository contains a **lightweight machine learning model** built to predict house prices using only the **most important features** from the original dataset.  
It is designed to be **user-friendly**, requiring fewer inputs while maintaining strong predictive performance.

> 💡 For the **full model built on the entire dataset**, please refer to the complete version here:  
> 🔗https://github.com/Mounika-17/houseprice-prediction-ml  

---

## 🚀 Project Overview

The goal of this project is to predict house prices using simplified yet effective features from the dataset.  
A streamlined version of the **Advanced House Price Prediction** pipeline was implemented to create a faster, smaller, and deployment-ready model.

This project demonstrates:
- End-to-end ML pipeline (data ingestion → transformation → model training → prediction)
- Modular project structure
- Logging and exception handling
- Deployment using **FastAPI + Render**

---

## 🧩 Model Performance (Training Logs)

Below are the results from the model training pipeline (`logs/ml_pipeline_logger`):

| Model | CV RMSE | Test RMSE (Original Scale) | R² Score |
|--------|----------|----------------------------|-----------|
| Linear Regression | 0.1365 ± 0.0139 | 29,334 | 0.8878 |
| Ridge Regression | 0.1352 ± 0.0159 | 29,944 | 0.8831 |
| Lasso | 0.1363 ± 0.0166 | 30,192 | 0.8812 |
| ElasticNet | 0.1356 ± 0.0150 | 29,515 | 0.8864 |
| Bayesian Ridge | 0.1352 ± 0.0155 | 29,698 | 0.8850 |
| Huber Regressor | 0.1369 ± 0.0146 | 29,194 | 0.8889 |
| SVR (**Best Model**) | **0.1295 ± 0.0119** | **25,341** | **0.9163** |
| Random Forest | 0.1503 ± 0.0142 | 28,586 | 0.8935 |
| Gradient Boosting | 0.1351 ± 0.0158 | 25,743 | 0.9136 |
| AdaBoost | 0.1824 ± 0.0083 | 38,516 | 0.8066 |
| XGBoost | 0.1340 ± 0.0145 | 25,091 | 0.9179 |

🏆 **Best Performing Model:** SVR (`kernel='rbf'`, `C=1`, `gamma='auto'`)  
📦 Saved Model: `artifacts/model.pkl`

---

## 📁 Project Structure
<pre> ```bash HOUSEPRICE-PREDICTION-V2/ │ ├── artifacts/ │ ├── model.pkl │ ├── train.csv │ └── test.csv │ ├── logs/ │ └── (log files generated during data processing & training) │ ├── notebook/ │ ├── data/ │ ├── create_lightweight_data.py │ └── eda_model_training.ipynb │ ├── src/ │ ├── components/ │ ├── pipeline/ │ ├── config.py │ ├── exception.py │ ├── logger.py │ └── utils.py │ ├── app.py ├── requirements.txt ├── setup.py ├── runtime.txt ├── Procfile ├── README.md └── .gitignore ``` </pre>

## ⚙️ Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI
- **Deployment:** Render
- **Libraries:**  
  `numpy`, `pandas`, `scikit-learn`, `pydantic`, `uvicorn`, `pickle`, `joblib`

---

## 🔍 How It Works

1. **Data Ingestion:** Loads and splits dataset into train/test sets.  
2. **Data Transformation:** Creates preprocessing pipeline for numerical and categorical features.  
3. **Model Training:** Trains and evaluates multiple models using GridSearchCV.  
4. **Model Selection:** Saves the best-performing model and preprocessor.   
5. **Prediction Pipeline:** Uses `PredictPipeline` and `CustomData` classes to generate predictions.  

---

## 🌐 Deployment on Render (FastAPI)  

The model is deployed as a REST API using **FastAPI**.  
### 🔗 Live API:
https://houseprice-prediction-v2.onrender.com/  

### 📘 API Docs:  
https://houseprice-prediction-v2.onrender.com/docs  

## ✨ Key Highlights  

✅ Lightweight version → Predict quickly with fewer inputs  
✅ Modular and production-ready ML pipeline  
✅ Deployed using FastAPI + Render  
✅ Integrated logging and error handling  
✅ Fully reproducible results  

## 👩‍💻 Author  
Mounika Maradana  
📧 https://www.linkedin.com/in/mounikamaradana/  
🌐 https://github.com/Mounika-17  

## 🏁 License  

This project is open-source and available under the MIT License.  

## 🧱 Acknowledgements  

Kaggle Advanced House Price Prediction Dataset  

FastAPI Documentation  

Render Deployment Platform  

