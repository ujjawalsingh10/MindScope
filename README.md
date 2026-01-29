

# 🧠 MindScope-ML — Mental Health Prediction System

MindScope-ML is an end-to-end Machine Learning system for predicting depression risk using demographic, academic, and lifestyle data.
The project originated from a Kaggle competition and was later extended into a full production-grade ML pipeline with automated deployment.

It demonstrates how a data science solution can be transformed into a scalable, real-world application.

---

## 🚀 Live Demo

🔗 **Deployed Application:**
[http://35.170.75.116:5000](http://35.170.75.116:5000)

> Users can enter their details and receive real-time mental health risk predictions.

---

## 📌 Problem Statement

Mental health disorders, especially depression, are becoming increasingly common among students and working professionals.
Early detection can help in timely intervention and support.

This project aims to:

* Analyze behavioral and demographic data
* Identify patterns related to depression risk
* Provide real-time predictions using Machine Learning

---

## 📊 Dataset

* Source: Kaggle Mental Health Dataset
* Contains information about:

  * Age, Gender, City
  * Academic/Work Pressure
  * Satisfaction Levels
  * Sleep Duration
  * Dietary Habits
  * Financial Stress
  * Family History of Mental Illness
  * Depression Label (Target)

🔗 Kaggle Notebook:
[https://www.kaggle.com/code/ujjawalsingh10/mental-health-data-gradientboosting-94-03](https://www.kaggle.com/code/ujjawalsingh10/mental-health-data-gradientboosting-94-03)

---

## 🏗️ Project Architecture

```
User → Web Interface → FastAPI Backend
     → Prediction Pipeline
     → Data Transformer
     → Trained ML Model
     → Result

CI/CD → GitHub Actions → Docker → ECR → EC2
```

---

## ⚙️ System Workflow

1. User submits data via web interface
2. Data is converted into DataFrame format
3. Prediction transformer applies same preprocessing as training
4. Model generates prediction
5. Result is returned to the UI
6. CI/CD pipeline redeploys on every push

---

## 🧩 Key Features

### ✔ Data Preprocessing

* Missing value handling
* Feature merging (Pressure, Satisfaction)
* Categorical cleaning
* Domain-based imputation
* Noise reduction

### ✔ Feature Engineering

* Academic + Work Pressure → Pressure
* Study + Job Satisfaction → Satisfaction
* CGPA handling based on profession
* City/Degree grouping
* Sleep categorization

### ✔ Machine Learning

* Gradient Boosting Classifier
* Hyperparameter tuning
* Performance evaluation
* Robust inference pipeline

### ✔ Web Application

* FastAPI backend
* Jinja2 frontend
* Dynamic forms
* Input validation
* Real-time prediction

### ✔ Deployment & MLOps

* Docker containerization
* AWS EC2 hosting
* Amazon ECR image registry
* GitHub Actions CI/CD
* Automated redeployment

---

## 🛠️ Tech Stack

| Category         | Tools                       |
| ---------------- | --------------------------- |
| Programming      | Python                      |
| ML Libraries     | Scikit-learn, NumPy, Pandas |
| Visualization    | Matplotlib, Seaborn, Plotly |
| Backend          | FastAPI                     |
| Frontend         | HTML, CSS, Jinja2           |
| Containerization | Docker                      |
| Cloud            | AWS EC2, ECR, S3            |
| CI/CD            | GitHub Actions              |
| Database         | MongoDB                     |
| Utilities        | PyYAML, dotenv, boto3       |

---

## 📁 Project Structure

```
mindscope/
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── entity/
│   ├── utils/
│   ├── logger/
│   └── exception/
│
├── templates/
├── static/
├── app.py
├── Dockerfile
├── requirements.txt
├── .github/workflows/
├── README.md
└── .env
```

---

## 🔄 Data Transformation Pipeline

Training and prediction use identical preprocessing logic:

* Drop unused columns
* Combine satisfaction and pressure
* Handle CGPA based on user type
* Encode categorical variables
* Fill missing numeric values
* Apply One-Hot Encoding

This ensures consistency between training and inference.

---

## 📦 Model Pipeline

1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation
6. Model Pushing
7. Prediction Pipeline

Each step produces artifacts for traceability.

---

## 🔁 CI/CD Pipeline

The project uses GitHub Actions for automation:

### On Every Push:

1. Build Docker Image
2. Push to Amazon ECR
3. Pull Image on EC2
4. Restart Container
5. Deploy Updated App

This enables Continuous Integration and Continuous Deployment.

---

## ▶️ Running Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/mindscope.git
cd mindscope
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

### 5️⃣ Open Browser

```
http://localhost:5000
```

---

## 🐳 Run with Docker

```bash
docker build -t mindscope .
docker run -p 5000:5000 mindscope
```

---

## 📈 Performance

* Accuracy: ~94%

Optimized for real-world robustness.

---

## 🔮 Future Enhancements

* SHAP Explainability
* Model Monitoring
* Drift Detection
* Model Registry
* A/B Testing
* Authentication
* Dashboard Analytics

---

## 🎯 Learning Outcomes

This project helped me gain hands-on experience in:

* End-to-End ML System Design
* Production ML Pipelines
* Cloud Deployment
* DevOps Integration
* MLOps Practices
* Scalable Web Applications

---

## 👨‍💻 Author

**Ujjawal Singh**

* Kaggle: [https://www.kaggle.com/ujjawalsingh10](https://www.kaggle.com/ujjawalsingh10)
* Leetcode: [https://leetcode.com/u/ujjawalsingh10/](https://leetcode.com/u/ujjawalsingh10/)

---

## 📜 License

This project is licensed under the MIT License.

---
