# 🩸 Blood Group Detection using Fingerprint

## 📌 Overview
This is an **academic machine learning project** that predicts a person’s **blood group** using a **fingerprint image**.  
The system uses a **Convolutional Neural Network (CNN)** for prediction and a **Flask web application** to provide a simple user interface for uploading fingerprint images and viewing results.

⚠️ This project is **experimental** and intended **only for academic and research purposes**.

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV, PIL  
- Flask  
- HTML, CSS  

---

## 📂 Project Structure


```text
blood_group_project/
├── flask_app/
│   ├── app.py
│   ├── model.py
│   ├── model/
│   │   └── final_cnn.h5        (not included on GitHub)
│   ├── templates/
│   │   └── index.html
│   ├── uploads/               (created automatically at runtime)
│   ├── dataset/               (user-added, ignored by git)
│   └── cnn_training.ipynb
├── README.md
├── .gitignore


--

## Dataset
The dataset used in this project is sourced from Kaggle and is not included
in this repository due to licensing restrictions.

To use this project:
1. Download the dataset from Kaggle
2. Extract it into a folder named `dataset/` inside `flask_app/`
3. Ensure the folder structure matches the training notebook

---

## 🤖 Trained Model
The trained CNN model file (`final_cnn.h5`) is **not included in this repository** because it exceeds GitHub’s **100 MB file size limit**.

### How to get the model
- Train the model using `ml/cnn_training.ipynb`, **or**
- Place a trained model manually at:
flask_app/model/final_cnn.h5

--

▶️ How to Run (One-Line Steps)

- Clone the project: git clone https://github.com/<your-username>/blood-group-project.git && cd blood-group-project
- Install dependencies: pip install -r requirements.txt
- Add trained model at flask_app/model/final_cnn.h5
- Start the app: cd flask_app && py app.py
- Open browser and visit: http://127.0.0.1:5000

## Disclaimer
This project is experimental and not medically certified.

## Contributors
- Aniket Shelake
- Mansi Pisal
- Rupesh Salunkhe
- Daivata Potekar
