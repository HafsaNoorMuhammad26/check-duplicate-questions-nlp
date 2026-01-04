# Check Duplicate Questions – NLP

## 📌 Project Description

This project focuses on **Duplicate Question Detection** using **Natural Language Processing (NLP)** techniques.  
The goal is to determine whether two questions are semantically similar (duplicates) or not. This type of system is commonly used in platforms like Quora, Stack Overflow, and discussion forums to reduce repeated questions and improve content organization.

The project includes:
- Text preprocessing
- Feature extraction
- Model training and evaluation
- A demo application to test duplicate question detection

---

## 👩‍🎓 Project Members

| Name | Roll Number |
|-----|-------------|
| Hafsa Noor Muhammad | **(22SP-051-SE)** |
| Huraira Riaz | **(22SP-036-SE)** |
| M. Zain | **(22SP-012-SE)** |


---

## 📁 Project Structure

```

check-duplicate-questions-nlp/
│
├── models/                  # Trained and saved models
├── nltk_data/               # NLTK resources
├── app.py                   # Application for demo/testing
├── train_model.ipynb        # Model training notebook
├── requirements.txt         # Required Python libraries
└── README.md                # Project documentation

````

---

````
## 🛠️ Installation & Setup (Reproducibility Steps)

Follow these steps to **reproduce the demo successfully**:

### Step 1: Clone the Repository

```bash
git clone https://github.com/HafsaNoorMuhammad26/check-duplicate-questions-nlp.git
cd check-duplicate-questions-nlp
````

---

```
### Step 2: Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / macOS**

```bash
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Train the Model

Open the notebook:

```bash
jupyter notebook train_model.ipynb
```

Run all cells to:

* Load and preprocess the dataset
* Train the duplicate question detection model
* Save the trained model inside the `models/` folder

---

### Step 5: Run the Demo Application

```bash
python app.py
```

Open your browser and go to:

```
http://localhost:8501
```

(Port may vary depending on framework used)

---

## 🧪 Demo Usage

1. Enter **Question 1**
2. Enter **Question 2**
3. Click **Predict**
4. Output:

   * **Duplicate** ✅
   * **Not Duplicate** ❌

---

## 📊 Dataset

The model uses a **question-pair dataset** (e.g., Quora Question Pairs or a custom dataset) containing:

* Question 1
* Question 2
* Label (Duplicate / Not Duplicate)

---

## 📦 Technologies Used

* Python
* NLTK
* Scikit-learn
* Pandas
* NumPy
* Streamlit / Flask (for demo)

---
