# Pickups streamlit app

## 📝 About
This project analyzes taxi rides in New York City and attempts to identify patterns to optimize driver distribution.

---

## ✨ Features
- Context : Provide statistics and metadata on the dataset.
- Dataset Exploration : Analysing trends on the dataset.
- Clusters Analysis : Visualization of clusters generated with KNN and DBScan models.

---

## 🔧 Requirements
- Python 3.12+
- Python libraries specified in ```requirements.txt```

---

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/Olivier-52/st-pickups-app.git
cd st-pickups-app
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Launch the application:
```
streamlit run app.py
```

---

## 📂 Project Structure
```
st-newsletter-app/
├── .streamlit/
│   └── config.toml
├── data/
│   └── new_york_taxi_cluster.csv
├── images/
│   └── new-york-pickups.png 
├── .gitattributes
├── .gitignore 
├── app.py
├── LICENSE
├── README.md
└── requirements.txt  
```

---

## 📜 Licence
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---