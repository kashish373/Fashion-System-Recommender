👗✨ Fashion Recommender System — AI-Powered Outfit Recommendation Web App

A smart, AI-based Fashion Recommender System that analyzes clothes, detects colors, understands patterns, and recommends matching outfits using machine learning.
This project combines Machine Learning, Computer Vision, Flask Backend, and a Full Frontend UI to simulate a real-world fashion recommendation platform.

⸻

🚀 Project Overview

Choosing what to wear can be confusing — this application solves that problem using AI.

Users can:
	•	Upload wardrobe images
	•	Manage their personal fashion inventory
	•	Get ML-based outfit recommendations
	•	Analyze clothes using color + feature extraction
	•	Interact with a sleek, user-friendly dashboard

This project demonstrates the full workflow of an AI system:
data → preprocessing → model training → backend → frontend → deployment.

⸻

📌 Key Features

🎨 1. Image Analysis
	•	Detects dominant color
	•	Extracts clothing patterns
	•	Identifies item type

🤖 2. ML-Based Outfit Recommender
	•	Uses a trained ML model (model.joblib)
	•	Suggests matching items from the wardrobe
	•	Displays recommendations with similarity scores

🧺 3. Wardrobe Manager
	•	Users can upload, view, and store their clothes
	•	Saved items appear in an interactive dashboard

🔐 4. Authentication System
	•	Signup & Login functionality
	•	User-specific wardrobe and preferences

🖥️ 5. Full Web App (Flask)
	•	HTML templates
	•	CSS styling
	•	JavaScript interactivity
	•	Organized templates/ and static/ folders

⸻

🧠 Machine Learning & Logic

ML Components:
	•	Color Extraction using image processing
	•	Outfit Matching Model trained on CSV datasets
	•	Similarity-based recommendation using custom logic
	•	Full model stored as model.joblib

Data Files Used:
	•	filtered_data_updated.csv
	•	sample_dataset.csv

These datasets help the system identify:
	•	Colors
	•	Clothing types
	•	Style pairings

⸻

🛠️ Tech Stack

🎯 Backend
	•	Python
	•	Flask
	•	SQLAlchemy
	•	Joblib (model loading)
	•	Pillow / OpenCV (image processing)

🎨 Frontend
	•	HTML5
	•	CSS3
	•	JavaScript
	•	Responsive UI

🧠 Machine Learning
	•	NumPy
	•	Pandas
	•	Scikit-learn
	•	Custom outfit matching logic

⸻

📂 Project Structure
├── app.py
├── extensions.py
├── models.py
├── recommender.py
├── ml_model.py
├── outfit_matcher.py
├── image_analyzer.py
├── init_db.py
│
├── model.joblib
├── sample_dataset.csv
├── filtered_data_updated.csv
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── wardrobe.html
│   ├── recommendations.html
│   ├── ml_train.html
│   └── image_analysis.html
│
├── static/
│   ├── styles.css
│   └── main.js
│
├── PROJECT_ANALYSIS.md
├── README.md
└── requirements.txt

## 👥 Contributors
- [@Ananyab1816](https://github.com/Ananyab1816)
- [@aaditrichandok](https://github.com/aaditrichandok)


