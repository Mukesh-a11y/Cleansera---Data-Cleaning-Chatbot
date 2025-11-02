[README (2).md](https://github.com/user-attachments/files/23291901/README.2.md)
Data Cleaning Chatbot
Project Overview

Data Cleaning Chatbot is an intelligent, Flask-based web application designed to automate and simplify the process of cleaning and preprocessing datasets.
It acts as an AI-powered data assistant, allowing users to upload raw CSV files, explore dataset columns, clean data interactively, and download the refined version — all through an intuitive chat-style interface.

This project brings together data engineering, AI, and UI automation principles to streamline tedious data preparation workflows in a user-friendly manner.

✨ Features

File Upload Support – Upload raw CSV datasets directly from your computer.

Data Column Overview – Automatically lists column names and basic info.

Data Cleaning Options – Detect and handle missing values, duplicates, or invalid data.

Undo Action – Revert the most recent cleaning step.

Download Cleaned File – Export cleaned datasets as CSV files.

AI Chat Interface – Interact with the assistant to request cleaning tasks.

Gemini AI Integration – Backend logic powered by Gemini for contextual data understanding.

Smart Logging – Track every cleaning action for transparency.

Responsive UI – Built with HTML, CSS, and JavaScript (no React) for fast and lightweight interaction.

Tech Stack
Component	Technology
Backend Framework	Flask (Python)
Frontend	HTML, CSS, JavaScript
AI Engine	Gemini API (for natural-language understanding)
File Handling	Pandas
Data Storage	Local temporary storage (uploads/, processed/)
Runtime Environment	Python 3.9+
Deployment Ready	Docker-compatible & Render-ready

Project Structure
Data-Cleaning-Chatbot/
│
├── app.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── img/
│       └── cleansera-logo.png
│
├── uploads/                # Temporarily stores uploaded files
├── processed/              # Stores cleaned output files
└── README.md

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/your-username/Data-Cleaning-Chatbot.git
cd Data-Cleaning-Chatbot

2️⃣ Create and Activate a Virtual Environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

3️⃣ Install Required Dependencies
pip install -r requirements.txt

4️⃣ Run the Flask Application
python app.py

5️⃣ Open the App

Visit:

http://127.0.0.1:5000/
