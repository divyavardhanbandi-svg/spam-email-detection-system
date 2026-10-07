# Spam Email Detection System

Simple ML project: TF-IDF + Naive Bayes / Logistic Regression.

## Setup
pip install -r requirements.txt

## Run
python spam_detector.py                       # demo with built-in sample data
python spam_detector.py --data spam.csv       # train on your own CSV (columns: label,text)
python spam_detector.py --predict "Win a free prize, click now!"
