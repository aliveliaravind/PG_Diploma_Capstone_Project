# Sentiment-Based Product Recommendation System

This project served as the Capstone Project for the Post Graduate Diploma in Machine Learning &amp; AI at IIIT Bangalore, demonstrating end-to-end ML model building, evaluation, and deployment.

## Objective
To improve e-commerce product recommendations by combining Collaborative Filtering with Natural Language Processing (NLP) — delivering personalized Top 5 product suggestions based on both user behavior and sentiment from product reviews.

## What We Built
1. **Data Pipeline** — Implemented text preprocessing, lemmatization, and TF-IDF vectorization on user reviews.
2. **Model Selection** — Trained Logistic Regression, Naive Bayes, Random Forest, and XGBoost with hyperparameter tuning.
3. **Sentiment Analysis** — Selected Logistic Regression with oversampled data as the best model, achieving 54% Specificity, 95% Sensitivity, and 0.77 F1-score for positive sentiment detection.
4. **Hybrid Recommendation** — Selected User-Based Collaborative Filtering (RMSE: 2.35) to generate the initial Top 20 products, which are then ranked by the sentiment model to deliver the final Top 5 products.
5. **UI Development** — Created an intuitive frontend interface to present personalized ML recommendations directly to end users.
6. **Deployment** — Packaged the ML artifacts (models, vectorizers, matrices) and deployed the fully interactive Flask app on Render.

## Live Demo
https://pg-diploma-capstone-project.onrender.com/
