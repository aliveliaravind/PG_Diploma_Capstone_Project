# Imports
import os
import pickle
import warnings

import pandas as pd
from flask import Flask, request, render_template

warnings.filterwarnings('ignore')

# Load Models & Data (once at startup)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    user_final_rating = pd.read_pickle(os.path.join(MODELS_DIR, "user_final_rating.pkl"))
    df = pd.read_csv(os.path.join(MODELS_DIR, "df.csv"))

    with open(os.path.join(MODELS_DIR, "word_vectorizer.pkl"), "rb") as f:
        word_vectorizer = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "logit_model.pkl"), "rb") as f:
        logit = pickle.load(f)

    username_list = sorted(user_final_rating.index.astype(str).tolist())
    print("All models and data loaded successfully.")

except Exception as e:
    print(f"Startup error while loading models/data: {e}")
    raise

# Recommendation Function
def recommend(user_input):
    """
    Returns:
      top20List: list of 20 product names from collaborative filtering
      top5List: list of top 5(product_name, sentiment_score%) tuples
    """
    user_input = user_input.strip()

    # Step 1 — Validate user exists
    if user_input not in user_final_rating.index:
        return None, None

    # Step 2 — Get top 20 products by predicted CF rating
    top20 = user_final_rating.loc[user_input].sort_values(ascending=False).head(20)

    # Small safety check
    if top20.empty:
        return [], []  # No products to recommend
    
    top20List = top20.index.tolist()

    # Step 3 & 4 — Score each product by positive sentiment %
    sentiment_scores = {}
    for prod_name in top20.index.tolist():

        # Fetch all reviews for this product
        product_reviews = (
            df[df["prod_name"] == prod_name]["Review"]
            .dropna()
            .astype(str)
            .tolist()
        )

        # If no reviews exist, assign 0
        if len(product_reviews) == 0:
            sentiment_scores[prod_name] = 0.0
            continue

        # TF-IDF transform → predict sentiment → mean = positive %
        features = word_vectorizer.transform(product_reviews)
        preds    = logit.predict(features)
        sentiment_scores[prod_name] = round(float(preds.mean() * 100), 1)

    # Step 5 — Sort by sentiment %, return top 5 as (name, score) tuples
    top5 = (
        pd.Series(sentiment_scores)
        .sort_values(ascending=False)
        .head(5)
    )

    top5List = [(name, score) for name, score in top5.items()]

    return top20List, top5List


# Flask App
app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html", usernameList=username_list)


# Recommend Route
@app.route("/recommend", methods=["POST"])
def predict():
    username      = str(request.form.get("username", "")).strip()

    # Validate — empty input
    if not username:
        return render_template(
            "index.html",
            usernameList=username_list,
            error="Please enter a username."
        )

    # Run recommendation
    top20List, result = recommend(username)

    # Validate — user not found
    if top20List is None:
        return render_template(
            "index.html",
            usernameList=username_list,
            error=f"User '{username}' not found. Please select a username from the list below."
        )
    
    # Validate — no recommendations generated
    if not result:
        return render_template(
            "index.html",
            usernameList=username_list,
            error="No recommendations available for this user."
        )

    # Success — render results page
    return render_template(
        "recommendations.html",
        username=username,
        top20List=top20List,   # list of product names
        productList=result        # list of (name, score) tuples
    )

# Run Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"** Starting Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)