import pickle
from flask import Flask, render_template, request
import spacy as sp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import RandomForestClassifier

global w1, data
rec_website = 0

class WebsiteRecommender:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = 'artifacts/websites.pkl'  # Provide the default path here
        with open(data_path, 'rb') as file:
            self.websites_df = pickle.load(file)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        #self.website_matrix = self.vectorizer.fit_transform(self.websites_df['cleaned_text'].fillna(''))
        self.website_matrix = self.vectorizer.fit_transform(self.websites_df['title'].fillna(''))

    def recommend_websites(self, input_website_title, num_recommendations=30):
        input_vector = self.vectorizer.transform([input_website_title])

        cosine_similarities = linear_kernel(input_vector, self.website_matrix).flatten()
        related_websites_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]

        #recommendations = self.websites_df.iloc[related_websites_indices][['cleaned_text', 'title', 'url']]
        recommendations = self.websites_df.iloc[related_websites_indices][['title', 'url']]
        return recommendations


def website_recommendation_model(input_title):
    recommender = WebsiteRecommender()

    input_title = str(input_title).lower()
    recommendations = recommender.recommend_websites(input_title)

    print(f"Recommended Websites based on '{input_title}':")
    print(recommendations)
    return recommendations

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def search():
    text = request.form['searchwebsite']
    w1 = text

    # Example Usage:
    rec_website = website_recommendation_model(w1)
    print(rec_website)
    return render_template("recommend.html", data=rec_website)  # Pass data to template

if __name__ == "__main__":
    app.run(debug=True)
