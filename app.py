from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import io
from flask import Flask, render_template, request, redirect
import pandas as pd
import pickle
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import joblib
import matplotlib.pyplot as plt
import matplotlib
import csv
import requests
from bs4 import BeautifulSoup
from dateutil import parser
matplotlib.use('Agg')

# All Vectorizers Model for Transform the new text according to model vectorizer
vectorizerEnternment = joblib.load(
    'D:\Big Data Analytics of Airline Company\Final model with transformation\Enternment_services\Entertenment_service_vectrorizer.pkl')
vectorizerFoodCatering = joblib.load(
    'D:\Big Data Analytics of Airline Company\Final model with transformation\Food_Catering\Food_Catering_service_vectrorizer.pkl')
vectorizerGroundService = joblib.load(
    'D:\Big Data Analytics of Airline Company\Final model with transformation\Ground_service\Ground_service_vectrorizer.pkl')
vectorizerSeatComfort = joblib.load(
    'D:\Big Data Analytics of Airline Company\Final model with transformation\Comfort_seat_service\Comfart_seat_service_vectrorizer.pkl')
vectorizerInFlight = joblib.load(
    'D:\Big Data Analytics of Airline Company\Final model with transformation\In_Flight_services\In_Flight_service_vectrorizer.pkl')
vectorizerRecommended = joblib.load(
    'D:\Big Data Analytics of Airline Company\Final model with transformation\Recommendation_flight\Recommendation_service_vectrorizer.pkl')

app = Flask(__name__, static_url_path='/static')

# Load sentimental analysis model and vectorizer
nltk.download('punkt')
nltk.download('stopwords')
vectorizer = TfidfVectorizer()

# All Load Models
loadModel_EnternamentService = pickle.load(open(
    r'D:\Big Data Analytics of Airline Company\Final model with transformation\Enternment_services\trained_entertainment_service.sav', 'rb'))
loadModel_FoodCatering = pickle.load(open(
    r'D:\Big Data Analytics of Airline Company\Final model with transformation\Food_Catering\trained_food_catering_services.sav', 'rb'))
loadModel_GroundService = pickle.load(open(
    r'D:\Big Data Analytics of Airline Company\Final model with transformation\Ground_service\trained_ground_services.sav', 'rb'))
loadModel_SeatComfort = pickle.load(open(
    r'D:\Big Data Analytics of Airline Company\Final model with transformation\Comfort_seat_service\trained_seat_comfort_services.sav', 'rb'))
loadModel_InFlight = pickle.load(open(
    r'D:\Big Data Analytics of Airline Company\Final model with transformation\In_Flight_services\trained_InFlight_services.sav', 'rb'))
loadModel_RecommendedService = pickle.load(open(
    r'D:\Big Data Analytics of Airline Company\Final model with transformation\Recommendation_flight\trained_Recommended_services.sav', 'rb'))


# Preprocess Text Function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Date formating the data after loading


def convert_date(date_str):
    # Parse the date string
    date_obj = parser.parse(date_str, dayfirst=True)
    # Format the date as YYYY-MM-DD
    formatted_date = date_obj.strftime("%Y-%m-%d")
    return formatted_date

# Function to scrape reviews from a given URL


def scrape_reviews(url):
    reviews_data = []
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    review_blocks = soup.find_all(
        "h3", class_="text_sub_header userStatusWrapper")
    for review_block in review_blocks:
        # Extract date
        date = review_block.find(
            "time", itemprop="datePublished").get_text(strip=True)
        # Convert date format
        formatted_date = convert_date(date)
        # Extract and print feedback/comment
        text = review_block.find_next(
            "div", class_="text_content").get_text(strip=True)
        text = text.replace("✅Trip Verified|", "").replace(
            "Not Verified", "").replace("|", "")
        reviews_data.append({"Date": formatted_date, "Review": text})
    return reviews_data

# Function to scrape reviews from multiple pages


def scrape_multiple_pages(base_url, max_pages):
    all_reviews = []
    # Convert max_pages to integer
    max_pages = int(max_pages)
    for page in range(1, max_pages + 1):
        page_url = base_url + "page/{}/".format(page)
        reviews_on_page = scrape_reviews(page_url)
        all_reviews.extend(reviews_on_page)
    return all_reviews

# Filter the reviews according to our json file


def filter_reviews(rvList, keywords_file):

    with open(keywords_file, 'r') as f:
        keywords = json.load(f)

    # empty list initilized
    matched_reviews = []

    # Iterate over each review
    for review in rvList:
        # Iterate over each keyword
        for keyword in keywords:
            # Check if the keyword is present in the review text
            if keyword.lower() in review.lower():
                # If keyword is found, add the entire row to the matched_reviews list
                matched_reviews.append(review)
                # If a review matches a keyword, break the loop to avoid adding it to multiple categories
                break

    return matched_reviews


# Generate Prediction for Enternmenet Services
def generate_prediction_Entertenment_Services(text, loadModel_EnternamentService, vectorizerEnternment):

    keywords_file = r'D:\Big Data Analytics of Airline Company\Keyword_wise_Validation_Data\Entertainment_Keywords.json'
    text = filter_reviews(text, keywords_file)

    # output_file = 'data.txt'
    # with open(output_file, 'w') as file:
    #     # Write each element of the list to the file
    #     for item in text:
    #         file.write('%s\n' % item)

    total_confidence = [0, 0, 0]
    num_texts = len(text)

    for text_item in text:
        # Convert text_item to string to handle float values
        text_item = str(text_item)

        # Text Preprocessing
        preprocessed_text = preprocess_text(text_item)
        X_pred_transform = vectorizerEnternment.transform(
            [preprocessed_text])

        # Load Model
        confidence = loadModel_EnternamentService.predict_proba(X_pred_transform)[
            0]

        # Accumulate confidence scores
        total_confidence[0] += confidence[0]  # Negative
        total_confidence[1] += confidence[1]  # Neutral
        total_confidence[2] += confidence[2]  # Positive

    avg_confidence = [score / num_texts for score in total_confidence]

    return avg_confidence


# Generate Prediction for FoodCatering Services
def generate_prediction_Food_Catering_Services(text, loadModel_FoodCatering, vectorizerFoodCatering):

    keywords_file = r'D:\Big Data Analytics of Airline Company\Keyword_wise_Validation_Data\Food_Keywords.json'
    text = filter_reviews(text, keywords_file)

    # Initialize total confidence scores for negative, neutral, and positive
    total_confidence = [0, 0, 0]
    num_texts = len(text)

    for text_item in text:
        # Convert text_item to string to handle float values
        text_item = str(text_item)

        # Text Preprocessing
        preprocessed_text = preprocess_text(text_item)
        X_pred_transform = vectorizerFoodCatering.transform(
            [preprocessed_text])

        # Load Model
        confidence = loadModel_FoodCatering.predict_proba(X_pred_transform)[0]

        # Accumulate confidence scores
        total_confidence[0] += confidence[0]  # Negative
        total_confidence[1] += confidence[1]  # Neutral
        total_confidence[2] += confidence[2]  # Positive

    avg_confidence = [score / num_texts for score in total_confidence]

    return avg_confidence


# Generate Prediction for Ground Services
def generate_prediction_Ground_Services(text, loadModel_GroundService, vectorizerGroundService):

    keywords_file = r'D:\Big Data Analytics of Airline Company\Keyword_wise_Validation_Data\Ground_Keywords.json'
    text = filter_reviews(text, keywords_file)

    total_confidence = [0, 0, 0]
    num_texts = len(text)

    for text_item in text:
        # Convert text_item to string to handle float values
        text_item = str(text_item)

        # Text Preprocessing
        preprocessed_text = preprocess_text(text_item)
        X_pred_transform = vectorizerGroundService.transform(
            [preprocessed_text])

        # Load Model
        confidence = loadModel_GroundService.predict_proba(X_pred_transform)[
            0]

        # Accumulate confidence scores
        total_confidence[0] += confidence[0]  # Negative
        total_confidence[1] += confidence[1]  # Neutral
        total_confidence[2] += confidence[2]  # Positive

    avg_confidence = [score / num_texts for score in total_confidence]

    return avg_confidence


# Generate Prediction for Seat Comfart Services
def generate_prediction_SeatComfart_Services(text, loadModel_SeatComfort, vectorizerSeatComfort):

    keywords_file = r'D:\Big Data Analytics of Airline Company\Keyword_wise_Validation_Data\Seat_Keywords.json'
    text = filter_reviews(text, keywords_file)

    total_confidence = [0, 0, 0]
    num_texts = len(text)

    for text_item in text:
        # Convert text_item to string to handle float values
        text_item = str(text_item)

        # Text Preprocessing
        preprocessed_text = preprocess_text(text_item)
        X_pred_transform = vectorizerSeatComfort.transform(
            [preprocessed_text])

        # Load Model
        confidence = loadModel_SeatComfort.predict_proba(X_pred_transform)[
            0]

        # Accumulate confidence scores
        total_confidence[0] += confidence[0]  # Negative
        total_confidence[1] += confidence[1]  # Neutral
        total_confidence[2] += confidence[2]  # Positive

    avg_confidence = [score / num_texts for score in total_confidence]

    return avg_confidence


# Generate Prediction for InFlight Services
def generate_prediction_InFlight_Services(text, loadModel_InFlight, vectorizerInFlight):

    keywords_file = r'D:\Big Data Analytics of Airline Company\Keyword_wise_Validation_Data\InFlight_Keywords.json'
    text = filter_reviews(text, keywords_file)

    total_confidence = [0, 0, 0]
    num_texts = len(text)

    for text_item in text:
        # Convert text_item to string to handle float values
        text_item = str(text_item)

        # Text Preprocessing
        preprocessed_text = preprocess_text(text_item)
        X_pred_transform = vectorizerInFlight.transform(
            [preprocessed_text])

        # Load Model
        confidence = loadModel_InFlight.predict_proba(X_pred_transform)[
            0]

        # Accumulate confidence scores
        total_confidence[0] += confidence[0]  # Negative
        total_confidence[1] += confidence[1]  # Neutral
        total_confidence[2] += confidence[2]  # Positive

    avg_confidence = [score / num_texts for score in total_confidence]

    return avg_confidence


# Generate Prediction for Recommended flight or not
def generate_prediction_Recommendation_flight(text, loadModel_RecommendedService, vectorizerRecommended):

    keywords_file = r'D:\Big Data Analytics of Airline Company\Keyword_wise_Validation_Data\Recommendation_Keywords.json'
    text = filter_reviews(text, keywords_file)

    total_confidence = [0, 0]
    num_texts = len(text)

    for text_item in text:
        # Convert text_item to string to handle float values
        text_item = str(text_item)

        # Text Preprocessing
        preprocessed_text = preprocess_text(text_item)
        X_pred_transform = vectorizerRecommended.transform(
            [preprocessed_text])

        # Load Model
        confidence = loadModel_RecommendedService.predict_proba(X_pred_transform)[
            0]

        # Accumulate confidence scores
        total_confidence[0] += confidence[0]  # Negative
        total_confidence[1] += confidence[1]  # Neutral

    avg_confidence = [score / num_texts for score in total_confidence]

    return avg_confidence


@app.route('/')
def front():
    return render_template("front.html")


@app.route('/service1', methods=['GET', 'POST'])
def extract_data():
    if request.method == 'POST':
        data = request.json
        max_pages = data['maxPages']

        # Base URL of the airline reviews page
        base_url = "https://www.airlinequality.com/airline-reviews/air-india/"
        all_reviews = scrape_multiple_pages(base_url, max_pages)

        # Write scraped data to a CSV file
        csv_file = "./Pipeline data/air_india_reviews.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["Date", "Review"])
            writer.writeheader()
            writer.writerows(all_reviews)

        return render_template('index.html')

    # For GET requests, simply render the template
    return render_template('dataExtract.html')


@app.route('/service2')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    csv_file = request.files['csv_file']

    # # Read CSV file into pandas DataFrame
    csv_data = csv_file.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data))

    # # # Convert 'date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # # # Filter DataFrame based on selected date range
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    reviews = ''
    rvList = filtered_df['Review'].tolist()
    # Iterate through each review and concatenate it to the reviews string
    # for review in filtered_df['Review'].astype(str).tolist():
    #     try:
    #         # Join each review to the reviews string
    #         reviews += review + '\n'
    #     except Exception as e:
    #         continue

    # reviewss = filtered_df['Review'].items()
    # reviews = '\n'.join(reviewss)

    if request.method == 'POST':
        # Perform predictions for all services
        confidence_entertainment = generate_prediction_Entertenment_Services(
            rvList, loadModel_EnternamentService, vectorizerEnternment)
        confidence_food_catering = generate_prediction_Food_Catering_Services(
            rvList, loadModel_FoodCatering, vectorizerFoodCatering)
        confidence_ground_services = generate_prediction_Ground_Services(
            rvList, loadModel_GroundService, vectorizerGroundService)
        confidence_seat_comfort = generate_prediction_SeatComfart_Services(
            rvList, loadModel_SeatComfort, vectorizerSeatComfort)
        confidence_inflight_services = generate_prediction_InFlight_Services(
            rvList, loadModel_InFlight, vectorizerInFlight)
        confidence_recommended_services = generate_prediction_Recommendation_flight(
            rvList, loadModel_RecommendedService, vectorizerRecommended)

        # Entertement services cofidence score
        Enterntenment_positive = confidence_entertainment[2] * 100
        Enterntenment_neutral = confidence_entertainment[1] * 100
        Enterntenment_negative = confidence_entertainment[0] * 100

        # Create bar chart for Enternment Services
        labelsEntertenment = ['Positive Feedback',
                              'Neutral Feedback', 'Negative Feedback']
        valuesEnternment = [Enterntenment_positive,
                            Enterntenment_neutral, Enterntenment_negative]
        # Adjust figure size
        plt.figure(figsize=(6, 4))

# Define colors for each sentiment
        colors = ['skyblue', 'yellow', 'red']
        plt.bar(labelsEntertenment, valuesEnternment, color=colors)
        # Annotate each bar with its percentage value
        for i, value in enumerate(valuesEnternment):
            plt.text(i, value + 2, f"{value:.1f}%", ha='center')
        plt.ylim(0, 110)
        plt.title('Sentiment Analysis - Entertainment Services')
        plt.xlabel('Sentiments')
        plt.ylabel('Percentage')

        # Save the bar chart as a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image as base64 string
        plot_url_Entertenment = base64.b64encode(img.getvalue()).decode()

        # Close the plot to release memory
        plt.close()

        # Food Catering services cofidence score
        FoodCatering_positive = confidence_food_catering[2] * 100
        FoodCatering_neutral = confidence_food_catering[1] * 100
        FoodCatering_negative = confidence_food_catering[0] * 100

        # Create bar chart for Food Catering Services
        labelsFoodCatering = ['Positive Feedback',
                              'Neutral Feedback', 'Negative Feedback']
        valuesFoodCatering = [FoodCatering_positive,
                              FoodCatering_neutral, FoodCatering_negative]
        plt.figure(figsize=(6, 4))

# Define colors for each sentiment
        colors = ['skyblue', 'yellow', 'red']
        plt.bar(labelsFoodCatering, valuesFoodCatering, color=colors)
        # Annotate each bar with its percentage value
        for i, value in enumerate(valuesFoodCatering):
            plt.text(i, value + 2, f"{value:.1f}%", ha='center')
        plt.ylim(0, 110)
        plt.title('Sentiment Analysis - Food Catering Services')
        plt.xlabel('Sentiments')
        plt.ylabel('Percentage')

        # Save the bar chart as a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image as base64 string
        plot_url_FoodCatering = base64.b64encode(img.getvalue()).decode()

        # Close the plot to release memory
        plt.close()

        # Ground services cofidence score
        GroundService_positive = confidence_ground_services[2] * 100
        GroundService_neutral = confidence_ground_services[1] * 100
        GroundService_negative = confidence_ground_services[0] * 100

        # Create bar chart for Ground Services
        labelsGroundServices = ['Positive Feedback',
                                'Neutral Feedback', 'Negative Feedback']
        valuesGroundServices = [GroundService_positive,
                                GroundService_neutral, GroundService_negative]
        plt.figure(figsize=(6, 4))

# Define colors for each sentiment
        colors = ['skyblue', 'yellow', 'red']
        plt.bar(labelsGroundServices, valuesGroundServices, color=colors)
        # Annotate each bar with its percentage value
        for i, value in enumerate(valuesGroundServices):
            plt.text(i, value + 2, f"{value:.1f}%", ha='center')
        plt.ylim(0, 110)
        plt.title('Sentiment Analysis - Ground Services')
        plt.xlabel('Sentiments')
        plt.ylabel('Percentage')

        # Save the bar chart as a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image as base64 string
        plot_url_GroundServices = base64.b64encode(img.getvalue()).decode()

        # Close the plot to release memory
        plt.close()

        # Seat Comfart services cofidence score
        SeatComfort_positive = confidence_seat_comfort[2] * 100
        SeatComfort_neutral = confidence_seat_comfort[1] * 100
        SeatComfort_negative = confidence_seat_comfort[0] * 100

        # Create bar chart for SeatComfart Services
        labelsSeatComfartServices = ['Positive Feedback',
                                     'Neutral Feedback', 'Negative Feedback']
        valuesSeatComfartServices = [SeatComfort_positive,
                                     SeatComfort_neutral, SeatComfort_negative]
        plt.figure(figsize=(6, 4))

# Define colors for each sentiment
        colors = ['skyblue', 'yellow', 'red']
        plt.bar(labelsSeatComfartServices,
                valuesSeatComfartServices, color=colors)
        # Annotate each bar with its percentage value
        for i, value in enumerate(valuesSeatComfartServices):
            plt.text(i, value + 2, f"{value:.1f}%", ha='center')
        plt.ylim(0, 110)
        plt.title('Sentiment Analysis - Seat Comfart Services')
        plt.xlabel('Sentiments')
        plt.ylabel('Percentage')

        # Save the bar chart as a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image as base64 string
        plot_url_SeatComfartServices = base64.b64encode(
            img.getvalue()).decode()

        # Close the plot to release memory
        plt.close()

        # InFlight services cofidence score
        Inflight_positive = confidence_inflight_services[2] * 100
        Inflight_neutral = confidence_inflight_services[1] * 100
        Inflight_negative = confidence_inflight_services[0] * 100

        # Create bar chart for InFlight Services
        labelsInFlightServices = ['Positive Feedback',
                                  'Neutral Feedback', 'Negative Feedback']
        valuesInFlightServices = [Inflight_positive,
                                  Inflight_neutral, Inflight_negative]
        plt.figure(figsize=(6, 4))

# Define colors for each sentiment
        colors = ['skyblue', 'yellow', 'red']
        plt.bar(labelsInFlightServices, valuesInFlightServices, color=colors)
        # Annotate each bar with its percentage value
        for i, value in enumerate(valuesInFlightServices):
            plt.text(i, value + 2, f"{value:.1f}%", ha='center')
        plt.ylim(0, 110)
        plt.title('Sentiment Analysis - In Flight Services')
        plt.xlabel('Sentiments')
        plt.ylabel('Percentage')

        # Save the bar chart as a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image as base64 string
        plot_url_InFlight_Services = base64.b64encode(
            img.getvalue()).decode()

        # Close the plot to release memory
        plt.close()

        # OverAll services cofidence score
        OverAll_positive = (Enterntenment_positive + FoodCatering_positive +
                            GroundService_positive + SeatComfort_positive + Inflight_positive) / 5
        OverAll_neutral = (Enterntenment_neutral + FoodCatering_neutral +
                           GroundService_neutral + SeatComfort_neutral + Inflight_neutral) / 5
        OverAll_negative = (Enterntenment_negative + FoodCatering_negative +
                            GroundService_negative + SeatComfort_negative + Inflight_negative) / 5

        # Create bar chart for OverAll Services
        labelsOverAllServices = ['Positive Feedback',
                                 'Neutral Feedback', 'Negative Feedback']
        valuesOverAllServices = [OverAll_positive,
                                 OverAll_neutral, OverAll_negative]
        plt.figure(figsize=(6, 4))

# Define colors for each sentiment
        colors = ['skyblue', 'yellow', 'red']
        plt.bar(labelsOverAllServices, valuesOverAllServices, color=colors)
        # Annotate each bar with its percentage value
        for i, value in enumerate(valuesOverAllServices):
            plt.text(i, value + 2, f"{value:.1f}%", ha='center')
        plt.ylim(0, 110)
        plt.title('Sentiment Analysis - Over All Services')
        plt.xlabel('Sentiments')
        plt.ylabel('Percentage')

        # Save the bar chart as a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image as base64 string
        plot_url_OverAll_Services = base64.b64encode(
            img.getvalue()).decode()

        # Close the plot to release memory
        plt.close()

        # Plot graph for Recommendations
        recommendedYes = confidence_recommended_services[1] * 100
        recommendedNo = confidence_recommended_services[0] * 100

        labelsRecommendation = ['Recommended', 'Not Recommended']
        valuesRecommendation = [recommendedYes, recommendedNo]

        plt.figure(figsize=(6, 4))
        plt.pie(valuesRecommendation, labels=labelsRecommendation,
                autopct='%1.1f%%', startangle=140)
        plt.title('Recommended Services')
        plt.axis('equal')

        # Convert plot to PNG image
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        plt.close()

        # Encode PNG image to base64 string
        plot_url_recommended_services = base64.b64encode(
            image_stream.getvalue()).decode('utf-8')

        return render_template('result.html', plot_entertainment=plot_url_Entertenment,
                               plot_food_catering=plot_url_FoodCatering,
                               plot_Ground=plot_url_GroundServices,
                               plot_SeatComfart=plot_url_SeatComfartServices,
                               plot_inFlight=plot_url_InFlight_Services,
                               plot_OverAll=plot_url_OverAll_Services,
                               plot_recommended=plot_url_recommended_services
                               )
    return redirect('/')


if __name__ == '__main__':
    app.run()
