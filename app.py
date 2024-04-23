from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import io
from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import joblib
import matplotlib.pyplot as plt
import matplotlib
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
vectorizerOverAll = joblib.load(
    'D:\Big Data Analytics of Airline Company\Final model with transformation\OverAll_Services\OverAll_service_vectrorizer.pkl')
vectorizerRecommended = joblib.load(
    'D:\Big Data Analytics of Airline Company\Final model with transformation\Recommendation_flight\Recommendation_service_vectrorizer.pkl')

app = Flask(__name__)

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
loadModel_OverAll = pickle.load(open(
    r'D:\Big Data Analytics of Airline Company\Final model with transformation\OverAll_Services\trained_OverAll_services.sav', 'rb'))
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


# Generate Prediction for Enternmenet Services
def generate_prediction_Entertenment_Services(text, loadModel_EnternamentService, vectorizerEnternment):

    # Text Preprocessing
    preprocessed_text = preprocess_text(text)
    X_pred_transform = vectorizerEnternment.transform([preprocessed_text])
    # Load Model
    prediction = loadModel_EnternamentService.predict(X_pred_transform)
    confidence = loadModel_EnternamentService.predict_proba(X_pred_transform)[
        0]
    return prediction, confidence


# Generate Prediction for FoodCatering Services
def generate_prediction_Food_Catering_Services(text, loadModel_FoodCatering, vectorizerFoodCatering):

    # Text Preprocessing
    preprocessed_text = preprocess_text(text)
    X_pred_transform = vectorizerFoodCatering.transform([preprocessed_text])
    # Load Model
    prediction = loadModel_FoodCatering.predict(X_pred_transform)
    confidence = loadModel_FoodCatering.predict_proba(X_pred_transform)[
        0]
    return prediction, confidence


# Generate Prediction for Ground Services
def generate_prediction_Ground_Services(text, loadModel_GroundService, vectorizerGroundService):

    # Text Preprocessing
    preprocessed_text = preprocess_text(text)
    X_pred_transform = vectorizerGroundService.transform([preprocessed_text])
    # Load Model
    prediction = loadModel_GroundService.predict(X_pred_transform)
    confidence = loadModel_GroundService.predict_proba(X_pred_transform)[
        0]
    return prediction, confidence


# Generate Prediction for Seat Comfart Services
def generate_prediction_SeatComfart_Services(text, loadModel_SeatComfort, vectorizerSeatComfort):

    # Text Preprocessing
    preprocessed_text = preprocess_text(text)
    X_pred_transform = vectorizerSeatComfort.transform([preprocessed_text])
    # Load Model
    prediction = loadModel_SeatComfort.predict(X_pred_transform)
    confidence = loadModel_SeatComfort.predict_proba(X_pred_transform)[
        0]
    return prediction, confidence


# Generate Prediction for InFlight Services
def generate_prediction_InFlight_Services(text, loadModel_InFlight, vectorizerInFlight):

    # Text Preprocessing
    preprocessed_text = preprocess_text(text)
    X_pred_transform = vectorizerInFlight.transform([preprocessed_text])
    # Load Model
    prediction = loadModel_InFlight.predict(X_pred_transform)
    confidence = loadModel_InFlight.predict_proba(X_pred_transform)[
        0]
    return prediction, confidence


# Generate Prediction for OverAll Services
def generate_prediction_OverAll_Services(text, loadModel_OverAll, vectorizerOverAll):

    # Text Preprocessing
    preprocessed_text = preprocess_text(text)
    X_pred_transform = vectorizerOverAll.transform([preprocessed_text])
    # Load Model
    prediction = loadModel_OverAll.predict(X_pred_transform)
    confidence = loadModel_OverAll.predict_proba(X_pred_transform)[
        0]
    return prediction, confidence


# Generate Prediction for Recommended flight or not
def generate_prediction_Recommendation_flight(text, loadModel_RecommendedService, vectorizerRecommended):

    # Text Preprocessing
    preprocessed_text = preprocess_text(text)
    X_pred_transform = vectorizerRecommended.transform([preprocessed_text])
    # Load Model
    prediction = loadModel_RecommendedService.predict(X_pred_transform)
    confidence = loadModel_RecommendedService.predict_proba(X_pred_transform)[
        0]
    return prediction, confidence


# @app.route('/')
# def index():
#     return render_template("index.html")


# @app.route('/analyze/entertainment', methods=['POST'])
# def analyze_entertainment():
#     if request.method == 'POST':
#         text_input = request.form['text_input']
#         prediction, confidence = generate_prediction_Entertenment_Services(
#             text_input, loadModel_EnternamentService, vectorizerEnternment)
#         positive_confidence = confidence[2] * 100
#         neutral_confidence = confidence[1] * 100
#         negative_confidence = confidence[0] * 100
#         prediction_result = f"Positive = {positive_confidence:.2f}% confidence\n Neutral = {neutral_confidence:.2f}% confidence\n Negative = {negative_confidence:.2f}% confidence"
#         return render_template('index.html', prediction=prediction_result)
#     return redirect('/')


# @app.route('/analyze/food_catering', methods=['POST'])
# def analyze_food_catering():
#     if request.method == 'POST':
#         text_input = request.form['text_input']
#         prediction, confidence = generate_prediction_Food_Catering_Services(
#             text_input, loadModel_FoodCatering, vectorizerFoodCatering)
#         positive_confidence = confidence[2] * 100
#         neutral_confidence = confidence[1] * 100
#         negative_confidence = confidence[0] * 100
#         prediction_result = f"Positive = {positive_confidence:.2f}% confidence\n Neutral = {neutral_confidence:.2f}% confidence\n Negative = {negative_confidence:.2f}% confidence"
#         return render_template('index.html', prediction=prediction_result)
#     return redirect('/')


# @app.route('/analyze/ground_services', methods=['POST'])
# def analyze_ground_services():
#     if request.method == 'POST':
#         text_input = request.form['text_input']
#         prediction, confidence = generate_prediction_Ground_Services(
#             text_input, loadModel_GroundService, vectorizerGroundService)
#         positive_confidence = confidence[2] * 100
#         neutral_confidence = confidence[1] * 100
#         negative_confidence = confidence[0] * 100
#         prediction_result = f"Positive = {positive_confidence:.2f}% confidence\n Neutral = {neutral_confidence:.2f}% confidence\n Negative = {negative_confidence:.2f}% confidence"
#         return render_template('index.html', prediction=prediction_result)
#     return redirect('/')


# @app.route('/analyze/seat_comfort', methods=['POST'])
# def analyze_seat_comfort():
#     if request.method == 'POST':
#         text_input = request.form['text_input']
#         prediction, confidence = generate_prediction_SeatComfart_Services(
#             text_input, loadModel_SeatComfort, vectorizerSeatComfort)
#         positive_confidence = confidence[2] * 100
#         neutral_confidence = confidence[1] * 100
#         negative_confidence = confidence[0] * 100
#         prediction_result = f"Positive = {positive_confidence:.2f}% confidence\n Neutral = {neutral_confidence:.2f}% confidence\n Negative = {negative_confidence:.2f}% confidence"
#         return render_template('index.html', prediction=prediction_result)
#     return redirect('/')


# @app.route('/analyze/inflight_services', methods=['POST'])
# def analyze_inflight_services():
#     if request.method == 'POST':
#         text_input = request.form['text_input']
#         prediction, confidence = generate_prediction_InFlight_Services(
#             text_input, loadModel_InFlight, vectorizerInFlight)
#         positive_confidence = confidence[2] * 100
#         neutral_confidence = confidence[1] * 100
#         negative_confidence = confidence[0] * 100
#         prediction_result = f"Positive = {positive_confidence:.2f}% confidence\n Neutral = {neutral_confidence:.2f}% confidence\n Negative = {negative_confidence:.2f}% confidence"
#         return render_template('index.html', prediction=prediction_result)
#     return redirect('/')


# @app.route('/analyze/overall_services', methods=['POST'])
# def analyze_overall_services():
#     if request.method == 'POST':
#         text_input = request.form['text_input']
#         prediction, confidence = generate_prediction_OverAll_Services(
#             text_input, loadModel_OverAll, vectorizerOverAll)
#         positive_confidence = confidence[2] * 100
#         neutral_confidence = confidence[1] * 100
#         negative_confidence = confidence[0] * 100
#         prediction_result = f"Positive = {positive_confidence:.2f}% confidence\n Neutral = {neutral_confidence:.2f}% confidence\n Negative = {negative_confidence:.2f}% confidence"
#         return render_template('index.html', prediction=prediction_result)
#     return redirect('/')


# @app.route('/analyze/recommended_services', methods=['POST'])
# def analyze_recommended_services():
#     if request.method == 'POST':
#         text_input = request.form['text_input']
#         prediction, confidence = generate_prediction_Recommendation_flight(
#             text_input, loadModel_RecommendedService, vectorizerRecommended)
#         recommendedYes = confidence[1] * 100
#         recommendedNo = confidence[0] * 100
#         prediction_result = f"Recommended = {recommendedYes:.2f}% confidence\n Not Recommended = {recommendedYes:.2f}% confidence"
#         return render_template('index.html', prediction=prediction_result)
#     return redirect('/')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def process():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    csv_file = request.files['csv_file']

    # Read CSV file into pandas DataFrame
    csv_data = csv_file.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data))

    # # Convert 'date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # # Filter DataFrame based on selected date range
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # # Extract 'review' column
    reviewss = filtered_df['Review'].tolist()

    reviews = '\n'.join(reviewss)
    # return reviews
    if request.method == 'POST':
        # Perform predictions for all services
        prediction_entertainment, confidence_entertainment = generate_prediction_Entertenment_Services(
            reviews, loadModel_EnternamentService, vectorizerEnternment)
        prediction_food_catering, confidence_food_catering = generate_prediction_Food_Catering_Services(
            reviews, loadModel_FoodCatering, vectorizerFoodCatering)
        prediction_ground_services, confidence_ground_services = generate_prediction_Ground_Services(
            reviews, loadModel_GroundService, vectorizerGroundService)
        prediction_seat_comfort, confidence_seat_comfort = generate_prediction_SeatComfart_Services(
            reviews, loadModel_SeatComfort, vectorizerSeatComfort)
        prediction_inflight_services, confidence_inflight_services = generate_prediction_InFlight_Services(
            reviews, loadModel_InFlight, vectorizerInFlight)
        prediction_overall_services, confidence_overall_services = generate_prediction_OverAll_Services(
            reviews, loadModel_OverAll, vectorizerOverAll)
        prediction_recommended_services, confidence_recommended_services = generate_prediction_Recommendation_flight(
            reviews, loadModel_RecommendedService, vectorizerRecommended)

        # Entertement services cofidence score
        Enterntenment_positive = confidence_entertainment[2] * 100
        Enterntenment_neutral = confidence_entertainment[1] * 100
        Enterntenment_negative = confidence_entertainment[0] * 100

        # Create bar chart for Enternment Services
        labelsEntertenment = ['Positive Feedback',
                              'Neutral Feedback', 'Negative Feedback']
        valuesEnternment = [Enterntenment_positive,
                            Enterntenment_neutral, Enterntenment_negative]
        plt.bar(labelsEntertenment, valuesEnternment)
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
        plt.bar(labelsFoodCatering, valuesFoodCatering)
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
        plt.bar(labelsGroundServices, valuesGroundServices)
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
        plt.bar(labelsSeatComfartServices, valuesSeatComfartServices)
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
        plt.bar(labelsInFlightServices, valuesInFlightServices)
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
        OverAll_positive = confidence_overall_services[2] * 100
        OverAll_neutral = confidence_overall_services[1] * 100
        OverAll_negative = confidence_overall_services[0] * 100

        # Create bar chart for OverAll Services
        labelsOverAllServices = ['Positive Feedback',
                                 'Neutral Feedback', 'Negative Feedback']
        valuesOverAllServices = [OverAll_positive,
                                 OverAll_neutral, OverAll_negative]
        plt.bar(labelsOverAllServices, valuesOverAllServices)
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

        plt.figure(figsize=(8, 6))
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
                               #    prediction_entertainment=f"Positive = {Enterntenment_positive:.2f}% confidence\n Neutral = {Enterntenment_neutral:.2f}% confidence\n Negative = {Enterntenment_negative:.2f}% confidence",
                               #    prediction_food_catering=f"Positive = {FoodCatering_positive:.2f}% confidence\n Neutral = {FoodCatering_neutral:.2f}% confidence\n Negative = {FoodCatering_negative:.2f}% confidence",
                               #    prediction_ground_services=f"Positive = {GroundService_positive:.2f}% confidence\n Neutral = {GroundService_neutral:.2f}% confidence\n Negative = {GroundService_negative:.2f}% confidence",
                               #    prediction_seat_comfort=f"Positive = {SeatComfort_positive:.2f}% confidence\n Neutral = {SeatComfort_neutral:.2f}% confidence\n Negative = {SeatComfort_negative:.2f}% confidence",
                               #    prediction_inflight_services=f"Positive = {Inflight_positive:.2f}% confidence\n Neutral = {Inflight_neutral:.2f}% confidence\n Negative = {Inflight_negative:.2f}% confidence",
                               #    prediction_overall_services=f"Positive = {OverAll_positive:.2f}% confidence\n Neutral = {OverAll_neutral:.2f}% confidence\n Negative = {OverAll_negative:.2f}% confidence",
                               #    prediction_recommended_services=f"Recommended = {recommendedYes:.2f}% confidence\n Not Recommended = {recommendedNo:.2f}% confidence"
                               )
    return redirect('/')


if __name__ == '__main__':
    app.run()
