# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup,redirect
from flask_cors import CORS
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

# from newsapi import NewsApiClient
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load( disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model


crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
CORS(app)
# render home page


@ app.route('/')
def home():
    title = 'Culti Vision - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Culti Vision - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Culti Vision - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

@app.route('/support')
def support_a_farmer():
    title = 'Support a Farmer'

    return render_template('support.html',title=title)
@app.route('/news')
def news():
    title = 'Welcome to the News page'
    api_key = "bb69cf460c550afa0647823b69f2d0aa"  # Replace with your GNews API key
    keyword = "weather"
    country = "in"  # ISO code for India
    url = f"https://gnews.io/api/v4/search?q={keyword}&country={country}&token={api_key}&lang=en"
    
    response = requests.get(url)
    data = response.json()
    articles = data['articles']

    desc = []
    news = []
    img = []

    for article in articles:
        news.append(article['title'])
        desc.append(article['description'])
        img.append(article['image'])

    mylist = zip(news, desc, img)

    return render_template('news.html', context=mylist, title=title)
# @app.route('/news')
# def news():
#     title = 'Welcome to the News page'
#     newsapi = NewsApiClient(api_key="5096bb6dcd6746e0885fd15197dcc809")
#     topheadlines = newsapi.get_top_headlines(sources="al-jazeera-english")
#     articles = topheadlines['articles']
 
#     desc = []
#     news = []
#     img = []
 
 
#     for i in range(len(articles)):
#         myarticles = articles[i]
 
 
#         news.append(myarticles['title'])
#         desc.append(myarticles['description'])
#         img.append(myarticles['urlToImage'])
 
 
 
#     mylist = zip(news, desc, img)
 
 
#     return render_template('news.html',context = mylist,title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Culti Vision - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Culti Vision - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    # df = pd.read_csv('fertilizer.csv')
    import pandas as pd
    from io import StringIO
# Assuming the CSV data is stored in a string variable 'csv_data'
    csv_data = ''',Crop,N,P,K,pH,soil_moisture
    0,rice,80,40,40,5.5,30
    3,maize,80,40,20,5.5,50
    5,chickpea,40,60,80,5.5,60
    12,kidneybeans,20,60,20,5.5,45
    13,pigeonpeas,20,60,20,5.5,45
    14,mothbeans,20,40,20,5.5,30
    15,mungbean,20,40,20,5.5,80
    18,blackgram,40,60,20,5,60
    24,lentil,20,60,20,5.5,90
    60,pomegranate,20,10,40,5.5,30
    61,banana,100,75,50,6.5,40
    62,mango,20,20,30,5,15
    63,grapes,20,125,200,4,60
    66,watermelon,100,10,50,5.5,70
    67,muskmelon,100,10,50,5.5,30
    69,apple,20,125,200,6.5,50
    74,orange,20,10,10,4,60
    75,papaya,50,50,50,6,20
    88,coconut,20,10,30,5,45
    93,cotton,120,40,20,5.5,70
    94,jute,80,40,40,5.5,20
    95,coffee,100,20,30,5.5,20'''
  
    # Read the CSV data into a DataFrame
    df = pd.read_csv(StringIO(csv_data))

    # Display the DataFrame
    print(df)


    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Culti Vision - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
