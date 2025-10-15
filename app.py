from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)
app=application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# When we visit the URL directly in a browser → the GET block runs. The default HTTP method is GET
# When we submit the form → it sends a POST request → the POST block runs.

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Extract data from form
            data = CustomData(
                    MSZoning=request.form.get('MSZoning'),
                    Neighborhood=request.form.get('Neighborhood'),
                    LotFrontage=float(request.form.get('LotFrontage')),
                    LotArea=int(request.form.get('LotArea')),
                    OverallQual=int(request.form.get('OverallQual')),
                    OverallCond=int(request.form.get('OverallCond')),
                    YearBuilt=int(request.form.get('YearBuilt')),
                    YearRemodAdd=int(request.form.get('YearRemodAdd')),
                    GrLivArea=int(request.form.get('GrLivArea')),
                    FullBath=int(request.form.get('FullBath')),
                    BedroomAbvGr=int(request.form.get('BedroomAbvGr')),
                    KitchenQual=request.form.get('KitchenQual'),
                    GarageCars=int(request.form.get('GarageCars')),
                    GarageArea=int(request.form.get('GarageArea')),
                    Fireplaces=int(request.form.get('Fireplaces')),
                    TotalBsmtSF=int(request.form.get('TotalBsmtSF'))
                )
            # Covert the above data into dataframe
            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            # Prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=int(results[0]))
        except Exception as e:
            return render_template('home.html', results=f"Error: {str(e)}")
       
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)