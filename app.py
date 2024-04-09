import sys
from src.exception import custom_exception
from flask  import Flask, render_template, request
from src.pipeline.predict_pipe import PredictionPipeline, CustomData


app = Flask(__name__)

 
@app.route('/', methods=['GET','POST'])
def predict():
    try:
        if request.method == 'POST':
            input_data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )
            
            DataFrame = input_data.to_dataframe()
            prediction_pipe = PredictionPipeline()
            prediction = prediction_pipe.prediction(DataFrame)
            print("Prediction:", prediction)
            formatted_prediction = round(prediction, 2)
            return render_template('index.html', prediction=formatted_prediction)
            
            
        else:  
            return render_template('index.html')
    
    except Exception as e:
        raise custom_exception(e, sys)




if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)