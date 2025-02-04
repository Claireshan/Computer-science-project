import os
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

models_dir = "model"  # Combined directory for all models
sarima_models = {}
rf_models = {}

try:
    for filename in os.listdir(models_dir):
        if filename.endswith('sarima_model.pkl'):
            #model_name = filename.replace('_sarima_model.pkl', '')  # Clean model name
            model_name = filename.replace('_sarima_model.pkl', '')  # Clean model name
            sarima_models[model_name] = joblib.load(os.path.join(models_dir, filename))
        elif filename.endswith('rf_model.pkl'):
            model_name = filename.replace('_rf_model.pkl', '')  # Clean model name
            rf_models[model_name] = joblib.load(os.path.join(models_dir, filename))
except Exception as e:
    raise Exception(f"Error loading models: {e}")

# Load preprocessed access point names from CSV
try:
    ap_names_df = pd.read_csv('model/mesh_nodes.csv') 
    ap_names = ap_names_df['name'].dropna().unique().tolist() 
except Exception as e:
    raise Exception(f"Error loading access point names from CSV: {e}")

# Helper function to preprocess features
def preprocess_features(start_date, end_date):
    """
    Convert input data into model features.
    """
    # ap_name_encoded = hash(ap_name) % 100  # Simple hash for example
    duration = (end_date - start_date).days
    features = np.array([duration])
    return features

@app.route('/')
def home():  
    # Create an empty Plotly figure
    fig = go.Figure()
    fig.update_layout(
        autosize=True,
        title="No Data Available",
        xaxis_title="Date",
        yaxis_title="Predicted Value",
        annotations=[dict(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No Data Available",
            showarrow=False,
            font=dict(size=16, color="gray")
        )]
    )

    # Convert to HTML
    plot_html = fig.to_html(full_html=False)
    return render_template('index.html', ap_names=ap_names,model_name=sarima_models.keys(), plot_html=plot_html)

@app.route('/bandwidth')
def bandwidth():  
    # Create an empty Plotly figure
    fig = go.Figure()
    fig.update_layout(
        autosize=True,
        title="No Data Available",
        xaxis_title="Date",
        yaxis_title="Predicted Value",
        annotations=[dict(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No Data Available",
            showarrow=False,
            font=dict(size=16, color="gray")
        )]
    )
   # model_name = list(rf_models.keys())
    # Convert to HTML
    plot_html = fig.to_html(full_html=False)
    return render_template('bandwidth.html', ap_names=ap_names, bwmodel_name=rf_models.keys(), plot_html=plot_html)


#@app.route('/get_models/<category>', methods=['GET'])
#def get_models(category):
#    if category == 'users':
#        return jsonify(list(sarima_models.keys()))
#    elif category == 'bandwidth':
#        return jsonify(list(rf_models.keys()))
#    else:
#        return jsonify({'error': 'Invalid category'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ap_name = request.form['ap_name']
        start_date = pd.to_datetime(request.form['start_date'])
        end_date = pd.to_datetime(request.form['end_date'])
        selected_model_name = request.form.get('model_name', None)  # Model name from user

        if not selected_model_name or selected_model_name not in sarima_models:
            return jsonify({'error': 'Invalid or missing model selection'}), 400

        # Validate dates
        if start_date > end_date:
            return jsonify({'error': 'Start date must be earlier than end date'}), 400

        # Get the selected model
        selected_model = sarima_models[selected_model_name]

        # Generate the date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Predict values for the date range
        num_days = len(date_range)
        predictions = selected_model.forecast(steps=num_days)

        # Generate a Plotly graph of predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=date_range,
            y=predictions,
            mode='lines',
            name=f"Predicted Values"
        ))
        fig.update_layout(
            title=f"Predictions for {ap_name} ({start_date.date()} to {end_date.date()})",
            xaxis_title="Date",
            yaxis_title="Predicted Value"
        )

        # Convert Plotly figure to HTML
        plot_html = fig.to_html(full_html=False)

        return render_template(
            'index.html',
            ap_names=ap_names,
            plot_html=plot_html,
            model_names=list(sarima_models.keys()),
            prediction_summary=f"Predictions generated for {num_days} days."
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/bw_predict', methods=['POST'])
def bwPredict():
    try:
        ap_name = request.form['ap_name']
        start_date = pd.to_datetime(request.form['start_date'])
        end_date = pd.to_datetime(request.form['end_date'])
        selected_model_name = request.form.get('bwmodel_name', None)  # Model name from user

        if not selected_model_name or selected_model_name not in rf_models:
            return jsonify({'error': 'Invalid or missing model selection'}), 400

        # Validate dates
        if start_date > end_date:
            return jsonify({'error': 'Start date must be earlier than end date'}), 400

        # Get the selected model
        selected_model = rf_models[selected_model_name]

        # Generate the date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Predict values for the date range
        num_days = len(date_range)
        predictions = selected_model.predict([[num_days]])
        #predictions = selected_model.forecast(steps=num_days)

        # Generate a Plotly graph of predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=date_range,
            y=predictions,
            mode='lines',
            name=f"Predicted Values"
        ))
        fig.update_layout(
            title=f"Predictions for {ap_name} ({start_date.date()} to {end_date.date()})",
            xaxis_title="Date",
            yaxis_title="Predicted Value"
        )

        # Convert Plotly figure to HTML
        plot_html = fig.to_html(full_html=False)

        return render_template(
            'bandwidth.html',
            ap_names=ap_names,
            plot_html=plot_html,
            bwmodel_names=list(rf_models.keys()),
            prediction_summary=f"Predictions generated for {num_days} days."
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
