from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model at startup
def load_model():
    try:
        with open('svc_model_80_20.pkl', 'rb') as f:
            model = pickle.load(f)
            print("Model loaded successfully.")  # Debugging line
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        model = None
        print(f"Error loading model: {e}")  # Debugging line
    return model

model = load_model()

# Sentiment analysis function
def analyze_tweet_sentiment(text):
    if model is None:
        print("Model not loaded!")  # Debugging line
        return None, "Model not loaded"
    
    # Check the type of the model
    print(f"Model type: {type(model)}")  # Debugging line
    
    # Print the received text
    print(f"Received text: {text}")  # Debugging line
    
    try:
        # Predict sentiment using the model
        sentiment = model.predict([text])[0]  # Ensure the input is a list of texts
        print(f"Predicted sentiment: {sentiment}")  # Debugging line
    except Exception as e:
        print(f"Error during prediction: {e}")  # Debugging line
        return None, "Error during prediction"

    return sentiment, None

# Main route for sentiment analysis HTML form
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis</title>
    </head>
    <body>
        <h1>Analyze Sentiment</h1>
        <form action="/analyze" method="POST">
            <label for="message">Enter your message:</label>
            <input type="text" id="message" name="message">
            <button type="submit">Analyze</button>
        </form>
    </body>
    </html>
    '''

# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running'}), 200

# API route for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    tweet_text = request.form.get('message')
    if not tweet_text:
        return jsonify({'error': 'No message provided'}), 400
    
    sentiment, error = analyze_tweet_sentiment(tweet_text)
    
    # Check if there was an error during prediction
    if sentiment is None:
        return jsonify({'error': error}), 500
    
    return jsonify({
        'message': tweet_text,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Set debug=False for production
