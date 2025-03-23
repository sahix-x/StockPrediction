import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime, timedelta
import logging
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
from google.cloud import language_v1, storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data"
MODEL_DIR = "models"
SENTIMENT_FILE = os.path.join(DATA_DIR, "sentiment_scores.json")
PREDICTION_OUTPUT = os.path.join(DATA_DIR, "stock_predictions.json")
DAYS_OF_HISTORY = 60  # Days of historical data to use

# Google Cloud Configuration
BUCKET_NAME = 'webscraper_bucket'  # Set your bucket name
SERVICE_ACCOUNT_KEY_PATH = ''      # Set your service account key

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Cloud NLP and Storage setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH
language_client = language_v1.LanguageServiceClient()
storage_client = storage.Client()


class GCSHandler:
    """Handles Google Cloud Storage operations"""
    
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
    
    def upload_to_gcs(self, source_file, destination_blob):
        """Uploads a file to Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(destination_blob)
            blob.upload_from_filename(source_file)
            logger.info(f"File {source_file} uploaded to {destination_blob}.")
            return True
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            return False
    
    def download_from_gcs(self, file_name, local_file=None):
        """Downloads a file from Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(file_name)
            
            if local_file:
                blob.download_to_filename(local_file)
                return True
            else:
                content = blob.download_as_text()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error downloading from GCS: {str(e)}")
            return None


class SentimentAnalyzer:
    """Handles web scraping and sentiment analysis"""
    
    def __init__(self, gcs_handler=None):
        self.url = "https://finviz.com/news.ashx?v=3"
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.gcs_handler = gcs_handler
    
    def scrape_finviz_news(self):
        """Scrape news headlines and ticker symbols from Finviz"""
        try:
            response = requests.get(self.url, headers=self.headers)
            soup = BeautifulSoup(response.text, "html.parser")
        
            # Extract headlines and ticker symbols
            news_data = []
            news_rows = soup.find_all("td", class_="news_link-cell")
        
            for row in news_rows:
            # Extract headline
                headline_tag = row.find("a", class_="nn-tab-link")
                headline = headline_tag.get_text().strip() if headline_tag else None
            
            # Extract ticker symbols - look for any ticker labels
            # Note: We're using a more general selector to catch different states (negative, positive, neutral, etc.)
                ticker_tags = row.find_all("a", class_=lambda c: c and "fv-label stock-news-label" in c)
                tickers = [tag.find("span").get_text().strip() for tag in ticker_tags if tag.find("span")]
            
            # Store the data if both headline and at least one ticker are found
                if headline and tickers:
                    for ticker in tickers:
                         news_data.append({"headline": headline, "ticker": ticker})
        
            logger.info(f"Scraped {len(news_data)} news items from Finviz")
        
        # Save to a local JSON file
            json_filename = os.path.join(DATA_DIR, "finviz_news_with_tickers.json")
            with open(json_filename, "w") as f:
                json.dump({"news": news_data}, f, indent=4)
        
        # Upload to GCS if handler is available
            if self.gcs_handler:
                self.gcs_handler.upload_to_gcs(
                    json_filename, 
                    'news/finviz_news_with_tickers.json'
                 )
                logger.info("News data uploaded to Google Cloud Storage")
        
            return news_data
        
        except Exception as e:
            logger.error(f"Error scraping Finviz news: {str(e)}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of text using Google Cloud Natural Language API"""
        try:
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            sentiment = language_client.analyze_sentiment(document=document).document_sentiment
            return {
                "score": sentiment.score,
                "magnitude": sentiment.magnitude
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"score": 0, "magnitude": 0}
    
    def process_news_sentiment(self, symbols):
        """Process news headlines and analyze sentiment for specified symbols"""
        # Get news data
        news_data = self.scrape_finviz_news()
        
        # Initialize sentiment data structure
        sentiment_data = {symbol: {
            "headlines": [],
            "sentiment_scores": [],
            "average_sentiment": 0,
            "headline_count": 0
        } for symbol in symbols}
        
        # Process each news item
        for item in news_data:
            ticker = item["ticker"]
            if ticker in symbols:
                headline = item["headline"]
                sentiment_result = self.analyze_sentiment(headline)
                
                # Store headline and sentiment
                sentiment_data[ticker]["headlines"].append(headline)
                sentiment_data[ticker]["sentiment_scores"].append(sentiment_result["score"])
                
                # Add complete sentiment data to each headline for reference
                sentiment_data[ticker]["headline_sentiments"] = sentiment_data[ticker].get("headline_sentiments", [])
                sentiment_data[ticker]["headline_sentiments"].append({
                    "headline": headline,
                    "score": sentiment_result["score"],
                    "magnitude": sentiment_result["magnitude"]
                })
        
        # Calculate average sentiment for each symbol
        for symbol in symbols:
            scores = sentiment_data[symbol]["sentiment_scores"]
            if scores:
                sentiment_data[symbol]["average_sentiment"] = sum(scores) / len(scores)
                sentiment_data[symbol]["headline_count"] = len(scores)
        
        # Save sentiment data to file
        with open(SENTIMENT_FILE, "w") as f:
            json.dump(sentiment_data, f, indent=4)
        
        # Upload to GCS if handler is available
        if self.gcs_handler:
            self.gcs_handler.upload_to_gcs(
                SENTIMENT_FILE, 
                'sentiment/sentiment_scores.json'
            )
            logger.info("Sentiment data uploaded to Google Cloud Storage")
        
        logger.info(f"Sentiment data saved to {SENTIMENT_FILE}")
        return sentiment_data


class StockPredictor:
    def __init__(self, symbols, gcs_handler=None):
        self.symbols = symbols
        self.models = {}
        self.scalers = {}
        self.gcs_handler = gcs_handler
        
    def download_stock_data(self):
        """Download historical stock data for the specified symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS_OF_HISTORY)
        
        logger.info(f"Downloading stock data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        stock_data = {}
        
        for symbol in self.symbols:
            try:
                # Use yfinance to download data
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    stock_data[symbol] = data
                    logger.info(f"Downloaded {len(data)} days of data for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {str(e)}")
        
        return stock_data
    
    def calculate_features(self, stock_data):
        """Calculate technical indicators and features for each stock"""
        features = {}
        
        for symbol, data in stock_data.items():
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Calculate basic technical indicators
            
            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            # Price changes
            df['DailyReturn'] = df['Close'].pct_change()
            df['5DayReturn'] = df['Close'].pct_change(periods=5)
            
            # Volatility
            df['Volatility'] = df['DailyReturn'].rolling(window=5).std()
            
            # Volume changes
            df['VolumeChange'] = df['Volume'].pct_change()
            
            # Relative position
            df['RSI'] = self._calculate_rsi(df['Close'], window=14)
            
            # Target: Price movement direction (1 for up, 0 for down)
            df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            
            # Drop NaN values
            df = df.dropna()
            
            features[symbol] = df
        
        return features
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate avg gains and losses over window
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def train_models(self, features_data):
        """Train a model for each stock using historical data and technical features"""
        for symbol, df in features_data.items():
            try:
                # Select features
                feature_cols = ['MA5', 'MA10', 'MA20', 'DailyReturn', '5DayReturn', 
                              'Volatility', 'VolumeChange', 'RSI']
                
                X = df[feature_cols].values
                y = df['Target'].values
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train a Random Forest model
                model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                model.fit(X_scaled, y)
                
                # Save model and scaler
                self.models[symbol] = model
                self.scalers[symbol] = scaler
                
                # Save to disk
                model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
                scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                
                # Upload to GCS if handler is available
                if self.gcs_handler:
                    self.gcs_handler.upload_to_gcs(
                        model_path, 
                        f'models/{symbol}_model.pkl'
                    )
                    self.gcs_handler.upload_to_gcs(
                        scaler_path, 
                        f'models/{symbol}_scaler.pkl'
                    )
                
                logger.info(f"Trained model for {symbol}")
                
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
    
    def predict_with_sentiment(self, features_data, sentiment_data):
        """Make predictions using models and incorporate sentiment as a bias"""
        predictions = {}
        
        for symbol, df in features_data.items():
            try:
                # Skip if model not available
                if symbol not in self.models:
                    continue
                
                # Get sentiment bias if available
                sentiment_bias = 0
                sentiment_confidence = 0
                if symbol in sentiment_data:
                    sentiment_bias = sentiment_data[symbol]["average_sentiment"]
                    headline_count = sentiment_data[symbol]["headline_count"]
                    sentiment_confidence = min(headline_count / 5, 1.0)
                
                # Get latest features
                latest_data = df.iloc[-1]
                feature_cols = ['MA5', 'MA10', 'MA20', 'DailyReturn', '5DayReturn', 
                              'Volatility', 'VolumeChange', 'RSI']
                X = latest_data[feature_cols].values.reshape(1, -1)
                
                # Scale features
                X_scaled = self.scalers[symbol].transform(X)
                
                # Get model prediction
                model = self.models[symbol]
                pred_proba = model.predict_proba(X_scaled)[0]
                
                # Adjust prediction probabilities with sentiment bias
                up_prob = pred_proba[1]
                sentiment_adjustment = sentiment_bias * sentiment_confidence * 0.2  # Scale factor
                
                adjusted_up_prob = max(0, min(1, up_prob + sentiment_adjustment))
                
                # Determine prediction based on adjusted probability
                if adjusted_up_prob > 0.6:
                    prediction = "UP"
                    confidence = adjusted_up_prob
                elif adjusted_up_prob < 0.4:
                    prediction = "DOWN"
                    confidence = 1 - adjusted_up_prob
                else:
                    prediction = "NEUTRAL"
                    confidence = 1 - abs(adjusted_up_prob - 0.5) * 2
                
                # Store prediction with details
                predictions[symbol] = {
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "model_prob": float(pred_proba[1]),
                    "adjusted_prob": float(adjusted_up_prob),
                    "sentiment_bias": float(sentiment_bias),
                    "headline_count": int(sentiment_data[symbol]["headline_count"]),
                    "headlines": sentiment_data[symbol]["headlines"],
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"{symbol}: {prediction} (Confidence: {confidence:.2f}, Sentiment Bias: {sentiment_bias:.2f})")
                
            except Exception as e:
                logger.error(f"Error making prediction for {symbol}: {str(e)}")
        
        return predictions


def save_predictions(predictions, gcs_handler=None):
    """Save predictions to a JSON file and optionally to GCS"""
    with open(PREDICTION_OUTPUT, "w") as f:
        json.dump(predictions, f, indent=4)
    
    # Upload to GCS if handler is available
    if gcs_handler:
        gcs_handler.upload_to_gcs(
            PREDICTION_OUTPUT, 
            'predictions/stock_predictions.json'
        )
        logger.info("Predictions uploaded to Google Cloud Storage")
    
    logger.info(f"Predictions saved to {PREDICTION_OUTPUT}")


def main():
    try:
        # Define stocks to track
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
        
        print("\n" + "="*80)
        print("STOCK PREDICTION WITH REAL-TIME SENTIMENT ANALYSIS AND CLOUD STORAGE")
        print("="*80)
        
        # Initialize GCS handler
        gcs_handler = GCSHandler(BUCKET_NAME)
        
        # Initialize sentiment analyzer and get sentiment data
        print("\nStep 1: Scraping news and analyzing sentiment...")
        sentiment_analyzer = SentimentAnalyzer(gcs_handler)
        sentiment_data = sentiment_analyzer.process_news_sentiment(symbols)
        
        # Print loaded sentiment data
        print("\n" + "="*80)
        print("SENTIMENT ANALYSIS RESULTS")
        print("="*80)
        
        for symbol in symbols:
            if symbol in sentiment_data:
                sent_data = sentiment_data[symbol]
                print(f"\n{symbol}:")
                print(f"  Average Sentiment: {sent_data['average_sentiment']:.4f}")
                print(f"  Headlines Analyzed: {sent_data['headline_count']}")
                if sent_data['headlines']:
                    print(f"  Recent Headlines:")
                    for i, headline in enumerate(sent_data['headlines'][:3]):  # Show up to 3 headlines
                        score = sent_data["sentiment_scores"][i] if i < len(sent_data["sentiment_scores"]) else 0
                        print(f"    {i+1}. {headline} (Sentiment: {score:.4f})")
            else:
                print(f"\n{symbol}: No sentiment data available")
        
        # Initialize stock predictor
        print("\nStep 2: Downloading historical stock data...")
        predictor = StockPredictor(symbols, gcs_handler)
        stock_data = predictor.download_stock_data()
        
        # Calculate features
        print("\nStep 3: Calculating technical indicators...")
        features_data = predictor.calculate_features(stock_data)
        
        # Train models
        print("\nStep 4: Training prediction models...")
        predictor.train_models(features_data)
        
        # Make predictions with sentiment bias
        print("\nStep 5: Making predictions with sentiment bias...")
        predictions = predictor.predict_with_sentiment(features_data, sentiment_data)
        
        # Print detailed prediction results
        print("\n" + "="*80)
        print("STOCK PREDICTIONS WITH SENTIMENT BIAS")
        print("="*80)
        
        for symbol, pred in predictions.items():
            print(f"\n{symbol}:")
            print(f"  Prediction: {pred['prediction']}")
            print(f"  Confidence: {pred['confidence']:.4f}")
            print(f"  Original Model Probability (Up): {pred['model_prob']:.4f}")
            print(f"  Sentiment-Adjusted Probability (Up): {pred['adjusted_prob']:.4f}")
            print(f"  Sentiment Bias Applied: {pred['sentiment_bias']:.4f}")
            
            # Show the impact of sentiment
            impact = abs(pred['adjusted_prob'] - pred['model_prob'])
            if impact > 0.05:
                impact_str = "STRONG"
            elif impact > 0.02:
                impact_str = "MODERATE"
            else:
                impact_str = "MINIMAL"
                
            direction = "POSITIVE" if pred['sentiment_bias'] > 0 else "NEGATIVE" if pred['sentiment_bias'] < 0 else "NEUTRAL"
            print(f"  Sentiment Impact: {impact_str} {direction} (Changed prediction by {impact:.4f})")
            
            # Show headlines that influenced prediction
            if pred.get('headlines'):
                print(f"  Headlines that influenced this prediction:")
                for i, headline in enumerate(pred['headlines'][:3]):  # Show up to 3 headlines
                    print(f"    {i+1}. {headline}")
        
        # Save predictions
        save_predictions(predictions, gcs_handler)
        print(f"\nPredictions saved to {PREDICTION_OUTPUT} and uploaded to Google Cloud Storage")
        
    except Exception as e:
        logger.error(f"Error in main program: {str(e)}")
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
