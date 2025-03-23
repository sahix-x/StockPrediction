import pandas as pd
import numpy as np
import json
import os
import matplotlib
# Force matplotlib to use a non-interactive backend
matplotlib.use('Agg')  # This must come before any other matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf

# Configuration
DATA_DIR = "data"
MODEL_DIR = "models"
PREDICTION_FILE = os.path.join(DATA_DIR, "stock_predictions.json")
SENTIMENT_FILE = os.path.join(DATA_DIR, "sentiment_scores.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "visualizations")
DAYS_OF_HISTORY = 60  # Match the number of days used in the prediction model

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_prediction_data(file_path):
    """Load prediction data from JSON file"""
    try:
        with open(file_path, "r") as f:
            predictions = json.load(f)
        return predictions
    except Exception as e:
        print(f"Error loading prediction data: {str(e)}")
        return None

def load_sentiment_data(file_path):
    """Load sentiment data from JSON file"""
    try:
        with open(file_path, "r") as f:
            sentiment = json.load(f)
        return sentiment
    except Exception as e:
        print(f"Error loading sentiment data: {str(e)}")
        return None

def download_stock_data(symbols, days=DAYS_OF_HISTORY):
    """Download historical stock data for the specified symbols"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Downloading stock data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    stock_data = {}
    
    for symbol in symbols:
        try:
            # Use yfinance to download data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not data.empty:
                stock_data[symbol] = data
                print(f"Downloaded {len(data)} days of data for {symbol}")
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")
    
    return stock_data

def create_simple_stock_chart(symbol, stock_data, predictions, sentiment_data=None):
    """Create a simplified chart for a single stock showing price data and prediction"""
    if symbol not in stock_data or symbol not in predictions:
        print(f"Missing data for {symbol}. Skipping chart creation.")
        return None
    
    # Get data
    df = stock_data[symbol]
    pred = predictions[symbol]
    
    # Calculate moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot price and moving averages
    plt.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5)
    plt.plot(df.index, df['MA5'], label='5-Day MA', color='blue', alpha=0.7)
    plt.plot(df.index, df['MA20'], label='20-Day MA', color='green', alpha=0.7)
    
    # Add prediction information
    last_date = df.index[-1]
    last_price = df['Close'].iloc[-1]
    
    # Create a vertical line at the last date
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
    
    # Add prediction label
    pred_color = 'green' if pred['prediction'] == 'UP' else 'red' if pred['prediction'] == 'DOWN' else 'gray'
    plt.text(last_date, last_price * 1.05, 
             f" Prediction: {pred['prediction']}\n Confidence: {pred['confidence']:.2f}\n Model: {pred['model_prob']:.2f}\n With Sentiment: {pred['adjusted_prob']:.2f}", 
             fontsize=9, color=pred_color, 
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add sentiment information if available
    if sentiment_data and symbol in sentiment_data:
        sent = sentiment_data[symbol]
        sentiment_color = 'green' if sent['average_sentiment'] > 0 else 'red' if sent['average_sentiment'] < 0 else 'gray'
        sentiment_text = f"Sentiment: {sent['average_sentiment']:.2f} ({sent['headline_count']} headlines)"
        plt.text(df.index[0], last_price * 1.05, sentiment_text, 
                fontsize=9, color=sentiment_color,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Configure axis
    plt.title(f"{symbol} Stock Price and Prediction", fontsize=16)
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Format dates on x-axis
    plt.gcf().autofmt_xdate()
    date_format = mdates.DateFormatter('%Y-%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, f"{symbol}_prediction_simple.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved simple prediction chart for {symbol} to {output_file}")
    return output_file

def create_model_vs_sentiment_chart(predictions):
    """Create a bar chart comparing model predictions with and without sentiment"""
    # Extract data
    symbols = []
    model_probs = []
    adjusted_probs = []
    prediction_types = []
    
    for symbol, data in predictions.items():
        try:
            symbols.append(symbol)
            model_probs.append(data["model_prob"] * 100)  # Convert to percentage
            adjusted_probs.append(data["adjusted_prob"] * 100)  # Convert to percentage
            prediction_types.append(data["prediction"])
        except KeyError as e:
            print(f"Missing key in prediction data for {symbol}: {e}")
    
    if not symbols:
        print("No valid prediction data for visualization")
        return False
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(symbols))
    
    # Create bars
    plt.bar(index, model_probs, bar_width, label='Model Only', color='royalblue')
    plt.bar(index + bar_width, adjusted_probs, bar_width, label='With Sentiment', color='mediumseagreen')
    
    # Add prediction symbols
    for i, pred in enumerate(prediction_types):
        marker = '▲' if pred == 'UP' else '▼' if pred == 'DOWN' else '◆'
        color = 'green' if pred == 'UP' else 'red' if pred == 'DOWN' else 'gray'
        plt.text(i + bar_width/2, max(model_probs[i], adjusted_probs[i]) + 3, 
                marker, color=color, ha='center', va='bottom', fontsize=14)
    
    # Add labels and title
    plt.xlabel('Stock Symbol')
    plt.ylabel('Upward Movement Probability (%)')
    plt.title('Model Probability vs. Sentiment-Adjusted Probability')
    plt.xticks(index + bar_width/2, symbols)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, "model_vs_sentiment.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved model vs sentiment chart to {output_file}")
    return output_file

def create_sentiment_impact_chart(predictions):
    """Create a chart showing the impact of sentiment on predictions"""
    # Extract data
    symbols = []
    impacts = []
    prediction_types = []
    
    for symbol, data in predictions.items():
        try:
            impact = data["adjusted_prob"] - data["model_prob"]
            symbols.append(symbol)
            impacts.append(impact * 100)  # Convert to percentage
            prediction_types.append(data["prediction"])
        except KeyError:
            continue
    
    if not symbols:
        return None
    
    # Sort by impact
    sorted_indices = np.argsort(impacts)
    symbols = [symbols[i] for i in sorted_indices]
    impacts = [impacts[i] for i in sorted_indices]
    prediction_types = [prediction_types[i] for i in sorted_indices]
    
    # Define colors based on prediction
    colors = ['green' if pred == 'UP' else 'red' if pred == 'DOWN' else 'gray' 
             for pred in prediction_types]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bars
    bars = plt.bar(symbols, impacts, color=colors)
    
    # Add labels to bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        sign = '+' if height > 0 else ''
        plt.text(bar.get_x() + bar.get_width()/2, 
                height + 0.3 if height >= 0 else height - 0.7,
                f"{sign}{height:.1f}%", ha='center', va='bottom')
    
    # Add labels and title
    plt.xlabel('Stock Symbol')
    plt.ylabel('Sentiment Impact (Change in Probability %)')
    plt.title('Impact of Sentiment Analysis on Prediction Probabilities')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, "sentiment_impact.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved sentiment impact chart to {output_file}")
    return output_file

def create_prediction_summary(predictions, sentiment_data=None):
    """Create a summary visualization of all predictions"""
    # Set up the figure
    plt.figure(figsize=(14, len(predictions) * 0.6 + 2))
    
    # Extract data for each stock
    stocks = []
    probabilities = []
    sentiments = []
    pred_types = []
    
    for i, (symbol, data) in enumerate(predictions.items()):
        stocks.append(symbol)
        probabilities.append(data['adjusted_prob'] * 100)  # Convert to percentage
        sentiments.append(data.get('sentiment_bias', 0))
        pred_types.append(data['prediction'])
    
    # Sort by prediction type and then by probability
    sort_order = {'UP': 0, 'NEUTRAL': 1, 'DOWN': 2}
    sorted_indices = sorted(range(len(stocks)), 
                           key=lambda i: (sort_order[pred_types[i]], -probabilities[i]))
    
    stocks = [stocks[i] for i in sorted_indices]
    probabilities = [probabilities[i] for i in sorted_indices]
    sentiments = [sentiments[i] for i in sorted_indices]
    pred_types = [pred_types[i] for i in sorted_indices]
    
    # Define colors based on prediction
    colors = {'UP': 'green', 'NEUTRAL': 'gray', 'DOWN': 'red'}
    
    # Create horizontal bars for probability
    y_pos = np.arange(len(stocks))
    bars = plt.barh(y_pos, probabilities, color=[colors[p] for p in pred_types], alpha=0.6)
    
    # Add sentiment indicators
    for i, (sentiment, prob) in enumerate(zip(sentiments, probabilities)):
        if sentiment > 0:
            marker = '+'
            marker_color = 'green'
        elif sentiment < 0:
            marker = '-'
            marker_color = 'red'
        else:
            marker = 'o'
            marker_color = 'gray'
        
        # Add sentiment marker
        plt.text(prob + 1, i, marker * min(3, abs(int(sentiment * 10))), 
                color=marker_color, va='center', fontweight='bold')
    
    # Add stock labels
    plt.yticks(y_pos, stocks)
    
    # Add labels and title
    plt.xlabel('Upward Movement Probability (%)')
    plt.title('Stock Prediction Summary with Sentiment Influence')
    
    # Add a vertical line at 50%
    plt.axvline(x=50, color='black', linestyle='--', alpha=0.5)
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Set x-axis limits
    plt.xlim(0, 105)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['UP'], alpha=0.6, label='UP'),
        Patch(facecolor=colors['NEUTRAL'], alpha=0.6, label='NEUTRAL'),
        Patch(facecolor=colors['DOWN'], alpha=0.6, label='DOWN')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add explanation of sentiment markers
    plt.figtext(0.5, 0.01, 
                "+ = Positive sentiment bias  |  - = Negative sentiment bias  |  o = Neutral",
                ha='center', fontsize=9)
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, "prediction_summary.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved prediction summary to {output_file}")
    return output_file

def create_headline_sentiment_summary(sentiment_data):
    """Create a summary of headline sentiment for each stock"""
    if not sentiment_data:
        return None
    
    # Extract data
    symbols = []
    avg_sentiments = []
    headline_counts = []
    
    for symbol, data in sentiment_data.items():
        if 'average_sentiment' in data and 'headline_count' in data:
            symbols.append(symbol)
            avg_sentiments.append(data['average_sentiment'])
            headline_counts.append(data['headline_count'])
    
    if not symbols:
        return None
    
    # Sort by sentiment value
    sorted_indices = np.argsort(avg_sentiments)
    symbols = [symbols[i] for i in sorted_indices]
    avg_sentiments = [avg_sentiments[i] for i in sorted_indices]
    headline_counts = [headline_counts[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create colors based on sentiment
    colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in avg_sentiments]
    
    # Create bars
    bars = plt.barh(symbols, avg_sentiments, color=colors, alpha=0.7)
    
    # Add headline counts
    for i, (bar, count) in enumerate(zip(bars, headline_counts)):
        plt.text(bar.get_width() + 0.01 if bar.get_width() >= 0 else bar.get_width() - 0.05,
                bar.get_y() + bar.get_height()/2,
                f"{count} headlines", va='center', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Average Sentiment Score')
    plt.title('News Headline Sentiment by Stock')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add a vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, "headline_sentiment.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved headline sentiment summary to {output_file}")
    return output_file

def main():
    print("\n" + "="*80)
    print("SIMPLIFIED STOCK PREDICTION VISUALIZATION")
    print("="*80)
    
    # Load prediction and sentiment data
    print("\nLoading prediction and sentiment data...")
    predictions = load_prediction_data(PREDICTION_FILE)
    sentiment_data = load_sentiment_data(SENTIMENT_FILE)
    
    if not predictions:
        print("Error loading prediction data. Exiting.")
        return
    
    # Get list of symbols
    symbols = list(predictions.keys())
    print(f"Loaded predictions for {len(symbols)} stocks: {', '.join(symbols)}")
    
    # Create basic comparison charts
    print("\nGenerating comparison visualizations...")
    create_model_vs_sentiment_chart(predictions)
    create_sentiment_impact_chart(predictions)
    create_prediction_summary(predictions, sentiment_data)
    
    # Create sentiment summary if available
    if sentiment_data:
        print("\nGenerating sentiment summary...")
        create_headline_sentiment_summary(sentiment_data)
    
    try:
        # Download historical stock data
        print("\nDownloading historical stock data...")
        stock_data = download_stock_data(symbols)
        
        if stock_data:
            # Create individual stock charts
            print("\nGenerating individual stock charts...")
            for symbol in symbols:
                try:
                    create_simple_stock_chart(symbol, stock_data, predictions, sentiment_data)
                except Exception as e:
                    print(f"Error creating chart for {symbol}: {str(e)}")
        else:
            print("No stock data downloaded. Skipping individual stock charts.")
    except Exception as e:
        print(f"Error in historical data processing: {str(e)}")
    
    print("\n" + "="*80)
    print(f"All visualizations have been saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
