import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import logging

# Adjust path to import from backend.index first
# This assumes 'index.py' is in the 'backend' directory, and 'test_sentiment.py' is in 'backend/tests/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# CRITICAL: Patch binance.client.Client BEFORE importing from index.py
# This ensures that when index.py is imported, it uses the mocked Client.
# The 'binance.client.Client' is the actual location of the class.
# We are creating a mock object that will replace the actual Client class.
mock_binance_client = MagicMock()
# Configure the mock instance that Client() will return, specifically its ping method.
# This prevents the real ping call during Client initialization in index.py.
mock_binance_client.return_value.ping.return_value = {} 

# Apply the patch globally for this test module's scope
# Start the patch; it will be active until stop_patch_binance_client() is called.
patcher_binance_client = patch('binance.client.Client', mock_binance_client)
patcher_binance_client.start() # Start the patch

# Now import the functions from index.py. They will see the mocked Client.
from index import analyze_sentiment, fetch_news_data, get_average_sentiment, NEWSAPI_KEY, NEWS_SOURCES

# Ensure NLTK VADER lexicon is available.
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    SentimentIntensityAnalyzer() 
except LookupError:
    import nltk
    nltk.download('vader_lexicon', quiet=True) # quiet=True to reduce console noise

# Suppress other logging for tests unless specifically needed
logging.basicConfig(level=logging.CRITICAL)

class TestSentimentAnalysis(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        """Stop all module-level patches after all tests in this class have run."""
        patcher_binance_client.stop() # Stop the global patch
        super().tearDownClass()

    def test_analyze_sentiment_positive(self):
        """Test that positive text returns a score > 0."""
        text = "Crypto is soaring, great profits today! Excellent news."
        self.assertGreater(analyze_sentiment(text), 0, "Positive text should have a score > 0")

    def test_analyze_sentiment_negative(self):
        """Test that negative text returns a score < 0."""
        text = "Market crash imminent, huge losses expected. Terrible outlook."
        self.assertLess(analyze_sentiment(text), 0, "Negative text should have a score < 0")

    def test_analyze_sentiment_neutral(self):
        """Test that neutral text returns a score around 0."""
        text = "The stock market is open for trading on weekdays."
        self.assertAlmostEqual(analyze_sentiment(text), 0, delta=0.1, msg="Neutral text should have a score around 0")

    def test_analyze_sentiment_empty_string(self):
        """Test that an empty string returns a score of 0."""
        text = ""
        self.assertEqual(analyze_sentiment(text), 0, "Empty string should have a score of 0")

    @patch('index.NewsApiClient') # This mock is specific to the test method
    def test_fetch_news_data_success(self, MockNewsApiClient):
        """Test successful fetching and processing of news data."""
        mock_api_instance = MockNewsApiClient.return_value
        mock_api_instance.get_everything.return_value = {
            'status': 'ok',
            'totalResults': 2,
            'articles': [
                {'title': 'Article 1', 'content': 'Content for article 1.'},
                {'title': 'Article 2', 'content': 'Content for article 2.'},
                {'title': 'Article 3 no content', 'content': None}
            ]
        }
        
        symbol = "BTCUSDT"
        articles = fetch_news_data(symbol)
        
        keyword = symbol.replace('USDT', '').replace('BUSD', '')
        MockNewsApiClient.assert_called_once_with(api_key=NEWSAPI_KEY)
        mock_api_instance.get_everything.assert_called_once_with(
            q=keyword,
            sources=NEWS_SOURCES,
            language='en',
            sort_by='publishedAt',
            page_size=20
        )
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0], 'Content for article 1.')
        self.assertEqual(articles[1], 'Content for article 2.')

    @patch('index.NewsApiClient')
    @patch('index.logging.error') # To check if error is logged
    def test_fetch_news_data_api_error(self, mock_logging_error, MockNewsApiClient):
        """Test behavior when NewsAPI returns an error."""
        mock_api_instance = MockNewsApiClient.return_value
        mock_api_instance.get_everything.side_effect = Exception("NewsAPI is down")
        
        articles = fetch_news_data("ETHUSDT")
        
        self.assertEqual(articles, [])
        mock_logging_error.assert_called_once()
        self.assertIn("Error fetching news data for ETHUSDT: NewsAPI is down", mock_logging_error.call_args[0][0])

    @patch('index.NewsApiClient')
    def test_fetch_news_data_no_articles(self, MockNewsApiClient):
        """Test behavior when NewsAPI returns no articles."""
        mock_api_instance = MockNewsApiClient.return_value
        mock_api_instance.get_everything.return_value = {
            'status': 'ok',
            'totalResults': 0,
            'articles': []
        }
        
        articles = fetch_news_data("ADAUSDT")
        self.assertEqual(articles, [])

    @patch('index.fetch_news_data') # Mocks fetch_news_data used by get_average_sentiment
    def test_get_average_sentiment_calculates_average(self, mock_fetch_news_data_for_avg):
        """Test correct calculation of average sentiment."""
        mock_fetch_news_data_for_avg.return_value = [
            "This is great news!",
            "This is bad news.",
            "This is neutral news."
        ]
        
        # Patch analyze_sentiment used by get_average_sentiment
        with patch('index.analyze_sentiment') as mock_analyze_sentiment_for_avg:
            mock_analyze_sentiment_for_avg.side_effect = [0.7, -0.5, 0.1] # Controlled scores
            avg_sentiment = get_average_sentiment("TESTSYMBOL")
            
            mock_fetch_news_data_for_avg.assert_called_once_with("TESTSYMBOL")
            self.assertEqual(mock_analyze_sentiment_for_avg.call_count, 3)
            mock_analyze_sentiment_for_avg.assert_has_calls([
                call("This is great news!"),
                call("This is bad news."),
                call("This is neutral news.")
            ])
            self.assertAlmostEqual(avg_sentiment, (0.7 - 0.5 + 0.1) / 3)

    @patch('index.fetch_news_data')
    def test_get_average_sentiment_no_articles(self, mock_fetch_news_data_for_avg):
        """Test behavior when fetch_news_data returns no articles."""
        mock_fetch_news_data_for_avg.return_value = []
        
        avg_sentiment = get_average_sentiment("NOSYMBOL")
        
        mock_fetch_news_data_for_avg.assert_called_once_with("NOSYMBOL")
        self.assertEqual(avg_sentiment, 0.0)

    @patch('index.fetch_news_data')
    @patch('index.analyze_sentiment') # Mocks analyze_sentiment used by get_average_sentiment
    def test_get_average_sentiment_single_article(self, mock_analyze_sentiment_for_avg, mock_fetch_news_data_for_avg):
        """Test average sentiment with a single article."""
        mock_fetch_news_data_for_avg.return_value = ["Only one article here."]
        mock_analyze_sentiment_for_avg.return_value = 0.8 # Assume this score
        
        avg_sentiment = get_average_sentiment("SINGLE")
        
        mock_fetch_news_data_for_avg.assert_called_once_with("SINGLE")
        mock_analyze_sentiment_for_avg.assert_called_once_with("Only one article here.")
        self.assertEqual(avg_sentiment, 0.8)

if __name__ == '__main__':
    unittest.main()
