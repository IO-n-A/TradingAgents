# core/preprocess/text_preprocessor.py
"""
Cleans and normalizes raw news text (headlines/summaries) for further processing
like sentiment analysis.
"""
import re
import string
import logging
from typing import List, Optional

# Attempt to import nltk, required for stopword removal.
# If NLTK is not available, a basic stopword list will be used.
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True) # Ensure stopwords are downloaded
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Basic English stopwords if NLTK is not available
    BASIC_STOPWORDS = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# This class provides methods to preprocess text data.
# It includes functionalities like lowercasing, removing punctuation, and removing stopwords.
# The main method `process_text` applies these steps sequentially.
class TextPreprocessor:
    """
    A class for cleaning and normalizing raw news text.

    This preprocessor handles tasks such as lowercasing, removing punctuation,
    and removing stopwords to prepare text for sentiment analysis or other
    NLP tasks. It can use NLTK for stopwords if available, otherwise, it falls
    back to a basic list.
    """

    def __init__(self, language: str = 'english'):
        """
        Initializes the TextPreprocessor.

        Args:
            language (str): The language for stopword removal. Defaults to 'english'.
                            Currently, only 'english' is robustly supported.
        """
        self.language = language
        self.stopwords_set = set()
        if NLTK_AVAILABLE:
            try:
                self.stopwords_set = set(stopwords.words(self.language))
                logger.info(f"NLTK stopwords for '{self.language}' loaded successfully.")
            except OSError: # Handle cases where the specific language stopwords might be missing
                logger.warning(
                    f"NLTK stopwords for '{self.language}' not found. "
                    f"Falling back to basic English stopwords if language is 'english'."
                )
                if self.language == 'english':
                    self.stopwords_set = BASIC_STOPWORDS
                else:
                    logger.error(f"No basic stopwords available for language: {self.language}. Stopword removal will be ineffective.")
        elif self.language == 'english':
            self.stopwords_set = BASIC_STOPWORDS
            logger.info("NLTK not available. Using basic English stopwords.")
        else:
            logger.warning(
                f"NLTK not available and no basic stopwords for language: {self.language}. "
                "Stopword removal will be ineffective."
            )
        # This constructor initializes the TextPreprocessor.
        # It sets up the stopwords list based on NLTK availability and the specified language.
        print(f"TextPreprocessor initialized for {self.language}. Stopwords loaded: {len(self.stopwords_set) > 0}")


    # This method converts text to lowercase.
    # It takes a string as input.
    # It returns the lowercased string.
    def _lowercase(self, text: str) -> str:
        """Converts text to lowercase."""
        processed_text = text.lower()
        # The text has been converted to lowercase.
        # This step helps in standardizing the text for further processing.
        # print(f"Text lowercased: '{text}' -> '{processed_text}'") # Too verbose for production
        return processed_text

    # This method removes punctuation from text.
    # It takes a string as input.
    # It returns the string with punctuation removed.
    def _remove_punctuation(self, text: str) -> str:
        """Removes punctuation from text."""
        translator = str.maketrans('', '', string.punctuation)
        processed_text = text.translate(translator)
        # Punctuation has been removed from the text.
        # This helps in cleaning the text by removing characters that might not be useful for analysis.
        # print(f"Punctuation removed: '{text}' -> '{processed_text}'") # Too verbose for production
        return processed_text

    # This method removes stopwords from text.
    # It takes a string as input.
    # It returns the string with stopwords removed.
    def _remove_stopwords(self, text: str) -> str:
        """Removes stopwords from text."""
        if not self.stopwords_set:
            logger.warning("Stopwords set is empty. Skipping stopword removal.")
            return text
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords_set]
        processed_text = " ".join(filtered_words)
        # Stopwords have been removed from the text.
        # This step reduces noise by eliminating common words that usually don't carry significant meaning.
        # print(f"Stopwords removed: '{text}' -> '{processed_text}'") # Too verbose for production
        return processed_text

    # This method removes URLs from text.
    # It takes a string as input.
    # It returns the string with URLs removed.
    def _remove_urls(self, text: str) -> str:
        """Removes URLs from text."""
        processed_text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # URLs have been removed from the text.
        # This step cleans the text by removing web links which are generally not needed for sentiment analysis.
        # print(f"URLs removed: '{text}' -> '{processed_text}'") # Too verbose for production
        return processed_text

    # This method removes HTML tags from text.
    # It takes a string as input.
    # It returns the string with HTML tags removed.
    def _remove_html(self, text: str) -> str:
        """Removes HTML tags from text."""
        processed_text = re.sub(r'<.*?>', '', text)
        # HTML tags have been removed from the text.
        # This step cleans the text by removing markup language tags.
        # print(f"HTML removed: '{text}' -> '{processed_text}'") # Too verbose for production
        return processed_text

    # This method processes a single text string by applying all preprocessing steps.
    # It takes a raw text string as input.
    # It returns the cleaned and normalized text string.
    def process_text(self, text: str) -> str:
        """
        Processes a single text string by applying all preprocessing steps.

        The steps include: removing HTML, removing URLs, lowercasing,
        removing punctuation, and removing stopwords.

        Args:
            text (str): The raw text string to process.

        Returns:
            str: The cleaned and normalized text string.
        """
        if not isinstance(text, str):
            logger.warning(f"Input is not a string (type: {type(text)}). Returning as is.")
            return str(text) # Attempt to convert to string, or handle as per requirements

        logger.debug(f"Original text: '{text[:100]}...'") # Log snippet
        processed_text = self._remove_html(text)
        processed_text = self._remove_urls(processed_text)
        processed_text = self._lowercase(processed_text)
        processed_text = self._remove_punctuation(processed_text)
        processed_text = self._remove_stopwords(processed_text)
        # The input text has been fully processed through all cleaning and normalization steps.
        # The output is a refined version of the text, ready for subsequent analysis tasks.
        print(f"Text processed. Original length: {len(text)}, Processed length: {len(processed_text)}")
        logger.debug(f"Processed text: '{processed_text[:100]}...'")
        return processed_text

    # This method processes a list of text strings.
    # It takes a list of raw text strings as input.
    # It returns a list of cleaned and normalized text strings.
    def process_texts(self, texts: List[str]) -> List[str]:
        """
        Processes a list of text strings.

        Applies the `process_text` method to each string in the input list.

        Args:
            texts (List[str]): A list of raw text strings.

        Returns:
            List[str]: A list of cleaned and normalized text strings.
        """
        processed_texts = [self.process_text(text) for text in texts]
        # A list of texts has been processed.
        # Each text in the list has undergone the standard cleaning and normalization pipeline.
        print(f"Processed {len(texts)} texts.")
        return processed_texts


if __name__ == '__main__':
    # Example Usage:
    preprocessor = TextPreprocessor(language='english')

    sample_text_html = "<p>This is an <b>example</b> sentence! Check out http://example.com for more.</p>"
    sample_text_simple = "Another EXCITING news headline about $STOCK and markets."
    sample_texts_list = [
        "Markets are UP today due to positive news!",
        "Analysts predict a DOWNTURN. More at www.news.com/article123",
        "This is just a neutral statement.",
        "<h1>Big News!</h1><br>The company announced record profits.",
        None, # Test with None
        12345 # Test with non-string
    ]

    logger.info("Processing sample HTML text...")
    processed_sample_html = preprocessor.process_text(sample_text_html)
    logger.info(f"Original HTML: '{sample_text_html}'")
    logger.info(f"Processed HTML: '{processed_sample_html}'")
    print("-" * 20)

    logger.info("Processing simple sample text...")
    processed_simple_text = preprocessor.process_text(sample_text_simple)
    logger.info(f"Original Simple: '{sample_text_simple}'")
    logger.info(f"Processed Simple: '{processed_simple_text}'")
    print("-" * 20)

    logger.info("Processing list of sample texts...")
    processed_texts_list = preprocessor.process_texts(sample_texts_list)
    for original, processed in zip(sample_texts_list, processed_texts_list):
        logger.info(f"Original: '{original}'\nProcessed: '{processed}'\n---")

    # Example with a different language (if NLTK supports it, otherwise will use basic or none)
    # preprocessor_es = TextPreprocessor(language='spanish')
    # spanish_text = "Esto es una frase de ejemplo en español."
    # processed_spanish = preprocessor_es.process_text(spanish_text)
    # logger.info(f"Original Spanish: '{spanish_text}'")
    # logger.info(f"Processed Spanish: '{processed_spanish}'")