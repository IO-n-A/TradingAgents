"""
Initial Sentiment Labeler

This script provides a basic function to apply an initial sentiment label
to news content based on simple keyword matching.
"""

import re

def get_initial_sentiment(news_content: str, positive_keywords: list, negative_keywords: list) -> str:
    """
    Applies an initial sentiment label to news content based on keywords.

    Args:
        news_content (str): The text content of the news article.
        positive_keywords (list): A list of keywords indicating positive sentiment.
        negative_keywords (list): A list of keywords indicating negative sentiment.

    Returns:
        str: The initial sentiment label ("positive", "negative", "neutral").
    """
    if not news_content or not isinstance(news_content, str):
        return "neutral"

    text_lower = news_content.lower()
    
    positive_score = 0
    negative_score = 0

    for keyword in positive_keywords:
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
            positive_score += 1
    
    for keyword in negative_keywords:
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
            negative_score += 1

    if positive_score > negative_score:
        return "positive"
    elif negative_score > positive_score:
        return "negative"
    else:
        return "neutral"

if __name__ == '__main__':
    # Example Usage
    sample_content_positive = "The company reported record profits and excellent growth."
    sample_content_negative = "Stock prices plummeted after the disastrous earnings call."
    sample_content_neutral = "The market remained flat today."
    
    pos_kw = ["profit", "excellent", "growth", "gains", "strong"]
    neg_kw = ["plummeted", "disastrous", "loss", "poor", "weak"]

    print(f"Content: '{sample_content_positive}'")
    print(f"Sentiment: {get_initial_sentiment(sample_content_positive, pos_kw, neg_kw)}\n")

    print(f"Content: '{sample_content_negative}'")
    print(f"Sentiment: {get_initial_sentiment(sample_content_negative, pos_kw, neg_kw)}\n")

    print(f"Content: '{sample_content_neutral}'")
    print(f"Sentiment: {get_initial_sentiment(sample_content_neutral, pos_kw, neg_kw)}\n")

    no_content = ""
    print(f"Content: '{no_content}'")
    print(f"Sentiment: {get_initial_sentiment(no_content, pos_kw, neg_kw)}\n")

    none_content = None
    # print(f"Content: '{none_content}'") # This would cause an error if not handled
    # print(f"Sentiment: {get_initial_sentiment(none_content, pos_kw, neg_kw)}\n")