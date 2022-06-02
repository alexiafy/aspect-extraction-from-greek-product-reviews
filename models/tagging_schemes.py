from enum import Enum

class TaggingScheme(Enum):
    IOB1 = 'IOB1_tagging'
    IOB2 = 'IOB2_tagging'
    BIOES = 'BIOES_tagging'
    IOB1SentimentC3 = 'IOB1_sentiment_c3'
    IOB1SentimentC5 = 'IOB1_sentiment_c5'
    IOB2SentimentC3 = 'IOB2_sentiment_c3'
    IOB2SentimentC5 = 'IOB2_sentiment_c5'