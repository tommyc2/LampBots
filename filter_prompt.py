import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

words_to_filter = ["shit","bastard","cunt", "fuck", "bitch", "rat", "cunt", "animal", "wanker", "arsehole", "ass", "arses", "ballsack", "shit", "blowjob", "damn", "crap", "hell", "hardcoresex", "lust", "orgasm", "prick", "rectum", "rimming", "semen", "sex", "spunk", "teets", "testicle", "tit", "tits", "twat", "viagra", "vulva"]

nltk.download('vader_lexicon')

def is_negative(prompt_output):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(prompt_output)
    # if score less than 0, classify as negative
    if scores['compound'] < 0:
        return True
    else:
        return False

def filter(ai_output):
    for word in words_to_filter:
        if word in ai_output.lower():
            if is_negative(ai_output.lower()):
                print("Found a bad word in the input_text:", word)
                return True
    return False
