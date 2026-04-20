import pandas as pd

def num_features(df):
    #this will return all the numeric features that i decided to use for the classifier and their labels

    # new labels that i created, which i think might increase the accuracy score of the model and are relevant
    df['exclamation_to_word_ratio'] = df['num_exclamation_marks'] / (df['num_words']+1) #+1 so we dont accidentally divide by 0
    df['money_urgency_ratio'] = df['contains_money_terms'] * df['contains_urgency_terms']
    df['exclamation_money_ratio'] = df['num_exclamation_marks'] * df['contains_money_terms']
    df["spam_likely"] = df['contains_money_terms'] + df["contains_urgency_terms"] +df['num_exclamation_marks']
    #these are spam keywords that are often used in spam emails
    spam_words = ['free', 'win', 'winner', 'prize', 'money', 'urgent', 'offer', 'click', 'buy', 'cheap']
    df['spam_word_count'] = df['email_text'].str.lower().apply(lambda x: sum(word in x for word in spam_words))

    features = [
        'num_characters',
        'num_words',
        'num_exclamation_marks',
        "num_links",
        'has_suspicious_link',
        'num_attachments',
        'sender_reputation_score',
        'num_recipients',
        'contains_money_terms',
        'contains_urgency_terms',
        'exclamation_to_word_ratio',
        'money_urgency_ratio',
        'exclamation_money_ratio',
        'spam_likely',
        'spam_word_count'
    ]

    features = df[features]
    labels = df['label']

    return features, labels