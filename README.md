# spam_classifier
This is a repository for spam classifier project.

# Project desctiption
    This project builds a spam email classifier. The goal is to predict whether an email is spam or not using both numerical and text features.

# Data
    The dataset I used is called "Spam Email Detection Dataset" from kaggle. It has around 10,000 emails with features like email text and subject, number of words, links, and exclamation marks, information of sender, and urgency or money related terms. It also has a label which tells whether the email is a spam or not.

# Project Structure
    The project is organized into separate folders to keep the code clean and reusable.
    - `readers/spam_reader.py` loads the dataset from the .csv file and returns it as a dataframe.
    - `features/spam_features.py` creates the features used for training the model. It includes both the original labels from the dataset and the labels I thought would be meaningful to add.
    - `classifiers/reusable_classifier.py` defines a reusable classifier class that can train and evaluate ML models like logistic regression or random forest classifier.
    - `scripts/train_spam_classifier.py` is the main script that runs the pipeline. It loads data, adds features, trains and tests the model, and prints the accuracy score results.
    - `data/` contains the dataset file used for training and testing.

# Results and Conclusion
    I first trained the model using some of the original numeric features from the dataset to test how effective is the numerical data. The logistic regression model had an accuracy score of 0.87 and random forest classifier had a score of 0.89.

    After that, I added some of my own features (some of them were brainstormed using generative AI). Couple of them were numeric, and one different feature that I had was `spam_word_count`, which counts the words that are mostly used in spam emails from the `email_text`. 

    This one text feature itself had a huge impact on the performance of the model. After including it, both logistic regression and random forest achieved and accuracy score of 0.99.

    Overall, I did really enjoy working on this project.

