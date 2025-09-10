import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# 1. Set the path to the main 'bbc' folder
# Replace this with the actual path on your machine
base_path = 'BBC News Summary/News Articles' # e.g., "bbc", "bbc-news", "data/bbc"

# 2. Identify the categories (the folder names)
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

# 3. Initialize empty lists to store our data
texts = []
labels = []

# 4. Loop through each category folder
for category in categories:
    # Construct the path to the category folder (e.g., "bbc/business/")
    category_path = os.path.join(base_path, category)
    
    # Check if the path exists to avoid errors
    if not os.path.exists(category_path):
        print(f"Warning: Directory {category_path} not found. Skipping.")
        continue
        
    # List all the .txt files in the category folder
    # os.listdir() gets all files, we filter for .txt files
    for file_name in os.listdir(category_path):
        if file_name.endswith('.txt'):
            # Construct the full file path
            file_path = os.path.join(category_path, file_name)
            
            # Read the content of the file
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                # Append the content and its label (the category) to our lists
                texts.append(content)
                labels.append(category)
            except UnicodeDecodeError:
                # Sometimes you might need to try a different encoding
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        content = file.read()
                    texts.append(content)
                    labels.append(category)
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")

# 5. Create a DataFrame from the lists
df = pd.DataFrame({
    'text': texts,
    'category': labels
})

# 6. Explore the new DataFrame
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDistribution of categories:")
print(df['category'].value_counts())

# 7. (Optional) Shuffle the DataFrame to mix the categories
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df['category_num'] = df['category'].map({'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4})

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    #Put all to lowercase
    text = text.lower()
    
    #Remove non letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #Split
    words = text.split()

    #Remove Stopwords
    processed_text = [stemmer.stem(word) for word in words if word not in stop_words]

    #Join back
    return ' '.join(processed_text)

print('Preprocessing messages...')
df['processed_text'] = df['text'].apply(preprocess_text)



X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['category_num'], test_size=0.2 ,random_state=42, stratify=df['category_num'])

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

#If you only used Naives Bayes and NOT Logistic regression together you could use the shortcut below. But you need TFIDF to be a variable on its own for Logistic regression to use it.
"""  
from sklearn.pipeline import Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
model.fit(X_train, y_train)

"""

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

""" NAIVES BAYES MODEL """

nb_classifier = MultinomialNB()

nb_classifier.fit(X_train_tfidf, y_train)

y_pred_nb = nb_classifier.predict(X_test_tfidf)

accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("The Classification Report")
print(classification_report(y_test, y_pred_nb, target_names=['business', 'entertainment', 'politics', 'sport', 'tech']))

""" LOGISTIC REGRESSION MODEL """

lr_classifier = LogisticRegression()

lr_classifier.fit(X_train_tfidf, y_train)

y_pred_lr = lr_classifier.predict(X_test_tfidf)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("The Classification Report")
print(classification_report(y_test, y_pred_lr, target_names=['business', 'entertainment', 'politics', 'sport', 'tech']))


print(f'The accuracy of Naives Bayes Model is {accuracy_nb}')
print(f'The accuracy of Logistic Regression Model is {accuracy_lr}')

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_nb = confusion_matrix(y_test, y_pred_nb)

fig, axes = plt.subplots(1,2, figsize=(15, 8))

# Naive Bayes confusion matrix
sns.heatmap(cm_nb, annot=True,fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Naive Bayes Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['business', 'entertainment', 'politics', 'sport', 'tech'])
axes[0].set_yticklabels(['business', 'entertainment', 'politics', 'sport', 'tech'])

# Logistic Regression confusion matrix
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', ax=axes[1])
axes[1].set_title('Logistic Regression Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['business', 'entertainment', 'politics', 'sport', 'tech'])
axes[1].set_yticklabels(['business', 'entertainment', 'politics', 'sport', 'tech'])

plt.tight_layout()
plt.show()

feature_names = tfidf.get_feature_names_out()
coefficient = lr_classifier.coef_

categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

# Create a DataFrame for feature importance for each class
feature_importance_df = pd.DataFrame(coefficient.T, index= feature_names, columns= categories)

for category in categories:
    print(f"\n--- Top 10 features for: {category.upper()} ---")
    # Sort the column for this category and show top 10 (most positive coefficients)
    top_10 = feature_importance_df[category].sort_values(ascending=False).head(10)
    print(top_10)

    print(f"\n--- Bottom 10 features for: {category.upper()} ---")
    # Show bottom 10 (most negative coefficients). These are words that make the article LESS likely to be this category.
    bottom_10 = feature_importance_df[category].sort_values(ascending=True).head(10)
    print(bottom_10)