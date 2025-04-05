# Import necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
# Example dataset - replace 'reviews.csv' with your dataset file
data = {
    'Review Text': [
        "Amazing product! Highly recommended.",
        "Terrible service, I'm very disappointed.",
        "Great quality, would buy again.",
        "Waste of money. Awful experience.",
        "Exceeded my expectations. Fantastic!"
    ],
    'Sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
}
df = pd.DataFrame(data)

# Step 2: Preprocess the text data
def preprocess_text(text):
    # Remove punctuation, special characters, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

df['Cleaned Review'] = df['Review Text'].apply(preprocess_text)

print("First few rows of the dataset:")
print(df.head())

# Step 3: Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['Cleaned Review'])
y = df['Sentiment']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Bonus: Predict sentiment for a new review
new_review = "Absolutely love this product! Excellent quality."
cleaned_review = preprocess_text(new_review)
vectorized_review = vectorizer.transform([cleaned_review])
prediction = model.predict(vectorized_review)
print(f"\nSentiment Prediction for review: '{new_review}' -> {prediction[0]}")