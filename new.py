"""
from collections import deque

# Function to perform BFS traversal
def bfs(graph, start):
    visited = set()           # To keep track of visited nodes
    queue = deque([start])    # Queue for BFS
    traversal_order = []      # To store traversal order
    
    while queue:
        node = queue.popleft()   # Dequeue a vertex
        if node not in visited:
            visited.add(node)
            traversal_order.append(node)
            
            # Add all unvisited neighbors to queue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return traversal_order


# Example graph (Adjacency List)
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Run BFS from node 'A'
print("BFS Traversal starting from A:", bfs(graph, 'A'))



from collections import deque

def water_jug_bfs(jug1, jug2, target):
    # Queue stores states as tuples (amount_in_jug1, amount_in_jug2)
    queue = deque()
    visited = set()

    # Start state: both jugs empty
    queue.append((0, 0))
    visited.add((0, 0))

    while queue:
        x, y = queue.popleft()
        print(f"Jug1: {x}, Jug2: {y}")  # Printing states

        # If we reach target in either jug
        if x == target or y == target:
            print("Reached the target!")
            return True

        # Possible next states:
        next_states = [
            (jug1, y),    # Fill Jug1
            (x, jug2),    # Fill Jug2
            (0, y),       # Empty Jug1
            (x, 0),       # Empty Jug2
            # Pour Jug1 -> Jug2
            (x - min(x, jug2 - y), y + min(x, jug2 - y)),
            # Pour Jug2 -> Jug1
            (x + min(y, jug1 - x), y - min(y, jug1 - x))
        ]

        for state in next_states:
            if state not in visited:
                visited.add(state)
                queue.append(state)

    print("Target cannot be reached.")
    return False

# Example usage:
jug1_capacity = 4
jug2_capacity = 3
target_amount = 2

water_jug_bfs(jug1_capacity, jug2_capacity, target_amount)




import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (uncomment if running for the first time)
#nltk.download('punkt')

#nltk.download('stopwords')

with open('input.txt', 'r') as file:
    text = file.read()

stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
filtered_words = [word for word in words if word.lower() not in stop_words]
filtered_text = ' '.join(filtered_words)

with open('output.txt', 'w') as file:
    file.write(filtered_text)

print("Original passage:\n", text)
print("\nAfter removing stop words:\n", filtered_text)


import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download resources (uncomment these if running for the first time)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Input sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
words = word_tokenize(sentence)

# Perform POS tagging
pos_tags = pos_tag(words)

# Display results

print("Input Sentence:")
print(sentence)

print("\nPOS Tagging Result:")
print(pos_tags)


# Lemmatization using NLTK 
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet 
# Download required datasets (only first time) 
#nltk.download('wordnet') 
#nltk.download('omw-1.4') 
# Create WordNetLemmatizer object 
lemmatizer = WordNetLemmatizer() 
# Sample words 
words = ["running", "flies", "better", "studies", "children", "feet"] 
print("Original Word -> Lemmatized Word") 
for word in words: 
    print(f"{word} -> {lemmatizer.lemmatize(word)}") 
# Lemmatization with Part of Speech (POS) 
print("\nLemmatization with POS tags:") 
print("running (verb) ->", lemmatizer.lemmatize("running", pos="v")) 
print("better (adjective) ->", lemmatizer.lemmatize("better", pos="a")) 



# Text Classification using NLTK

import nltk
from nltk.classify import NaiveBayesClassifier

# Training dataset (sentence, label)
training_data = [
    ("I love this product", "Positive"),
    ("This is an amazing place", "Positive"),
    ("I feel great about the things", "Positive"),
    ("This is my best experience", "Positive"),
    ("I do not like this product", "Negative"),
    ("This is the worst thing ever", "Negative"),
    ("I feel bad about it", "Negative"),
    ("This is a terrible experience", "Negative"),
]

# Feature extractor
def extract_features(words):
    return {word: True for word in words.split()}

# Prepare training set
training_features = [(extract_features(text), label) for (text, label) in training_data]

# Train Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(training_features)

# Test sentence
test_sentence = "I love this amazing product"
test_features = extract_features(test_sentence)

# Classification
print("Test Sentence:", test_sentence)
print("Classification:", classifier.classify(test_features))





# Simple Linear Regression Demonstration

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset (Hours studied vs. Marks obtained)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)   # Independent variable
y = np.array([2, 4, 5, 4, 5])                  # Dependent variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Print slope and intercept
print("Slope (b):", model.coef_[0])
print("Intercept (a):", model.intercept_)

# Test prediction
print("Predicted marks for 6 hours of study:", model.predict([[6]])[0])

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()


"""

# k-Nearest Neighbor Classification on Iris Dataset 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 

# Load Iris dataset 
iris = load_iris() 
X = iris.data      # Features 
y = iris.target    # Labels 

# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.3, random_state=42 
) 

# Create KNN model with k=3 

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) 

# Predict 

y_pred = knn.predict(X_test) 

# Accuracy 

print("Accuracy:", accuracy_score(y_test, y_pred)) 

# Test with a new flower sample 

sample = [[5.1, 3.5, 1.4, 0.2]]  # Sepal length, Sepal width, Petal length, Petal width 
prediction = knn.predict(sample) 
print("Prediction for sample flower:", iris.target_names[prediction][0])





"""
# Naïve Bayes Classifier Demonstration

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Sample dataset (Text messages with labels)

texts = [
    "I love this phone", 
    "This is an amazing movie",
    "I feel great today", 
    "This product is good",
    "I hate this phone", 
    "This is a terrible movie",
    "I feel bad today", 
    "This product is awful"
]

labels = [
    "Positive", "Positive", "Positive", "Positive",
    "Negative", "Negative", "Negative", "Negative"
]


# Convert text to feature vectors

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)


# Train Naïve Bayes Model

nb = MultinomialNB()
nb.fit(X_train, y_train)


# Predictions

y_pred = nb.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


# Test with a new sentence

test_sentence = ["I love this amazing product"]
test_vector = vectorizer.transform(test_sentence)
prediction = nb.predict(test_vector)

print("Prediction for test sentence:", prediction[0])
"""