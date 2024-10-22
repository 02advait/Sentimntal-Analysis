# Sentimntal-Analysis

<p align="justify"> 
Developed a sentiment analysis model on the IMDB movie review dataset using deep learning techniques. 
Implemented a hybrid architecture combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM)
networks to classify reviews as positive or negative. Preprocessed the dataset by tokenizing and padding sequences,
and used word embeddings for feature representation. Trained the model on TPUs in a Kaggle notebook for efficient computation.
Achieved [mention accuracy or other metrics] on the test data, demonstrating the model’s ability to effectively capture both local
and long-term dependencies in text data.
</p>

## Table of Contents

- **Overview**
- **Technologies**
- **Code Explaination**

## A: Overview

<p align="justify">
This project focuses on building a sentiment analysis model to classify movie reviews from the IMDB dataset as either positive or negative. The dataset contains 50,000 movie reviews, with an equal number of positive and negative labels. Sentiment analysis is crucial for understanding public opinion, improving customer service, and making data-driven business decisions.
</p>
<p align="justify">
To achieve this, we implemented a hybrid deep learning architecture combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The CNN is used to capture local patterns in text (like important word sequences), while the LSTM helps to model long-term dependencies. We preprocessed the dataset by tokenizing the text and padding sequences for uniform input size. Word embeddings were used to represent the text data in a meaningful numerical format.
</p>
<p align="justify">
The model was trained using TPUs in a Kaggle notebook for faster computation, and it achieved [mention your accuracy or other metrics here]. This project demonstrates the effectiveness of deep learning in sentiment analysis tasks and showcases efficient model training and deployment.
</p>

## B:  Technologies

<p align="justify">
This project leverages a variety of technologies and libraries to implement sentiment analysis effectively. The key technologies used include:
</p>

- **Python**: The primary programming language used for development.
- **TensorFlow**: An open-source library for numerical computation that makes machine learning faster and easier, used for building and training the deep learning models.
- **Keras**: A high-level API running on top of TensorFlow, simplifying the process of building and training neural networks.
- **NumPy**: A library for numerical operations in Python, used for handling arrays and performing mathematical operations.
- **Pandas**: A data manipulation and analysis library that makes it easy to handle structured data, used for data preprocessing.
- **scikit-learn**: A library for machine learning in Python, used for model evaluation and metrics.
- **Kaggle**: A platform used for hosting the project, providing access to TPUs for efficient training of the model.

<p align="justify">
These technologies work together to facilitate data preprocessing, model training, and evaluation, enabling the successful implementation of the sentiment analysis project.
</p>

## C: Code Explaination

Here’s a breakdown of the code used in this project:

```python
import pandas as pd
import numpy as np
```
### Explanation:
- pandas: This library is used for data manipulation and analysis, especially for working with structured data in the form of tables (dataframes).<br>
- numpy: This library is used for numerical operations and handling large arrays and matrices, along with a collection of mathematical functions.


```python
from sklearn.model_selection import train_test_split
```
### Explanation:
- This imports the train_test_split function from the sklearn library. This function is used to divide the dataset into two parts: a training set for training the model and a testing set for evaluating its performance.
```python
from tensorflow.keras.models import Sequential
```
### Explanation: 
- This imports the Sequential model from Keras. A Sequential model is a linear stack of layers that allows you to build a neural network layer by layer.
```python
from tensorflow.keras.layers import Dense, Embedding, LSTM
```
### Explanation: 
- This imports the Tokenizer class, which is used to convert words in the text into integer indices (or vectors). This is essential for processing text data before feeding it into a neural network.
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
### Explanation:
- This imports the pad_sequences function, which ensures that all sequences (e.g., sentences) have the same length by adding padding (zeros) where necessary. This is important because neural networks require input data to have consistent shapes.
```python
from tensorflow.keras.layers import Input
```
### Explanation: 
- This imports the Input layer, which is used to define the shape of the input data for the neural network.

```python
data = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
```
### Explanation:
- Importing file from kaggel

```python
data['sentiment'].value_counts()
```
### Explanation:
- This code counts the number of occurrences of each unique value in the sentiment column of the DataFrame.

```python
import seaborn as sns
sns.countplot(x='sentiment', data=data)
```
<img src="https://github.com/user-attachments/assets/af907e54-dd48-47d6-91c3-3bc357f1ec05" alt="Screenshot 2024-10-22 124757" width="400">

### Explanation:
- This code creates a count plot to visualize the distribution of the sentiment column in the data DataFrame, displaying the counts of positive and negative sentiments.

```python
data.replace({'sentiment': {'positive':1,'negative':0}}, inplace=True)
```
### Explanation:
- This code replaces the values in the sentiment column of the data DataFrame, converting 'positive' to 1 and 'negative' to 0 for easier numerical processing.

```python
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
```
### Explanation:
- This code splits the data DataFrame into training and testing sets, with 20% of the data allocated for testing and 80% for training. The random_state parameter ensures that the split is reproducible.

```python
print(train_data.shape)
print(test_data.shape)
```
### Explanation:
- This code prints the dimensions (number of rows and columns) of the train_data and test_data DataFrames, allowing you to verify the sizes of the training and testing datasets.

```python
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200) # pad_sequence makes sure input length remaing constant
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)
```
### Explanation:
- The first line initializes a Tokenizer to keep track of the 5,000 most common words in the dataset.
- The second line fits the tokenizer on the review column of the train_data, preparing it to convert text to sequences.
- The third and fourth lines convert the reviews in both the training and testing sets into sequences of integers and pad them to ensure that all sequences have the same length of 200.

```python
model = Sequential()
model.add(Input(shape=(200,)))                                           
model.add(Embedding(input_dim=5000, output_dim=128))                    
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))                
model.add(Dense(1, activation="sigmoid")) 
```
### Explanation:
- The first line initializes a Sequential model to build the neural network layer by layer.
- The second line specifies the input shape for the model, indicating that each input sequence has a length of 200.
- The third line adds an Embedding layer that transforms the 5,000 most common words into 128-dimensional vectors.
- The fourth line adds an LSTM layer with 128 neurons, incorporating dropout to prevent overfitting by randomly setting 20% of the input data to zero during training.
- The fifth line adds a Dense layer with a sigmoid activation function, which outputs a single value representing the predicted sentiment.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
### Explanation:
- This code compiles the model by specifying the optimizer (adam), the loss function (binary_crossentropy for binary classification), and the evaluation metric (accuracy) to assess the model's performance during training.
```python
model.summary()
```
<img src="https://github.com/user-attachments/assets/7f5bdc0f-4138-4a37-80fe-0802d840660c" alt="Screenshot 2024-10-22 125310" width="700">

### Explanation:
- This code displays a summary of the model architecture, including the layers, output shapes, the number of parameters in each layer, and the total number of parameters in the model.
```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```
### Explanation:
- This code compiles the model by specifying the optimizer (adam), the loss function (binary_crossentropy for binary classification), and the evaluation metric (accuracy) to assess the model's performance during training.
```python
history = model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_test, Y_test))
```

![Screenshot 2024-10-22 130734](https://github.com/user-attachments/assets/b418a463-2ae2-49f0-bf40-5b834d676ab2)

### Explanation:
- This code trains the model using the training data (X_train and Y_train) for 5 epochs with a batch size of 32. It also evaluates the model on the validation data (X_test and Y_test) after each epoch and stores the training history in the history variable.
```python
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
```
![image](https://github.com/user-attachments/assets/08387e47-473c-4624-85c8-cb256978ae21)

### Explanation:
- This code evaluates the trained model on the test data (X_test and Y_test) and retrieves the loss and accuracy values. It then prints the test loss and test accuracy to provide an assessment of the model's performance on unseen data.
```python
def predict_sentiment(review):
  # tokenize and pad the review
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment
```
### Explanation:
- This function takes a movie review as input, tokenizes and pads it to ensure it matches the input shape of the model, then predicts the sentiment using the trained model. It returns "positive" if the predicted probability is greater than 0.5, otherwise it returns "negative."

```python
# example usage
new_review = "This movie was fantastic. I loved it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
```
![image](https://github.com/user-attachments/assets/6a22b8c6-0b1e-4642-b222-8184213cdd5b)

### Explanation:
- This code demonstrates how to use the predict_sentiment function. It defines a new movie review, predicts its sentiment using the function, and prints the result indicating whether the sentiment is positive or negative.
```python
# example usage
new_review = "This movie was not that good"
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
```
![Screenshot 2024-10-22 130955](https://github.com/user-attachments/assets/71132405-5250-40c6-a3d8-8147a14651d2)

### Explanation:
- This code provides an example of using the predict_sentiment function with a different movie review. It defines a new review, predicts its sentiment, and prints the outcome, indicating whether the sentiment is positive or negative.

```python
# example usage
new_review = "This movie was ok but not that good."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
```
![Screenshot 2024-10-22 131016](https://github.com/user-attachments/assets/e6ee2317-a3ed-4c42-8935-9bfdf14ff501)

### Explanation:
- This code demonstrates how to use the predict_sentiment function with another movie review. It defines a new review, predicts its sentiment, and prints the result, indicating whether the sentiment is classified as positive or negative.

