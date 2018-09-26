DEPENDENCIES/REQUIREMENTS

Keras
Sklearn
NLTK
GLOVE PRETRAINED MODEL (https://nlp.stanford.edu/projects/glove/)

The dataset ag_news is been trained on various model(Deep neural network, Random Forest, SVM) and is generalised to all of them.
This code uses Glove embeddings to represent each document in vector space.
Getting highest accuracy with Deep neural network 91%.
90% accuracy with SVM, 85% accuracy with Random Forest

HOW TO RUN CODE?
After installing all the dependencies, call model class from Model.py and pass which model you want the corpus to train on.
for eg. model_obj = model("neural_networks"), model("random_forrest"), model("SVM")


from that object call fit_predict method  and pass the train and test file and run the file.
for train and test file change the path of files from file_path.py

TEXT PREPROCESSING-
For cleaning the text I have used regular expression to remove all puntuation marks, convert all words to lowercase,remove stopwords and
use WordNetLemmatizer to lemmatize the words in which it just keeps the root meaning of the word.
for eg. "Multiplying" is converted to "Multiply"

EMBEDDINGS-
We can train our own Glove model by co-occurence matrix which uses N-grams principle, Matrix-Factorization but
it need a large amount of training data.
Load glove pretrained file in which there are 1.2M word embeddings.
choose only those words which are in corpus rest add 0 vector for that word.
converting the cleaned articles in vectors from Tfidf.
shape of article_matrix = (12000,vocab_of_corpus)
shape of glove embeddings = (vocab_of_corpus,300)
dot producting will give us the embeddings for 12000 articles.

LEARNING MODEL-
Pass the embeddings for train and test data in classifier and it will provide the predictions
pass predictions to accuracy_score method to get the accuracy

