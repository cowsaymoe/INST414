import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Load the dataset
    data = pd.read_csv('data.csv')
    

    # Preprocess the data
    data['Gross'] = data['Gross'].str.replace(',', '').astype(float)
    data['Meta_score'] = data['Meta_score'].astype(float)
    data['Meta_score'].fillna(data['Meta_score'].mean(), inplace=True)
    data['Blockbuster'] = data['Gross'] > data['Gross'].median()
    # Select features and target
    features = ['Meta_score', 'No_of_Votes', 'Released_Year']
    X = data[features]
    y = data['Blockbuster']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.head())
    # Train and evaluate multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {accuracy:.2f}')

    # Classify a new movie (example) using Random Forest
    new_movie = pd.DataFrame({
        'Meta_score': [8.5],
        'No_of_Votes': [1000000],
        'Released_Year': [2022]
    })
    prediction = models['Random Forest'].predict(new_movie)
    print('Blockbuster' if prediction[0] else 'Non-blockbuster')

    
if __name__ == '__main__':
    main()