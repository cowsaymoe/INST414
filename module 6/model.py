import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

def main():
    # Load the data
    data = pd.read_csv('engineered_posts.csv')

    # Convert non-numeric columns to numeric using LabelEncoder
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Define features and target
    X = data.drop(columns=['has_question', 'sentiment', 'hour_sin', 'hour_cos'])
    y = data['final_score']

    # Convert continuous target to binary target
    threshold = y.median()
    y = (y > threshold).astype(int)

    # Initialize the model
    model = RandomForestClassifier()

    # Perform stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5)
    y_pred = cross_val_predict(model, X, y, cv=skf)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    # Output the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Identify and print 5 examples it predicted wrong
    wrong_predictions = X[y != y_pred]
    wrong_predictions['true_label'] = y[y != y_pred]
    wrong_predictions['predicted_label'] = y_pred[y != y_pred]
    print("\n5 Examples of Wrong Predictions:")
    print(wrong_predictions.head(5))

if __name__ == '__main__':
    main()
