import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # Load the Iris dataset
    df = pd.read_csv('data/iris.csv')  # make sure iris.csv is in the same folder
    
    # Assume the last column is the target; adjust if yours is named differently (e.g., 'species')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train the Decision Tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Output metrics
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}\n')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()
