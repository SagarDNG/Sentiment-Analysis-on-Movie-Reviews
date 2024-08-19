from src.preprocessing import load_data, preprocess_data, split_data
from src.feature_extraction import vectorize_text
from src.model import train_model, predict
from src.evaluate import evaluate_model

def main():
    # Load and preprocess data
    df = load_data('data/movie_reviews.csv')
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Feature extraction
    X_train_vect, X_test_vect = vectorize_text(X_train, X_test)

    # Train model
    model = train_model(X_train_vect, y_train)

    # Predict and evaluate
    y_pred = predict(model, X_test_vect)
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
