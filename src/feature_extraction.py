from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    return X_train_vect, X_test_vect
