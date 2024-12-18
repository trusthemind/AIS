import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_FILE = 'data_random_forests.txt'
TEST_SIZE = 0.25
RANDOM_STATE = 42
CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesClassifier(random_state=RANDOM_STATE),
}
PARAM_GRID = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7]
}

def load_data(file_path):
    try:
        logging.info(f"Loading data from {file_path}...")
        data = np.loadtxt(file_path, delimiter=',')
        X, y = data[:, :-1], data[:, -1]
        logging.info("Data loaded successfully.")
        return X, y
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def plot_decision_boundaries(classifier, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=ax)

def train_and_evaluate(X_train, y_train, X_test, y_test, classifier_name, param_grid):
    logging.info(f"Training {classifier_name} with GridSearchCV...")
    classifier = CLASSIFIERS[classifier_name]
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logging.info(f"Best parameters for {classifier_name}: {grid_search.best_params_}")

    logging.info(f"Evaluating {classifier_name} on test set...")
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=[f"Class-{int(i)}" for i in np.unique(y_test)])
    logging.info(f"{classifier_name} Classification Report:\n{report}")
    return best_model

def main():
    X, y = load_data(INPUT_FILE)

    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class-{int(label)}", edgecolors='k')
    plt.title("Raw Data")
    plt.legend()
    plt.savefig("raw_data.png")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train 
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    for i, (name, model) in enumerate(CLASSIFIERS.items()):
        best_model = train_and_evaluate(X_train, y_train, X_test, y_test, name, PARAM_GRID)

        plot_decision_boundaries(best_model, X_train, y_train, f"{name} (Training)", axes[2 * i])
        plot_decision_boundaries(best_model, X_test, y_test, f"{name} (Test)", axes[2 * i + 1])

    plt.tight_layout()
    plt.savefig("decision_boundaries.png")
    plt.show()

if __name__ == "__main__":
    main()
