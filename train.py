import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle as pkl

class ModelBuilding:
    def __init__(self, X_train, X_test, Y_train, Y_test, path):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.path = path
        self.model = None  # Initialize model attribute

    def build_model(self):
        self.model = xgb.XGBClassifier()  # Create the model
        self.model.fit(self.X_train, self.Y_train)  # Train the model
        self.save_artifacts()  # Save the model

    def evaluate_model(self):
        if self.model is None:
            print("Model is not trained yet!")
            return

        # Make predictions
        predictions = self.model.predict(self.X_test)

        # Log metrics with MLflow
        mlflow.log_metric("accuracy", accuracy_score(self.Y_test, predictions))
        mlflow.log_metric("roc_auc", roc_auc_score(self.Y_test, predictions))

        # Log confusion matrix and classification report
        print("Accuracy score: ", accuracy_score(self.Y_test, predictions), '\n')
        print("Classification report: \n", classification_report(self.Y_test, predictions), '\n')
        print("Confusion Matrix: \n", confusion_matrix(self.Y_test, predictions), '\n')

    def save_artifacts(self):
        if self.model is None:
            print("No model to save!")
            return
        
        # Save the model to MLflow
        with mlflow.start_run():  # Track the experiment
            mlflow.log_param("model_type", "XGBClassifier")  # Log model type as a parameter
            mlflow.sklearn.log_model(self.model, "model")  # Log the trained model to MLflow
            print(f"Model saved to MLflow")

    def build(self):
        self.build_model()  # Build the model
        self.evaluate_model()  # Evaluate the model

