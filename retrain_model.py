# retrain_model.py

#import joblib
#from sklearn.svm import SVC
#from sklearn.datasets import load_iris  # Just for testing

# 1. Load sample data (replace with your actual data and preprocessing!)
#X, y = load_iris(return_X_y=True)

# 2. Create and train the model
#model = SVC()
#model.fit(X, y)

# 3. Save the model using joblib
#joblib.dump(model, 'model.pkl')

#print("✅ Model retrained and saved as model.pkl using scikit-learn 1.6.1")
from sklearn.svm import SVC
import joblib

# Train your SVC model with probability=True
svc = SVC(probability=True)
# Example: Load a dataset and split it into training and testing sets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load sample data
X, y = load_iris(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
svc.fit(X_train, y_train)

# Save the model again
joblib.dump(svc, 'svc.pkl')  # Use this new file in your Flask app
