from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
from StringIO import io 
import numpy as np
import pandas as pd
import argparse
import os
import json

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--train-file', type=str, default='train-V-1.csv')
    parser.add_argument('--test-file', type=str, default='test-V-1.csv')

    args, _ = parser.parse_known_args() 
    
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features = list(train_df.columns)
    label = features.pop(-1)
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]
    
    print(features)
    
    print('Training Random forest model--')
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=True)
    model.fit(X_train, y_train)
    print()
    
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print('Model persisted at ' + model_path)
    print()
    
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)
    
    print('Model Accuracy:')
    print(test_acc)
    print('Testing report:')
    print(test_rep)

    
