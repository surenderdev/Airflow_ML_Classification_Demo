from airflow.models import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator

import pandas as pd
import os
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, classification_report


def prepare_data():
    df=pd.read_csv("https://raw.githubusercontent.com/surenderdev/MLDatasets/refs/heads/master/Iris.csv")
    df.dropna()
    df.to_csv(f'trainingdata.csv', index=False)

def split_data():

    all_data=pd.read_csv(f'trainingdata.csv')
    target_class='Species'
    xx=all_data.loc[:,all_data.columns!=target_class]
    xx = all_data.loc[:, (all_data.columns != target_class) & (all_data.columns != 'Id')]
    yy = all_data.loc[:, all_data.columns == target_class]
    xtrain, xtest, ytrain, ytest = train_test_split(xx,yy,test_size=0.3,stratify=yy,random_state=47)

    np.save(f'xtrain.npy',xtrain)
    np.save(f'xtest.npy',xtest)
    np.save(f'ytrain.npy',ytrain)
    np.save(f'ytest.npy',ytest)
    print("splitting data complete")

def trainbasic_classifier():

    x_train=np.load(f'xtrain.npy',allow_pickle=True)
    y_train=np.load(f'ytrain.npy',allow_pickle=True)

    classifier=LogisticRegression(max_iter=500)

    classifier.fit(x_train,y_train)

    with open(f'mlmodel.pkl', "wb") as f:
        pickle.dump(classifier, f)
    
    print("ML model training done and saved to /mlmodel.pkl")


def predict_testdata():


    with open(f'mlmodel.pkl', "rb") as f:
        mlmodel=pickle.load(f)

    x_test=np.load(f'xtest.npy',allow_pickle=True)
    y_pred=mlmodel.predict(x_test)

    np.save(f"y_pred.npy",y_pred)

def predictprob_testdata():


    with open(f'mlmodel.pkl',"rb") as f:
        mlmodel=pickle.load(f)

    x_test=np.load(f'xtest.npy',allow_pickle=True)
    y_pred=mlmodel.predict_proba(x_test)

    np.save(f"y_pred_prob.npy",y_pred)

def get_metrics():
        ytest=np.load(f'ytest.npy',allow_pickle=True)
        y_pred=np.load(f'y_pred.npy',allow_pickle=True)
        y_pred_prob=np.load(f'y_pred_prob.npy',allow_pickle=True)

        accu = accuracy_score(ytest, y_pred)
        prec = precision_score(ytest, y_pred, average='macro')
        rec = recall_score(ytest, y_pred, average='macro')
        ent = log_loss(ytest, y_pred_prob)

        print("\n")
        print(f"Accuracy: {accu}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"Log Loss: {ent}")
        print("\n")
        print("----------------------------------------------")
        print("\n",classification_report(ytest, y_pred))
        print("----------------------------------------------")



with DAG(
      dag_id='ml_pipeline_demo',
      schedule_interval='@daily',
      start_date=datetime(2024,3,4),
      catchup = False

) as dag:
    task_prepare_data=PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )

    task_traintest_split=PythonOperator(
        task_id='train_test_split',
        python_callable=split_data,
    )

    task_training_basicclassifier=PythonOperator(
        task_id='training_basic_classifier',
        python_callable=trainbasic_classifier,
    )

    task_predict_testdata=PythonOperator(
        task_id='predict_testdata',
        python_callable=predict_testdata,
    )

    task_predict_prob_testdata=PythonOperator(
        task_id='predict_prob_testdata',
        python_callable=predictprob_testdata,
    )

    task_getmetrics=PythonOperator(
        task_id='getmetrics',
        python_callable=get_metrics,
    )


task_prepare_data >> task_traintest_split >>task_training_basicclassifier >> \
    task_predict_testdata >> task_predict_prob_testdata >> task_getmetrics




