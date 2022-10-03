import dill
import json
import pandas as pd
import os
import datetime


def load_model():
    for root, dirs, files in os.walk(r"C:\Users\Marian\airflow_hw\data\models"):
        with open(root + r'\\' + files[0], 'rb') as file:
            model = dill.load(file)
    return model


def save_csv(df):
    file_name = f'preds_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv'
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, 'data\\predictions')) + '\\' + file_name
    return df.to_csv(file_path, sep=',', index=False)


def predict():
    df = pd.DataFrame({'car_id': [], 'pred': []})
    for root, dirs, files in os.walk(r"C:\Users\Marian\airflow_hw\data\test"):
        for i in files:
            with open(root + r'\\' + i, 'rb') as file:
                sample = pd.DataFrame.from_dict([json.load(file)])
                y = load_model().predict(sample)
                pred_df = pd.DataFrame({'car_id': str([sample['id'][0]])[1:-1], 'pred': [y[0]]})
                df = pd.concat([df, pred_df])
    save_csv(df)


if __name__ == '__main__':
    predict()
