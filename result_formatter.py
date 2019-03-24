from typing import List
import numpy as np


class ResultFormatter:
    def __init__(self):
        self.index2label_dict = {0: 'clean', 1: 'messy'}  # モデル依存

    def decode_predictions(self, preds, top: int = 2) -> List[tuple]:
        PROB_INDEX = 1
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [(self.index2label_dict[i], pred[i]) for i in top_indices]
            result.sort(key=lambda x: x[PROB_INDEX], reverse=True)
            results.append(result)
        return results

    def convert(self, preds: np.ndarray) -> dict:
        results = self.decode_predictions(preds)
        formatted_result = self.convert_core(results)
        formatted_result['formatter'] = self.__class__.__name__
        return formatted_result

    def convert_core(self, results: List[tuple]) -> dict:
        data = {}
        data['predictions'] = []

        for (label, prob) in results[0]:
            predict_result = {'label': label, 'probability': float(prob)}
            data['predictions'].append(predict_result)

        # 一番確率が大きかったカテゴリを入れる
        data['prediction'] = results[0][0][0]  # 最初のラベル

        return data


class ThreeLevelResultFormatter(ResultFormatter):

    def convert_core(self, results: List[tuple]) -> dict:
        data = {}
        data['predictions'] = []

        for (label, prob) in results[0]:
            predict_result = {'label': label, 'probability': float(prob)}
            data['predictions'].append(predict_result)

        # 一番確率が大きかったカテゴリを入れる
        # 全てのカテゴリが0.5以下だったら、unknown
        # 差が0.2以内だったら、so-so
        # それ以外だったら、先頭のやつ
        work_probs = [x['probability'] for x in data['predictions']]
        prediction_result = 'None'

        if work_probs[0] < 0.5:  # Top1のラベルの確率が0.5未満（自信がない判定）
            prediction_result = 'unknown'
        elif (work_probs[0] - work_probs[1]) < 0.2:
            prediction_result = 'so-so'
        else:
            prediction_result = results[0][0][0] # 最初のラベル

        data['prediction'] = prediction_result

        return data
