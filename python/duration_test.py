import numpy as np
from datetime import datetime
import numpy_model as np_model
import tensorflow_model as tf_model
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

testList = [[0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
            [0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
            [0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0]]

R = np.array(testList, dtype=np.float32)

np_model = np_model.RecommendModel(R)
tf_model = tf_model.RecommendModel(R)


def duration_test(model):
    t1 = datetime.now()
    predicts, regularization, losses = model.train()
    t2 = datetime.now()
    duration = t2 - t1

    visualize(predicts, regularization, losses)
    return model.predict(), duration


def visualize(predicts, regularization, losses):
    plt.plot([i for i in range(15)], predicts)
    plt.xlabel('Prediction')
    plt.show()
    plt.plot([i for i in range(15)], regularization)
    plt.xlabel('Regularization')
    plt.show()
    plt.plot([i for i in range(15)], losses)
    plt.xlabel('Loss')
    plt.show()


np_prediction, np_duration = duration_test(np_model)
tf_prediction, tf_duration = duration_test(tf_model)

print(f"Numpy model duration: {np_duration}, Prediction: {np_prediction}")
print(f"Tensorflow model duration: {tf_duration}, Prediction: {tf_prediction}")
'''
Using np => Duration :  0:00:00.235455
Using tf => Duration :  0:00:07.405049
'''
