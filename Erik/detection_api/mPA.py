import pandas as pd

GT_PATH = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/TestGTSDB/gt.txt'
RESULT_PATH = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/results/cloud/combined_test/result.csv'

gt = pd.read_csv(GT_PATH)
result = pd.read_csv(RESULT_PATH)

