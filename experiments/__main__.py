"""Run every experiment in turn: python -m experiments"""
from experiments import (
    adaboost, admissions_regression, collaborative_filtering, colour_quantization, image_inpainting,
    logistic_regression, majority_vote, neural_network, news_classification, speech_emotion, svm,
)

for module in (admissions_regression, image_inpainting, colour_quantization, logistic_regression, svm,
               majority_vote, adaboost, neural_network, news_classification, speech_emotion,
               collaborative_filtering):
    module.main()
