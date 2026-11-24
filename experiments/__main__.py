"""Run every experiment in turn: python -m experiments"""
from experiments import (
    admissions_regression, colour_quantization, image_inpainting, logistic_regression,
    majority_vote, news_classification, speech_emotion,
)

for module in (admissions_regression, image_inpainting, colour_quantization, logistic_regression,
               majority_vote, news_classification, speech_emotion):
    module.main()
