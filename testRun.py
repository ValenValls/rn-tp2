import argparse
from hebbiano.model import Hebbiano
from utils.dataUtils import get_data, normalize
import numpy as np
from matplotlib import pyplot as plt

X, Y = get_data("./data/tp2_training_dataset.csv")
X = normalize(X)
hebbianoOja = Hebbiano(850,9)
hebbianoOja.trainOja(X)
predictedOja = hebbianoOja.predict(X)

hebbianoSanger = Hebbiano(850,9)
hebbianoSanger.trainSanger(X)
predictedSanger = hebbianoSanger.predict(X)

print(Y)
print(predictedOja)
print(predictedSanger)

