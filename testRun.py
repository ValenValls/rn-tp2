# import argparse
from hebbiano.model import Hebbiano
from utils.dataUtils import get_data, normalize, plot3D

X, Y = get_data("./data/tp2_training_dataset.csv")
X = normalize(X)
hebbianoOja = Hebbiano(850,9)
hebbianoOja.trainOja(X)
predictedOja = hebbianoOja.predict(X)

hebbianoSanger = Hebbiano(850,9)
hebbianoSanger.trainSanger(X)
predictedSanger = hebbianoSanger.predict(X)


#print(predictedOja)
#print(predictedSanger)

plot3D(predictedOja,Y)
plot3D(predictedSanger,Y)


