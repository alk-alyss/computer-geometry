from model import Model
import os
from typing import List
import time

resources_folder = "./final/resources/"
coatings_folder = resources_folder + "coatings/"
objects_folder = resources_folder + "objects/"

def loadModels(path:str) -> List[Model]:
    models:List[Model] = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for i, file in enumerate(filenames):
            if i == 10: break
            print(f"loading model {i}")
            model = Model(coatings_folder + file)
            models.append(model)
            print("done")

    return models

def compareModles(m1:Model, m2:Model, coatings=False) -> float:
    eigs1 = m1.get_eigenvalues(high=coatings)
    eigs2 = m2.get_eigenvalues(high=coatings)
    min_len = min(len(eigs1), len(eigs2))

    similarity = 0
    for i in range(min_len):
        eig1 = abs(eigs1[i])**0.5
        eig2 = abs(eigs2[i])**0.5

        numerator = (eig1 - eig2)**2
        denominator = eig1 + eig2

        similarity += numerator/denominator

    return similarity/2

def classifyModels(models:List[Model], threshold:float = 1, coatings=False) -> List:
    similarities = []

    for i, model1 in enumerate(models):
        similarities.append([])
        for j, model2 in enumerate(models[i:]):
            similarity = compareModles(model1, model2, coatings=coatings)

            # if similarity <= threshold:
            #     similarities[-1].append(model2)
            similarities[-1].append(similarity)

    return similarities


if __name__ == "__main__":
    models = loadModels(coatings_folder)

    coatingSimilarities = classifyModels(models, coatings=True)
    # objectSimilarities = classifyModels(models, compareObjects)

    print(coatingSimilarities)