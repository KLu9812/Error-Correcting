import pandas as pd

q = pd.read_csv("4ommluresponsesall.csv")
for i in range(5):
    answers = []
    for j in range(1000):
        answers.append(q["answers"][i * 1000 + j])
    data = pd.DataFrame({"answers": answers})
    data.to_csv("4ommluresponses" + str(i + 1) + ".csv")
