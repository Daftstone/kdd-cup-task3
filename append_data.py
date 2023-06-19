import pandas as pd

# add all test data of task1 and task2
data1 = pd.read_csv("data/sessions_test_task1_phase1.csv")
data2 = pd.read_csv("data/sessions_test_task1_phase2.csv")
data3 = pd.read_csv("data/sessions_test_task2_phase1.csv")
data4 = pd.read_csv("data/sessions_test_task2_phase2.csv")

data = data1.append(data2).append(data3).append(data4)
data.to_csv("data/append_data.csv", index=False)