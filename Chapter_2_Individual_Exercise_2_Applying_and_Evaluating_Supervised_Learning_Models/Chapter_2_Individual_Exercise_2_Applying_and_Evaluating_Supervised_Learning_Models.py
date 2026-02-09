import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


data = load_iris()

x = data.data
y = data.target

print(data.target_names)

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.2, random_state = 42, stratify = y)

def eval_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')

    return accuracy, f1, precision, recall



dt = DecisionTreeClassifier(random_state = 42)
dtresult = eval_model(dt, x_train, x_test, y_train, y_test)

rf = RandomForestClassifier(n_estimators = 100,random_state = 42)
rfresult = eval_model(dt, x_train, x_test, y_train, y_test)


gb = GradientBoostingClassifier(random_state = 42)
gbresult = eval_model(dt, x_train, x_test, y_train, y_test)


k_values = range(1, 21)
knn_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    knn_accuracies.append(acc)


plt.figure()
plt.plot(k_values, knn_accuracies, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs k")
plt.show()

optimal_k = k_values[np.argmax(knn_accuracies)]
print("Optimal k:", optimal_k)


knn_model = KNeighborsClassifier(n_neighbors=optimal_k)

knn_metrics = eval_model(
    knn_model, x_train, x_test, y_train, y_test
)



results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "KNN"
    ],
    "Accuracy": [
        dtresult[0],
        rfresult[0],
        gbresult[0],
        knn_metrics[0]
    ],
    "Precision": [
        dtresult[1],
        rfresult[1],
        gbresult[1],
        knn_metrics[1]
    ],
    "Recall": [
        dtresult[2],
        rfresult[2],
        gbresult[2],
        knn_metrics[2]
    ],
    "F1-Score": [
        dtresult[3],
        rfresult[3],
        gbresult[3],
        knn_metrics[3]
    ]
})

print(results)


