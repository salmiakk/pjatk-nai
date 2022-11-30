import pandas as pd
import pydotplus as pdp
from IPython.display import Image
from sklearn import tree
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from six import StringIO

"""
Load data from CSV file
"""
df = pd.read_csv("wheat_seeds_dataset.csv", delimiter="\t")

"""
x axis is for input data, y for output (seed class)
"""
x = df.drop('class', axis=1)
y = df['class']

"""
Split the data to approx. 67% given to training and 33% to test purposes.
"""
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=33)

"""
Prepare classifier for the decision tree
"""
clasifier = DecisionTreeClassifier()
clasifier = clasifier.fit(X_train, Y_train)
y_pred = clasifier.predict(X_test)

"""
Export decision tree to PNG
"""
dot_file = StringIO()
export_graphviz(
    clasifier,
    out_file=dot_file,
    filled=True,
    rounded=True,
    feature_names=list(x.columns),
    class_names=y.unique()
)
graph = pdp.graph_from_dot_data(dot_file.getvalue())
graph.write_png('wheat_seeds_tree.png')
Image(graph.create_png())

"""
Perform SVM classification with linear kernel
"""
svc = svm.SVC(kernel='linear').fit(x,y)
svc.fit(X_train, Y_train)
"""
Check accurancy of the model
"""
y_pred = svc.predict(X_test)
print("Accuracy is: ",metrics.accuracy_score(Y_test, y_pred))
