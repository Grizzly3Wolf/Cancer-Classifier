import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
breast_cancer_data=load_breast_cancer()
# print("Data:\n",breast_cancer_data.data[0])
# print("Names:\n",breast_cancer_data.feature_names)
# print("Target:\n",breast_cancer_data.target,"Target Names:\n",breast_cancer_data.target_names)
X_train,X_test,y_train,y_test=train_test_split(breast_cancer_data.data,breast_cancer_data.target,test_size=0.2,random_state=100)
# print(len(X_train),len(y_train))
from sklearn.neighbors import KNeighborsClassifier
sc=0
accuracies=[]
for i in range(1,100):
  knn=KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train,y_train)
  accuracies.append(knn.score(X_test,y_test))
  if knn.score(X_test,y_test)>sc:
    sc=knn.score(X_test,y_test)
    k=i
print(accuracies)
plt.plot(list(range(1,100)),accuracies)
plt.title("K value vs Score")
plt.xlabel("K")
plt.ylabel("Validation Accuracy")
plt.show()
plt.close()
print("The best score",sc,"\nBest K value:",k)
