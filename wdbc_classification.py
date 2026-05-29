from sklearn import datasets, svm, tree, neighbors, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드 [cite: 8, 20]
wdbc = datasets.load_breast_cancer()
X = wdbc.data
y = wdbc.target

# 데이터 정보 출력 [cite: 22-26]
print("Data shape:", X.shape)
print("Target shape:", y.shape)
print("Target names:", wdbc.target_names)
print("First 5 feature names:", wdbc.feature_names[:5])

# 데이터 분할 (훈련/테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 정의 [cite: 51-56]
models = {
    "SVM": svm.SVC(),
    "Decision Tree": tree.DecisionTreeClassifier(random_state=42),
    "KNN": neighbors.KNeighborsClassifier(),
    "Random Forest": ensemble.RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

# 모델 학습 및 비교 [cite: 48-61]
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\n최고 성능 모델: {best_model_name} (정확도: {best_accuracy:.4f})")

# 혼동 행렬 시각화 및 저장 [cite: 64-66]
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wdbc.target_names)
disp.plot()
plt.title(f"Confusion Matrix: {best_model_name}")
plt.savefig("wdbc_classification_matrix.png")

# 산점도 생성 및 저장 [cite: 67]
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
plt.xlabel(wdbc.feature_names[0])
plt.ylabel(wdbc.feature_names[1])
plt.title("WDBC Scatter Plot")
plt.savefig("wdbc_classification_scatter.png")
