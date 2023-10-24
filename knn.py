# Import library
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# Load dataset (contoh menggunakan dataset iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model K-NN dengan k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model dengan data latih
knn.fit(X_train, y_train)

# Menguji model dengan data uji
accuracy = knn.score(X_test, y_test)
print(f'Akurasi: {accuracy}')

# Menyimpan model ke dalam file
import joblib

joblib.dump(knn, 'knn_model.joblib')
