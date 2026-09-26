import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Veri setini oku
df = pd.read_csv("sample2000binary.csv", sep=";")

print("Veri seti boyutu:", df.shape)
print("\nEtiket dağılımı:")
print(df["Label"].value_counts())

# 2. Eksik kayıt varsa temizle
df = df.dropna(subset=["Label", "Text"])

# True = 1 / False = 0
df["Label"] = df["Label"].astype(int)

# 3. X ve y
X = df["Text"]
y = df["Label"]

# 4. Eğitim / test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nEğitim verisi:", len(X_train))
print("Test verisi:", len(X_test))

# 5. TF-IDF
tfidf = TfidfVectorizer(
    lowercase=True,
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF matrisi:", X_train_tfidf.shape)

# 6. Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 7. Tahmin
y_pred = model.predict(X_test_tfidf)

# 8. Performans
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL SONUÇLARI")
print("==============================")

print(f"Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Olumsuz", "Olumlu"]
    )
)

# 9. Örnek yorum testi
ornek_yorum = "Ürün çok güzel, hızlı geldi ve çok memnun kaldım."

ornek_tfidf = tfidf.transform([ornek_yorum])

tahmin = model.predict(ornek_tfidf)[0]
olasilik = model.predict_proba(ornek_tfidf)[0]

print("\n==============================")
print("ÖRNEK YORUM ANALİZİ")
print("==============================")

print("Yorum:", ornek_yorum)

if tahmin == 1:
    print("Tahmin: OLUMLU")
    print(f"Olumlu olasılığı: %{olasilik[1] * 100:.1f}")
else:
    print("Tahmin: OLUMSUZ")
    print(f"Olumsuz olasılığı: %{olasilik[0] * 100:.1f}")