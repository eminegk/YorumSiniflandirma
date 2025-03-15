import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Türkçe stop words listesi
stop_words_tr = ["ve", "veya", "ama", "fakat", "ancak", "gibi", "çok", "bu", "şu", "o", "bunun", "şunun", "o", "bir", "birkaç", "de", "da", "ile", "için", "üzerinde", "altında", "her", "bazı", "tüm", "bütün", "nasıl", "neden", "ne", "kim", "hangi", "niçin", "mı", "mi", "mu", "mü"]


dosya_yolu = "/Users/eminegok/Desktop/price_related_reviews.csv"

veri = pd.read_csv(dosya_yolu, encoding="unicode_escape")


girisler = veri["Text"]  #  yorumlar
cikis = veri["Category"]  # yorumların hangi kategoriye ait olduğu


vectorizer = TfidfVectorizer(stop_words=stop_words_tr)
X = vectorizer.fit_transform(girisler)

X_train, X_test, y_train, y_test = train_test_split(X, cikis, test_size=0.3, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

# görselleştir
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))