# 📝 Türkçe Yorum Sınıflandırıcı

**Türkçe Yorum Sınıflandırıcı**, Naive Bayes algoritmasını kullanarak Türkçe metinleri belirli kategorilere göre sınıflandıran bir makine öğrenmesi modelidir. **TF-IDF (Term Frequency - Inverse Document Frequency)** yöntemi ile kelime vektörleştirme işlemi yapılır ve **Multinomial Naive Bayes** algoritması kullanılarak model eğitilir.

## 🚀 Proje İçeriği
- **Veri Seti Yükleme**: `price_related_reviews.csv` dosyası pandas ile okunur.
- **Metin Ön İşleme**: TF-IDF yöntemi ile metin verisi vektörleştirilir ve Türkçe **stop words** (önemsiz kelimeler) filtrelenir.
- **Veri Bölme**: `train_test_split` ile veriler eğitim ve test olarak ayrılır.
- **Model Eğitimi**: **Multinomial Naive Bayes** algoritması kullanılarak model eğitilir.
- **Model Değerlendirme**: Model test verisi üzerinde değerlendirilir ve doğruluk oranı hesaplanır.
- **Confusion Matrix Görselleştirme**: `seaborn` kullanılarak hata matrisi görselleştirilir.

## 📦 Kullanılan Kütüphaneler
Bu proje aşağıdaki Python kütüphanelerini kullanmaktadır:
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Eğer bu kütüphaneler yüklü değilse, aşağıdaki komutu kullanarak yükleyebilirsiniz:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```
## 📊 Sonuç Görselleştirme

<img width="400" alt="Ekran Resmi 2025-03-15 17 29 39" src="https://github.com/user-attachments/assets/c33d0010-e82b-4595-b5bf-28b662c16be5" />

📌 Bu proje, Türkçe metin sınıflandırma konusunda temel bir model geliştirmek amacıyla hazırlanmıştır ve eğitim amaçlıdır.
