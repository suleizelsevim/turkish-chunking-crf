# Türkçe Metinlerde Chunking Etiketlerinin CRF ile Tahmini

Bu proje, Türkçe hikâye metinleri üzerinde chunking etiketlerini otomatik olarak tahmin etmek amacıyla hazırlanmıştır. Veri setindeki metinler CoNLL formatında işaretlenmiştir. Her kelime için `CHUNK-OUTER`, `CHUNK-INNER` ve `CLAUSE` bilgileri verilmiştir.

Model eğitimi ve değerlendirme aşamasında hedef etiket olarak `CHUNK-OUTER` sütunu kullanılmıştır.

---

## Proje Konusu

Bu çalışmada Türkçe cümlelerdeki isim öbekleri, fiil öbekleri ve zarf öbekleri tespit edilmeye çalışılmıştır.

Kullanılan ana etiketler şunlardır:

- `B-NP`: İsim öbeği başlangıcı
- `I-NP`: İsim öbeği devamı
- `B-VP`: Fiil öbeği başlangıcı
- `I-VP`: Fiil öbeği devamı
- `B-ADVP`: Zarf öbeği başlangıcı
- `I-ADVP`: Zarf öbeği devamı
- `O`: Herhangi bir öbeğe ait olmayan kelime

---

## Klasör Yapısı

Proje klasörü aşağıdaki yapıdadır:

```text
NLP_Chunking_Project/
│
├── code/
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
│
├── dataset/
│   └── data.conll
│
└── results/
    ├── metrics.json
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── dataset_statistics.json
```

---

## Veri Seti

Veri seti `dataset/data.conll` dosyasında bulunmaktadır.

Veri seti CoNLL formatında hazırlanmıştır. Her cümle şu yapıda gösterilmiştir:

```text
# sent_id = 1
# text = Gün ağarmaya başlamıştı bile.
# columns = ID FORM CHUNK-OUTER CHUNK-INNER CLAUSE
1 Gün B-NP _ O
2 ağarmaya B-VP _ O
3 başlamıştı I-VP _ O
4 bile O _ O
5 . O _ O
```

Sütunların anlamı:

- `ID`: Kelimenin cümle içindeki sırası
- `FORM`: Kelime
- `CHUNK-OUTER`: Ana chunk etiketi
- `CHUNK-INNER`: İç içe geçmiş öbek etiketi
- `CLAUSE`: Yan cümle / cümlecik bilgisi

Bu projede model eğitimi için `CHUNK-OUTER` sütunu kullanılmıştır.

---

## Kullanılan Yöntem

Projede sıralı etiketleme problemi için Conditional Random Fields, yani CRF, yöntemi kullanılmıştır.

CRF modeli, kelimenin yalnızca kendisini değil, önceki ve sonraki kelimelerden elde edilen özellikleri de dikkate alır. Bu nedenle chunking, named entity recognition ve part-of-speech tagging gibi doğal dil işleme görevlerinde sıkça kullanılan bir yöntemdir.

---

## Kullanılan Özellikler

Her kelime için aşağıdaki özellikler çıkarılmıştır:

- Kelimenin küçük harfli hâli
- Kelimenin büyük harfle başlayıp başlamadığı
- Kelimenin tamamen büyük harf olup olmadığı
- Kelimenin sayı olup olmadığı
- İlk 2 ve 3 karakteri
- Son 2 ve 3 karakteri
- Önceki kelime bilgisi
- Sonraki kelime bilgisi
- Cümle başı / cümle sonu bilgisi

---

## Kurulum

Projeyi çalıştırmadan önce gerekli Python kütüphaneleri yüklenmelidir.

Ana proje klasöründe terminal açılarak şu komut çalıştırılır:

```bash
pip install -r code/requirements.txt
```

Gerekli kütüphaneler `requirements.txt` dosyasında verilmiştir:

```text
numpy
scikit-learn
sklearn-crfsuite
matplotlib
seaborn
```

---

## Çalıştırma

Ana proje klasöründeyken aşağıdaki komut çalıştırılır:

```bash
python code/main.py
```

Program çalıştırıldığında:

1. `dataset/data.conll` dosyasını okur.
2. `CHUNK-OUTER` sütununu hedef etiket olarak alır.
3. Kelime özelliklerini çıkarır.
4. CRF modelini eğitir.
5. 5-fold cross-validation uygular.
6. Accuracy, Precision, Recall ve F-Measure değerlerini hesaplar.
7. Confusion matrix grafiğini oluşturur.
8. Sonuçları `results/` klasörüne kaydeder.

---

## Çıktılar

Program çalıştıktan sonra aşağıdaki dosyalar `results/` klasörüne kaydedilir:

```text
results/
├── metrics.json
├── classification_report.txt
├── confusion_matrix.png
└── dataset_statistics.json
```

Dosyaların açıklamaları:

- `metrics.json`: 5-fold cross-validation sonuçları ve ortalama metrikler
- `classification_report.txt`: Her sınıf için precision, recall ve f1-score değerleri
- `confusion_matrix.png`: Karışıklık matrisi grafiği
- `dataset_statistics.json`: Veri setindeki cümle, token ve etiket dağılımı bilgileri

---

## Değerlendirme

Model 5-fold cross-validation yöntemi ile değerlendirilmiştir. Veri seti beş parçaya ayrılmış, her deneyde dört parça eğitim ve bir parça test verisi olarak kullanılmıştır. Bu işlem beş kez tekrarlanmış ve sonuçların ortalaması alınmıştır.

Hesaplanan temel metrikler:

- Accuracy
- Precision
- Recall
- F-Measure / F1-score

---
