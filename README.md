# Laporan Proyek Akhir Machine Learning - Muhammad Revin Arnan

## Domain Proyek

Dalam era digital yang terus berkembang, industri hiburan seperti film dan televisi telah mengalami perubahan besar dalam cara konsumen menemukan dan mengakses konten. Platform streaming dan database film online telah menjadi sangat populer, menyediakan akses tak terbatas ke ribuan judul film dari berbagai genre dan tahun rilis. Namun, dengan banyaknya pilihan film yang tersedia, mencari tontonan yang sesuai dengan preferensi dan minat pengguna dapat menjadi tugas yang menantang.

Inilah di mana sistem rekomendasi film berbasis machine learning menjadi relevan dan berperan penting. Sistem rekomendasi film bertujuan untuk membantu pengguna menemukan film-film yang mungkin diminati berdasarkan preferensi sebelumnya dan perilaku menonton mereka. Hal ini menciptakan pengalaman yang lebih personal dan memungkinkan pengguna untuk menemukan film-film yang sesuai dengan selera mereka tanpa harus secara manual menjelajahi seluruh katalog film. [1]

Sebagai contoh, platform streaming terkemuka seperti Netflix, Amazon Prime Video, atau Disney+ menggunakan sistem rekomendasi untuk memberikan pengalaman pengguna yang disesuaikan dan meningkatkan retensi pelanggan. Sistem ini memanfaatkan teknologi machine learning dan algoritma yang kompleks untuk menganalisis data historis pengguna, seperti riwayat penontonannya, peringkat film yang telah ditonton, durasi menonton, dan interaksi dengan platform lainnya.

## Business Understanding

### Problem Statements

Bagaimana cara memberikan rekomendasi film-film yang dapat disukai pengguna berdasarkan film yang telah ditonton sebelumnya?

### Goals

Pengguna mendapatkan sejumlah rekomendasi film berdasarkan film yang telah ditonton sebelumnya.

### Solution statements

Membuat sistem rekomendasi dengan metode _content-based filtering_ dan _hybrid filtering_ dengan menghitung derajat kesamaan menggunakan *cosine similarity* berdasarkan deskripsi dari film dan model SVD (_Singular Value Decomposition_) untuk melihat relasi antara _users_ dan _items_.

## Data Understanding

*Dataset* yang digunakan pada proyek sistem rekomendasi ini adalah data "*The Movies Dataset*" dari Kaggle. *Dataset* dapat diakses pada *link* berikut: [Link dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).  

### Variabel-variabel pada *Metadata The Movies Dataset* dataset adalah sebagai berikut:

- *adult*: apakah film tergolong _X-Rated_/film dewasa atau tidak.
- *belongs_to_collection*: informasi tentang serial film yang dimiliki film tersebut.
- *budget*: adalah biaya produksi dalam USD.
- *genres*: genre yang terkait pada film.
- *homepage*: merupakan _official homepage_ dari film.
- *id*: id film.
- *imdb_id*: IMDB id film.
- *original_language*: adalah bahasa asli film.
- *original_title*: judul asli dari film.
- *overview*: deskripsi atau sinopsis film.
- *popularity*: popularitas skor dari IMDB.
- *poster_path*: adalah URL dari poster film.
- *production_companies*: perusahaan produsen film.
- *production_countries*: negara produsen film.
- *release_date*: tanggal rilis bioskop.
- *revenue*: total pendapatan film dalam USD.
- *runtime*: durasi film dalam menit.
- *spoken_language*: bahasa yang digunakan dalam percakapan film.
- *status*: status film (_released, to be released, announced, etc)_.
- *tagline*: tagline dari film.
- *title*: judul resmi film.
- *video*: apakah ada tayangan video dalam film.
- *vote_average*: rata-rata _rating_ dalam film.
- *vote_count*: jumlah vote yang diberikan dari pengguna.

Tahapan pemahaman *dataset* yang dilakukan diantaranya:

1.  Melihat macam dan jumlah kolom dengan df.info,

    *Tabel 1. Dataframe Info*

    | #   | Column                | Non-Null Count | Dtype   |
    | --- | --------------------- | -------------- | ------- |
    | 0   | adult                 | 45466 non-null | object  |
    | 1   | belongs_to_collection | 4494 non-null  | object  |
    | 2   | budget                | 45466 non-null | object  |
    | 3   | genres                | 45466 non-null | object  |
    | 4   | homepage              | 7782 non-null  | object  |
    | 5   | id                    | 45466 non-null | object  |
    | 6   | imdb_id               | 45449 non-null | object  |
    | 7   | original_language     | 45455 non-null | object  |
    | 8   | original_title        | 45466 non-null | object  |
    | 9   | overview              | 44512 non-null | object  |
    | 10  | popularity            | 45461 non-null | object  |
    | 11  | poster_path           | 45080 non-null | object  |
    | 12  | production_companies  | 45463 non-null | object  |
    | 13  | production_countries  | 45463 non-null | object  |
    | 14  | release_date          | 45379 non-null | object  |
    | 15  | revenue               | 45460 non-null | float64 |
    | 16  | runtime               | 45203 non-null | float64 |
    | 17  | spoken_languages      | 45460 non-null | object  |
    | 18  | status                | 45379 non-null | object  |
    | 19  | tagline               | 20412 non-null | object  |
    | 20  | title                 | 45460 non-null | object  |
    | 21  | video                 | 45460 non-null | object  |
    | 22  | vote_average          | 45460 non-null | float64 |
    | 23  | vote_count            | 45460 non-null | float64 |

    
    Pada Tabel 1, dapat dilihat bahwa *dataset* memiliki 24 kolom dan 45466 baris.

2. Menampilkan kata yang sering muncul pada judul film,
   <img width="650" alt="wc_title" src="https://github.com/revinarnan/project-ml-terapan/assets/45119832/c0ddc7e4-dec3-4387-80cb-8faadc1747a8">
   
   *Gambar 1. Word Cloud dari Judul Film*

   Dari hasil *title word cloud visualization*, dapat dilihat kata yang sering menjadi judul film adalah '_Love_', diikuti dengan '_Man_' dan '_Girl_'. Dari sini dapat diasumsikan banyak film ber-_genre_ _romance_ pada _dataset_.

3. Melihat kata yang sering muncul pada deskripsi film dengan *wordcloud*,

   <img width="650" alt="wc_overview" src="https://github.com/revinarnan/project-ml-terapan/assets/45119832/8fa19449-b149-408b-a387-88b7313f3436"> 

   *Gambar 2. Word Cloud dari Deskripsi Film*

   Kata yang sering muncul dalam deskripsi adalah _life, find, love, family_ dan sebagainya.

4. Melihat 10 negara produsen film terbanyak,

   <img width="650" alt="top10_country" src="https://github.com/revinarnan/project-ml-terapan/assets/45119832/7483b297-b512-438e-922c-75859558051d">

   *Gambar 4. Diagram Batang 10 Negara Produsen Film Teratas*

   Pada Gambar 4, USA menjadi negara produsen film terbanyak, diikuti oleh UK, dan France.

5. Melihat 10 studio produsen film terbanyak.
   
   <img width="650" alt="top10_studio" src="https://github.com/revinarnan/project-ml-terapan/assets/45119832/329e76fc-19ba-4c03-96e9-8b9b38af7c34">

   *Gambar 5. Diagram Batang 10 Studio Produsen Film Teratas*

   Pada Gambar 5, Canal+ menjadi studio dengan produksi film terbanyak, diikuti dengan Warner Bros.

## Data Preparation

Proses pembersihan data dan preparasi yang dilakukan diantaranya sebagai berikut:

- Menggabungkan dataset metadata dengan dataset links yang berisi movie_id, imdb_id, dan tmdb_id.
- Menambahkan kolom '_year_' pada dataframe.
- Menghapus kolom dengan id [19730, 29503, 35587] karena memiliki format yang tidak sesuai.
- Menghapus data duplikat berdasarkan kolom '_title_' dan '_overview_'.
- Menggabungkan kolom '_tagline_' dan kolom '_overview_' menjadi kolom '_description_'.
- *Tokenizing* *text*: Menandai setiap kata dengan angka dan memetakan data *text* pada *token* tersebut.

## Modeling

Proyek ini menggunakan metode _cosine similarity_ untuk menghitung derajat kesamaan dari dua entitas dalam vektor multidimensi (_dot product)_. Dalam konteks yang lebih umum, _cosine similarity_ digunakan untuk membandingkan dua objek atau entitas berdasarkan fitur atau atribut yang dimiliki oleh keduanya. Misalnya, terdapat dua vektor untuk mewakili dua film berdasarkan fitur seperti genre, popularitas, dan peringkat pengguna. 

   Rumus _cosine similarity_ adalah sebagai berikut:
      $$sim{(d1, d2)} = {\vec{V}(d1) \cdot \vec{V}(d2) \over (|\vec{V}(d1)| \cdot |\vec{V}(d2)|)}.$$

   Dimana:
   - `V~(d1)` mewakilkan vektor `d1`.
   - `V~(d2)` mewakilkan vektor `d2`.
   - `|V~(d1)|` adalah panjang vektor `d1`.
   - `|V~(d2)|` adalah panjang vektor `d2`.

_Cosine similarity_ menghitung sudut antara dua vektor ini, diukur dalam derajat atau radian. Jika kedua vektor berada pada arah yang sama atau sangat dekat, maka _cosine similarity_ akan mendekati nilai 1, yang berarti kedua entitas sangat mirip satu sama lain. Sebaliknya, jika kedua vektor berada pada arah yang berlawanan atau hampir berlawanan, _cosine similarity_ akan mendekati nilai -1, yang berarti kedua entitas sangat berbeda satu sama lain. Jika kedua vektor tegak lurus, maka _cosine similarity_ akan menjadi 0, yang menunjukkan bahwa kedua entitas tidak memiliki kesamaan fitur [2].

Selain itu, proyek ini juga menggunakan teknik _Singular Value Decomposition_ meminimalisir nilai RMSE (_Root Mean Square Error_). Singular Value Decomposition (SVD) adalah teknik yang digunakan untuk memecah suatu matriks menjadi tiga matriks yang lebih sederhana. Dengan cara ini, dapat mempermudah untuk memahami pola dan hubungan antara _users_ dan _items_ (film) [3].

## Evaluation & Result

Metriks yang digunakan pada proyek ini adalah metriks RMSE (_Root Mean Square Error_). Metriks ini digunakan untuk mengukur seberapa akurat model dalam memperkirakan nilai sebenarnya. RMSE menghitung perbedaan antara nilai yang diprediksi oleh model dan nilai yang sebenarnya. Artinya, untuk setiap data yang dimiliki, metrik ini akan menghitung selisih antara nilai prediksi dan nilai sebenarnya. Kemudian, akan mengambil rata-rata dari seluruh selisih tersebut dan menghitung akar kuadratnya. Semakin nilai RMSE mendekari 0, semakin baik pula model dalam memperkirakan nilai sebenarnya [4]. Berikut ini merupakan rumus dari metrik RMSE:

$$RMSE = {\sqrt{ \Sigma{(yᵢ - ȳ)^2 \over n}}}$$

Hasil metriks ini dari model yang dikembangkan adalah sebagai berikut: 

*Tabel 1. Dataframe Info*
|                | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean   | Std   |
|----------------|--------|--------|--------|--------|--------|--------|-------|
| RMSE (testset) | 0.8988 | 0.9064 | 0.8931 | 0.8984 | 0.8944 | 0.8982 | 0.0047|
| MAE (testset)  | 0.6926 | 0.6954 | 0.6893 | 0.6907 | 0.6871 | 0.6910 | 0.0028|
| Fit time       | 1.14   | 1.15   | 1.16   | 1.69   | 1.81   | 1.39   | 0.30  |
| Test time      | 0.14   | 0.12   | 0.13   | 0.22   | 0.38   | 0.20   | 0.10  |

Model mendapat nilai rata-rata RMSE dari 5 Fold sebesar 0.8982.

### Content-Based Filtering

Pada metode _Content-Based Filtering_, penulis mencari Top 30 Film yang memiliki kesamaan paling dekat dengan judul film yang diberikan menggunakan _cosine similarity_ dan mengurutkannya berdasarkan rata-rata nilai voting dari pengguna. Dalam hal ini penulis mencoba memberikan input judul film ' '

### Hybrid Filtering

## Kesimpulan

Model *machine learning* dapat dilatih untuk mengklasifikasikan berita palsu dan berita sesungguhnya. Dengan menggunakan teknik *deep learning*, model dapat memberikan akurasi sebesar 98,5%. Dalam kasus ini, hasil yang didapatkan cukup baik. Namun, model menggunakan *dataset* kumpulan berita dari sumber berbahasa Inggris, sehingga jika ingin digunakan dalam kasus klasifikasi berita dalam negeri, model perlu menggunakan *dataset* berita berbahasa Indonesia. Tentunya dengan beberapa penyesuaian pada tahap preparasi data.

## Saran

Untuk dapat memprediksi berita *hoax* di Indonesia, model perlu dilatih dengan menggunakan *dataset* dengan Bahasa Indonesia.

## Daftar Pustaka

[1] https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada

[2] https://www.sciencedirect.com/science/article/abs/pii/B9780123814791000022

[3] https://towardsdatascience.com/understanding-singular-value-decomposition-and-its-application-in-data-science-388a54be95d

[4] https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e

S. Kostadinov, “How recurrent neural networks work,” Medium, https://towardsdatascience.com/learn-how-recurrent-neural-networks-work-84e975feaaf7 (accessed Jun. 29, 2023). 

[2] B. Harikrishnan N, "Confusion Matrix, Accuracy, Precision, Recall, F1 Score", Analytics Vidhya, https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd (accessed Jun. 25, 2023).

[3] A. P. Gema and D. Suhartono, “Recurrent neural network (RNN) Dan Gated Recurrent Unit (GRU),” Recurrent Neural Network (RNN) dan Gated Recurrent Unit (GRU), https://socs.binus.ac.id/2017/02/13/rnn-dan-gru/ (accessed Jun. 25, 2023). 

[4] K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, “Fake news detection on social media: A Data Mining Perspective: ACM SIGKDD Explorations Newsletter: Vol 19, no 1,” ACM SIGKDD Explorations Newsletter, https://dl.acm.org/doi/10.1145/3137597.3137600 (accessed Jun. 24, 2023). 
