# Laporan Proyek Akhir Machine Learning - Muhammad Revin Arnan

## Domain Proyek

Dalam era digital yang terus berkembang, industri hiburan seperti film dan televisi telah mengalami perubahan besar dalam cara konsumen menemukan dan mengakses konten. _Streaming platform_ dan _database_ film _online_ telah menjadi sangat populer, menyediakan akses tak terbatas ke ribuan judul film dari berbagai genre dan tahun rilis. Namun, dengan banyaknya pilihan film yang tersedia, mencari tontonan yang sesuai dengan preferensi dan minat pengguna dapat menjadi tugas yang menantang.

Inilah di mana sistem rekomendasi film berbasis _machine learning_ menjadi relevan dan berperan penting. Sistem rekomendasi film bertujuan untuk membantu pengguna menemukan film-film yang mungkin diminati berdasarkan preferensi sebelumnya dan perilaku menonton mereka. Hal ini menciptakan pengalaman yang lebih personal dan memungkinkan pengguna untuk menemukan film-film yang sesuai dengan selera mereka tanpa harus secara manual menjelajahi seluruh katalog film [1].

Sebagai contoh, _streaming platform_ terkemuka seperti Netflix, Amazon Prime Video, atau Disney+ menggunakan sistem rekomendasi untuk memberikan pengalaman pengguna yang disesuaikan dan meningkatkan retensi pelanggan. Sistem ini memanfaatkan teknologi _machine learning_ dan algoritma yang kompleks untuk menganalisis data historis pengguna, seperti riwayat penontonannya, peringkat film yang telah ditonton, durasi menonton, dan interaksi dengan _platform_ lainnya.

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
- Menghapus kolom dengan id [19730, 29503, 35587] karena memiliki format yang tidak sesuai dengan _drop_.
- Menghapus data duplikat berdasarkan kolom '_title_' dan '_overview_' dengan _drop_duplicate_.
- Mengambil nilai dari 'name' pada kolom 'genres'.
- Menggabungkan kolom '_tagline_' dan kolom '_overview_' menjadi kolom '_description_'.
- *Tokenizing* *text*:
   - Menandai setiap kata dengan angka dan memetakan data *text* pada *token* tersebut.
   - Menggunakan _analyzer_ 'word'.
   - Parameter ngram_range bernilai (1, 2). Artinya tokenizing untuk setiap unigram dan bigram dalam corpus.
   - `min_df = 0` artinya tidak ada minimal _terms_ dari _vocabulary_ dalam setiap _document_.
   - `stop_words = 'english'` artinya menyaring kata yang termasuk dalam vocab english stop_words.

## Modeling & Result

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

### Content-Based Filtering

Pada metode _Content-Based Filtering_, penulis mencari Top 30 Film yang memiliki kesamaan paling dekat dengan judul film yang diberikan menggunakan _cosine similarity_ dan mengurutkannya berdasarkan rata-rata nilai voting dari pengguna. Dalam hal ini penulis mencoba memberikan input judul film 'Spectre', sistem rekomendasi _content-based filtering_ akan memberikan rekomendasi film selanjutnya seperti pada Tabel 2:

   *Tabel 2. Hasil Top 10 Rekomendasi Film Berdasarkan Film Spectre*

   |   title                |   year  |   vote_count  |   vote_average  |   id    |   mov_genre                                 |
   |------------------------|---------|---------------|-----------------|---------|---------------------------------------------|
   |   Skyfall              |   2012  |   7718.0      |   6.9           |   37724 |   [Action, Adventure, Thriller]             |
   |   Casino Royale        |   2006  |   3930.0      |   7.3           |   36557 |   [Adventure, Action, Thriller]             |
   |   Quantum of Solace    |   2008  |   3015.0      |   6.1           |   10764 |   [Adventure, Action, Thriller, Crime]      |
   |   Watchmen             |   2009  |   2892.0      |   7.0           |   13183 |   [Action, Mystery, Science Fiction]        |
   |   Die Another Day      |   2002  |   1112.0      |   5.8           |   36669 |   [Adventure, Action, Thriller]             |
   |   Dr. No               |   1962  |   953.0       |   6.9           |   646   |   [Adventure, Action, Thriller]             |
   |   Safe Haven           |   2013  |   840.0       |   6.9           |   112949|   [Romance]                                 |
   |   From Russia with Love|   1963  |   773.0       |   6.9           |   657   |   [Action, Thriller, Adventure]             |
   |   Thunderball          |   1965  |   572.0       |   6.5           |   660   |   [Adventure, Action, Thriller]             |
   |   Diamonds Are Forever |   1971  |   562.0       |   6.3           |   681   |   [Adventure, Action, Thriller]             |

Sebagai contoh lain, penulis mencoba memberikan input film 'Avengers: Age of Ultron', sistem akan merekomendasikan film yang dapat ditonton pengguna seperti pada Tabel 3:

   *Tabel 3. Hasil Top 10 Rekomendasi Film Berdasarkan Film Avengers: Age of Ultron*
   
   |   title                                   |   year  |   vote_count  |   vote_average  |   id    |   mov_genre                                          |
   |-------------------------------------------|---------|---------------|-----------------|---------|------------------------------------------------------|
   |   The Avengers                            |   2012  |   12000.0     |   7.4           |   24428 |   [Science Fiction, Action, Adventure]               |
   |   Iron Man                                |   2008  |   8951.0      |   7.4           |   1726  |   [Action, Science Fiction, Adventure]               |
   |   Iron Man 3                              |   2013  |   8951.0      |   6.8           |   68721 |   [Action, Adventure, Science Fiction]               |
   |   Captain America: Civil War              |   2016  |   7462.0      |   7.1           |   271110|   [Adventure, Action, Science Fiction]               |
   |   Iron Man 2                              |   2010  |   6969.0      |   6.6           |   10138 |   [Adventure, Action, Science Fiction]               |
   |   Kingsman: The Secret Service            |   2015  |   6069.0      |   7.6           |   207703|   [Crime, Comedy, Action, Adventure]                 |
   |   Captain America: The Winter Soldier     |   2014  |   5881.0      |   7.6           |   100402|   [Action, Adventure, Science Fiction]               |
   |   Men in Black 3                          |   2012  |   4228.0      |   6.3           |   41154 |   [Action, Comedy, Science Fiction]                  |
   |   Back to the Future Part II              |   1989  |   3926.0      |   7.4           |   165   |   [Adventure, Comedy, Family, Science Fiction]       |
   |   Total Recall                            |   2012  |   2540.0      |   5.8           |   64635 |   [Action, Science Fiction, Adventure, Thriller]     |


### Hybrid Filtering

Pada metode _Hybrid Filtering_, penulis mengkombinasikan pencarian Top 30 Film yang memiliki kesamaan paling dekat dengan judul film yang diberikan menggunakan _cosine similarity_, dengan hasil prediksi _rating_ dari id pengguna menggunakan SVD, dan mengurutkannya berdasarkan estimasi _rating_ tertinggi. Dalam hal ini penulis mencoba memberikan input film berjudul 'Spectre' dan membandingkan hasil rekomendasi antara pengguna dengan ID '3000' dan ID '404'. Hasil sistem rekomendasi sebagai berikut:

   *Tabel 4. Hasil Top 10 Rekomendasi Film berdasarkan Judul Spectre dan User ID 3000*

   |   title                                   |   year  |   vote_count  |   vote_average  |   id    |   mov_genre                                 |   rating_est  |
   |-------------------------------------------|---------|---------------|-----------------|---------|---------------------------------------------|---------------|
   |   Casino Royale                           |   2006  |   3930.0      |   7.3           |   36557 |   [Adventure, Action, Thriller]             |   3.926588    |
   |   Skyfall                                 |   2012  |   7718.0      |   6.9           |   37724 |   [Action, Adventure, Thriller]             |   3.924010    |
   |   The Spy Who Loved Me                    |   1977  |   515.0       |   6.6           |   691   |   [Adventure, Action, Thriller]             |   3.731661    |
   |   On Her Majesty's Secret Service         |   1969  |   464.0       |   6.5           |   668   |   [Adventure, Action, Thriller]             |   3.725076    |
   |   Watchmen                                |   2009  |   2892.0      |   7.0           |   13183 |   [Action, Mystery, Science Fiction]        |   3.704353    |
   |   Dr. No                                  |   1962  |   953.0       |   6.9           |   646   |   [Adventure, Action, Thriller]             |   3.670961    |
   |   The Man with the Golden Gun             |   1974  |   533.0       |   6.4           |   682   |   [Adventure, Action, Thriller]             |   3.663120    |
   |   The Tall Blond Man with One Black Shoe  |   1972  |   58.0        |   6.9           |   12089 |   [Comedy, Mystery]                         |   3.657653    |
   |   From Russia with Love                   |   1963  |   773.0       |   6.9           |   657   |   [Action, Thriller, Adventure]             |   3.638990    |
   |   To Live and Die in L.A.                 |   1985  |   129.0       |   6.8           |   9846  |   [Action, Crime, Thriller]                 |   3.621542    |


   *Tabel 5. Hasil Top 10 Rekomendasi Film berdasarkan Judul Spectre dan User ID 404*
   
   |   title                                    |   year  |   vote_count  |   vote_average  |   id    |   mov_genre                                 |   rating_est  |
   |--------------------------------------------|---------|---------------|-----------------|---------|---------------------------------------------|---------------|
   |   Casino Royale                            |   2006  |   3930.0      |   7.3           |   36557 |   [Adventure, Action, Thriller]             |   4.000000    |
   |   Skyfall                                  |   2012  |   7718.0      |   6.9           |   37724 |   [Action, Adventure, Thriller]             |   3.838095    |
   |   The Spy Who Loved Me                     |   1977  |   515.0       |   6.6           |   691   |   [Adventure, Action, Thriller]             |   3.715932    |
   |   Watchmen                                 |   2009  |   2892.0      |   7.0           |   13183 |   [Action, Mystery, Science Fiction]        |   3.682757    |
   |   Safe Haven                               |   2013  |   840.0       |   6.9           |   112949|   [Romance]                                 |   3.662248    |
   |   The Tall Blond Man with One Black Shoe   |   1972  |   58.0        |   6.9           |   12089 |   [Comedy, Mystery]                         |   3.647671    |
   |   On Her Majesty's Secret Service          |   1969  |   464.0       |   6.5           |   668   |   [Adventure, Action, Thriller]             |   3.633573    |
   |   Thunderball                              |   1965  |   572.0       |   6.5           |   660   |   [Adventure, Action, Thriller]             |   3.628781    |
   |   The Living Daylights                     |   1987  |   447.0       |   6.2           |   708   |   [Action, Adventure, Thriller]             |   3.610363    |
   |   To Live and Die in L.A.                  |   1985  |   129.0       |   6.8           |   9846  |   [Action, Crime, Thriller]                 |   3.597934    |


Dari perbandingan hasil rekomendasi pada Tabel 4 dan Tabel 5, dapat dilihat bahwa sistem merekomendasikan film dan estimasi _rating_ yang berbeda untuk tiap pengguna. Hal ini karena riwayat film yang pernah ditonton pengguna berbeda, sehingga sistem akan merekomendasikan berdasarkan data riwayat pengguna tersebut.


## Evaluation

Pada sistem rekomendasi dengan metode _content-based filtering_, digunakan metriks _precision_ sebagai evaluasi. Metriks ini menghitung jumlah rekomendasi yang relevan dibagi dengan jumlah total rekomendasi yang diberikan. Berikut merupakan rumus dari metriks _precision_:

$$Precision = {n \ \text{relevan} \over \text{total} \ n\ \text{items rekomendasi}}$$

Nilai metriks presisi dapat diambil contoh dari hasil top 10 rekomendasi film 'Spectre' dan 'Avengers: Age of Ultron'. Film Spectre memiliki genre '_Action, Adventure, Crime_'. Penulis memberi batasan minimal terdapat dua genre yang sama dari genre film masukan. Pada hasil rekomendasi, dapat dilihat terdapat 8 film yang memiliki genre yang serupa dengan genre film Spectre. Jika dimasukkan pada rumus menjadi:

   $${8 \over 10} *\text{100\\%} = 80\\%$$

Pada film 'Avengers: Age of Ultron', model memberikan 10 film yang relevan dengan genre dari film 'Avengers: Age of Ultron'. Jika dimasukkan pada rumus menjadi:

   $${10 \over 10} *\text{100\\%} = 100\\%$$

Jika diambil rata-rata, dapat dihitung nilai rata-rata presisinya:

   $${(80 + 100) \over 2} *\text{100\\%} = 90\\%$$

Pada metode _hybrid filtering_, metriks yang digunakan adalah metriks RMSE (_Root Mean Square Error_). Metriks ini digunakan untuk mengukur seberapa akurat model dalam memperkirakan nilai sebenarnya. RMSE menghitung perbedaan antara nilai yang diprediksi oleh model dan nilai yang sebenarnya. Artinya, untuk setiap data yang dimiliki, metrik ini akan menghitung selisih antara nilai prediksi dan nilai sebenarnya. Kemudian, akan mengambil rata-rata dari seluruh selisih tersebut dan menghitung akar kuadratnya. Semakin nilai RMSE mendekari 0, semakin baik pula model dalam memperkirakan nilai sebenarnya [4]. Berikut ini merupakan rumus dari metrik RMSE:

$$RMSE = {\sqrt{ \Sigma{(yᵢ - ȳ)^2 \over n}}}$$

Hasil metriks ini dari model yang dikembangkan adalah sebagai berikut: 

*Tabel 6. Hasil Metriks RMSE*
|                | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean   | Std   |
|----------------|--------|--------|--------|--------|--------|--------|-------|
| RMSE (testset) | 0.8988 | 0.9064 | 0.8931 | 0.8984 | 0.8944 | 0.8982 | 0.0047|
| MAE (testset)  | 0.6926 | 0.6954 | 0.6893 | 0.6907 | 0.6871 | 0.6910 | 0.0028|
| Fit time       | 1.14   | 1.15   | 1.16   | 1.69   | 1.81   | 1.39   | 0.30  |
| Test time      | 0.14   | 0.12   | 0.13   | 0.22   | 0.38   | 0.20   | 0.10  |

Model mendapat nilai rata-rata RMSE dari 5 Fold sebesar 0.8982.

## Kesimpulan

Pada metode _content-based filtering_, model memiliki skor presisi rata-rata sebesar 90% dari 2 percobaan rekomendasi film. Pada metode _hybrid filtering_ dengan model SVD mendapatkan skor rata-rata RMSE dari 5 Fold sebesar 0.8982. Nilai ini dapat dibilang cukup baik untuk model yang dikembangkan, karena masih belum menggunakan algoritma _deep learning_ dalam latihannya. Sistem dapat memberikan rekomendasi film yang dapat ditonton selanjutnya oleh pengguna. Metode _content-based filtering_ dapat memberikan rekomendasi film yang serupa dengan masukan film. Metode  _hybrid filtering_ dapat memberikan rekomendasi film berdasarkan kesamaan masukan film dan mengkombinasikan dengan riwayat pengguna sebelumnya.

## Saran

Untuk pengembangan selanjutnya dapat menggunakan metode _deep learning_ untuk mendapatkan skor akurasi RMSE yang lebih kecil, sehingga sistem dapat memberikan rekomendasi yang lebih tepat bagi pengguna.

## Daftar Pustaka

[1] D. Roy and M. Dutta, “A systematic review and research perspective on Recommender Systems,” Journal of Big Data, vol. 9, no. 1, 2022. doi:10.1186/s40537-022-00592-5 

[2] J. Han and M. Kamber, “2 - Getting to Know Your Data,” in Data Mining: Concepts and Techniques, Burlington, MA, US: Elsevier, 2012, pp. 39–82

[3] S. L. Brunton and J. Nathan Kutz, “Singular value decomposition (SVD),” Data-Driven Science and Engineering, pp. 3–46, 2019. doi:10.1017/9781108380690.002 

[4] T. O. Hodson, “Root-mean-square error (RMSE) or mean absolute error (mae): When to use them or not,” Geoscientific Model Development, vol. 15, no. 14, pp. 5481–5487, 2022. doi:10.5194/gmd-15-5481-2022 
