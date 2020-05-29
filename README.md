# DSL

- Untuk memproses data hasil crawling, jalankan prepare crawled data/clean_text.py
- Untuk membuat word embedding, jalankan program pada folder build embedding sesuai urutan nomornya
- Untuk melakukan klasifikasi:
  - Dengan top-n most common : jalankan separate class/classify_with_top_n.py
  - Dengan top-x most common dan top-y most similar : jalankan separate class/classify_with_similarity.py

Contoh penggunaan dari awal sampai akhir:
```
$ python3 "prepare crawled data/clean_text.py"
Please enter the input file name : crawled data/crawled-hp
Please enter the output file name : cleaned data/cleaned-hp
$ python3 "build embedding/01_parse.py" "cleaned data/cleaned-hp" hp_embedding
$ python3 "build embedding/02_clean_s2v.py"
Insert filename : hp_embedding/cleaned-hp.s2v
$ python3 03_fasttext.py ../fastText-0.9.1/fasttext hp_embedding hp_embedding
$ python3 04_export.py hp_embedding/vectors_w2v_300dim.vec hp_embedding/vocab.txt hp_embedding
$ python3 "separate class/classify_with_top_n.py"
Please input domain name (camera/hp/resto) : hp
Please input embedding folder name :hp_embedding
SVM? 1
Concatenate? 0
Insert N (max 1000): 100
```