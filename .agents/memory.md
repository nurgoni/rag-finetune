# Agent Memory

Memori jangka panjang untuk agent yang bekerja di repo ini.
Isinya **keputusan yang sudah diambil** beserta alasannya — supaya tidak diperdebatkan ulang
setiap sesi, dan supaya agent baru tidak "memperbaiki" sesuatu yang sengaja dibuat begitu.

**Cara pakai:**
- Baca seluruh file ini di awal sesi, sebelum menulis kode.
- Setelah mengambil keputusan arsitektur baru, tambahkan entri ADR di §2.
- Kalau sebuah keputusan dibatalkan, **jangan hapus** — ubah statusnya jadi `Superseded by ADR-xxx`.
- Fakta yang bisa dibaca dari kode (struktur folder, nama fungsi) **tidak** ditulis di sini. Yang ditulis di sini adalah *mengapa*-nya.

---

## 1. Status proyek

| | |
|---|---|
| Fase sekarang | **F2 — Data: dataset & corpus** |
| F0 | Selesai, merge ke `main` (PR `feat/f0-project-scaffold`) |
| F1 | Selesai di `develop`: `schema/` (Document, RetrievedDocument, QAItem, GenerationStep, JudgeDecision, VoteTally, Aggregation, QAResult, MetricReport), `config/schema.py` (Config + 8 sub-model + `config_hash()`), `config/loader.py` (`ConfigLoader`, `deep_merge`, `ConfigError`), `configs/base.yaml`, `configs/experiments/nq_sequential.yaml` |
| Verifikasi F1 | `ruff format --check` + `ruff check` bersih; `pytest` 27/27; `python -m rag_finetune.config.loader` mencetak config tervalidasi + hash |
| Berikutnya | F2 — `dataset/loader.py`, `corpus/loader.py`, `utils/jsonl.py`, fixture data kecil di `tests/fixtures/`. Q1/Q2 (dataset & corpus mana) harus ditanyakan ke pengguna sebelum mulai |
| Jumlah fase | F0–F10 aktif, F11+ terkunci (naik dari F0–F7 setelah ADR-009) |
| Branch | `develop` adalah working branch; `main` hanya diperbarui lewat merge `develop` di titik rilis (ADR-017) |
| Terakhir diperbarui | 2026-11-08 |

**Perubahan scope 2026-11-08:** keempat arsitektur pipeline masuk scope aktif (ADR-009).
Yang terdampak di luar ROADMAP: `QAResult` memakai `steps: list[GenerationStep]` sejak F3
(ADR-011), dan ada dua tipe komponen baru — `judger/` dan `aggregator/` (ADR-012).

---

## 2. Architecture Decision Records

### ADR-001 — Repo ini rewrite, bukan fork
**Status:** Diterima · 2026-11-07

FlashRAG dipakai sebagai **referensi konseptual**, bukan basis kode. Setiap komponen ditulis
ulang dari nol dengan nama dan struktur milik repo ini.

**Mengapa:** tujuannya memahami alur end-to-end. Menyalin kode melewatkan justru bagian
yang ingin dipelajari.

**Konsekuensi:** boleh membaca FlashRAG untuk memahami sebuah algoritma; tidak boleh
menyalin-tempel. Kalau sebuah keputusan desain sengaja berbeda dari FlashRAG, catat sebagai ADR.

---

### ADR-002 — Konfigurasi: Pydantic + YAML, bukan dict
**Status:** Diterima · 2026-11-07

Config divalidasi model Pydantic. YAML hanya data masukan. Komponen menerima **sub-model
yang sudah tervalidasi**, bukan seluruh objek config dan bukan `dict`.

**Mengapa:** di FlashRAG, config adalah `dict` yang dilewatkan ke semua komponen, jadi tidak
ada cara mengetahui apa yang sebenarnya dibutuhkan sebuah retriever selain membaca
seluruh implementasinya. Pydantic membuat kontraknya eksplisit dan menangkap salah ketip
saat start, bukan 20 menit kemudian saat evaluasi.

**Konsekuensi:** `extra="forbid"` di semua model config — field tak dikenal adalah error.
Menambah field berarti mengubah schema, dan itu memang disengaja.

---

### ADR-003 — Generator default: Ollama via API OpenAI-compatible
**Status:** Diterima · 2026-11-07

Backend generator pertama adalah klien HTTP ke endpoint OpenAI-compatible, dengan base URL
default `http://localhost:11434/v1` (Ollama).

**Mengapa:** jalan di laptop tanpa GPU, gratis, tanpa kunci API. Yang lebih penting:
permukaan API yang sama nanti melayani provider cloud hanya dengan ganti base URL + auth.

**Konsekuensi:** class-nya dinamai `OpenAICompatibleGenerator`, **bukan** `OllamaGenerator`.
Rencana ke depan (urutan sudah disepakati): Ollama → provider LLM via API → vLLM.
vLLM juga menyediakan server OpenAI-compatible, jadi kemungkinan besar memakai class yang sama.

---

### ADR-004 — Scope fase ini: inference saja, bukan fine-tuning
**Status:** Diterima · 2026-11-07

Meski repo bernama `rag-finetune`, F0–F10 **tidak mengandung training loop sama sekali**.

**Mengapa:** fine-tuning tanpa harness evaluasi yang tepercaya hanya menghasilkan angka
yang tidak bisa dipercaya. Evaluasi harus benar lebih dulu.

**Konsekuensi:** agent tidak boleh menambahkan kode training, `Trainer`, atau dependensi
`peft`/`trl` sebelum F10 selesai. Nama repo mencerminkan tujuan akhir, bukan scope sekarang.
Lihat ROADMAP F11 item 13–14.

**Catatan 2026-11-08:** nomor fase bergeser dari F7 ke F10 karena ADR-009 menambah tiga fase
arsitektur. Batasannya tidak berubah — tetap "setelah CLI & dokumentasi tuntas".

---

### ADR-005 — Walking skeleton sebelum komponen nyata
**Status:** Diterima · 2026-11-07

Fase F3 membangun pipeline lengkap memakai `RandomRetriever` dan `EchoGenerator` sebelum
retriever/generator sungguhan ada.

**Mengapa:** alur end-to-end jadi terbukti dan bisa dites tanpa jaringan, tanpa GPU, tanpa
build index. Mengejar bug orkestrasi jauh lebih mudah saat tidak ada model yang terlibat.

**Konsekuensi:** kedua komponen palsu itu **tetap dipertahankan** setelah ada yang nyata.
`RandomRetriever` menjadi lower bound untuk membandingkan retriever, dan keduanya menjaga
unit test tetap cepat. Jangan dihapus sebagai "kode mati".

---

### ADR-006 — BM25 sebelum dense
**Status:** Diterima · 2026-11-07

Retriever nyata pertama adalah BM25 (`bm25s`), baru kemudian dense (FAISS).

**Mengapa:** BM25 tidak butuh GPU, tidak butuh model embedding, dan build index-nya detik
bukan jam. Jadi jalur `index → retrieve → evaluate` bisa divalidasi murah dulu.
`bm25s` dipilih di atas Pyserini karena Pyserini butuh JVM.

**Konsekuensi:** tipe retriever pertama yang masuk registry adalah `bm25`.

---

### ADR-007 — `QAResult` membawa seluruh jejak satu query
**Status:** Diterima · 2026-11-07

Satu objek `QAResult` memuat `QAItem`, dokumen terambil, prompt final, output LLM mentah,
dan prediksi yang sudah di-parse.

**Mengapa:** kalau skor jelek, pertanyaan pertama selalu "ini salah di retrieval, prompt,
atau generasi?". Menyimpan seluruh jejak membuat pertanyaan itu terjawab dengan membaca
satu baris JSONL, tanpa perlu menambah logging atau menjalankan ulang.

**Konsekuensi:** `predictions.jsonl` berukuran besar. Itu konsekuensi yang diterima;
debuggability menang atas ukuran disk untuk ukuran eksperimen di repo ini.

---

### ADR-008 — Dependensi berat adalah extras opsional
**Status:** Diterima · 2026-11-07

`faiss`, `torch`, `sentence-transformers` masuk extras `[dense]`; `bm25s` masuk `[sparse]`.
Install inti hanya butuh Pydantic, PyYAML, httpx, typer.

**Mengapa:** seseorang harus bisa membaca, meng-install, dan menjalankan test pipeline
tanpa mengunduh PyTorch.

**Konsekuensi:** import dependensi berat terjadi **di dalam** method, bukan di level modul,
dan ketidakhadirannya memunculkan pesan error yang menyebutkan extras mana yang perlu di-install.
Ini satu-satunya pengecualian yang diizinkan dari aturan "import di atas file".

---

### ADR-009 — Keempat arsitektur pipeline masuk scope, diurutkan per biaya implementasi
**Status:** Diterima · 2026-11-08 · **Memperluas** ADR-005

Scope naik dari satu pipeline (`Sequential`) menjadi keempat arsitektur FlashRAG, masing-masing
dengan **satu algoritma termudah** lebih dulu: sequential/Standard RAG (F3) → loop/Iter-RetGen
(F7) → branching/majority-vote (F8) → conditional/LLM-judger (F9).

**Mengapa arsitekturnya diperluas:** pelajaran utama RAG ada di **pola orkestrasi**, bukan di
jumlah algoritma. Keempat pola memakai retriever dan generator yang sama persis; bedanya hanya
berapa kali dan dalam urutan apa keduanya dipanggil. Satu pipeline tidak memperlihatkan itu.

**Mengapa urutannya begitu** — kriteria yang dipakai, berurutan: (1) berapa abstraksi baru yang
dibutuhkan, (2) apakah butuh kemampuan model di luar "prompt masuk, teks keluar", (3) apakah
kondisi berhentinya statis. Hasilnya: loop = nol abstraksi baru; branching = satu fungsi murni;
conditional = satu ABC baru.

**Jebakan yang harus dihindari agent:** conditional *terdengar* paling sederhana ("kalau butuh
retrieval, retrieve") dan akan terus menggoda untuk dikerjakan lebih awal. Kriterianya adalah
biaya implementasi, bukan kerumitan konsep. Jangan ubah urutannya.

**Konsekuensi:** ROADMAP jadi F0–F10. Setiap arsitektur baru wajib punya **uji reduksi** —
konfigurasi yang membuatnya runtuh jadi sequential dan menghasilkan hasil identik
(`n_iterations=1`, `top_k=1`, `AlwaysRetrieveJudger`). Ini pertahanan terkuat melawan bug
orkestrasi, dan wajib ada di tiap fase.

---

### ADR-010 — Algoritma yang butuh logprob ditunda, tapi polanya disiapkan
**Status:** Diterima · 2026-11-08

REPLUG (branching) dan FLARE (loop) **tidak** diimplementasikan sekarang. Sebagai gantinya,
posisi mereka diisi algoritma yang polanya identik tapi hanya butuh teks:
majority-vote untuk REPLUG, iterasi tetap untuk FLARE.

**Mengapa:** keduanya butuh probabilitas token — REPLUG untuk membobot ensemble per-dokumen,
FLARE untuk mendeteksi kalimat ber-keyakinan rendah. Ollama tidak menjamin akses logprob
lewat API OpenAI-compatible-nya.

**Mengapa tidak sekadar dilewati:** majority-vote adalah REPLUG tanpa pembobotan. Fan-out,
pemetaan balik hasil, dan titik agregasinya identik. Jadi menambahkan REPLUG nanti adalah
**satu aggregator baru**, bukan pipeline baru.

**Konsekuensi:** `BaseGenerator` tetap `generate(prompts) -> list[str]`. Jangan tambahkan
parameter `logprobs` sekarang — itu antisipasi masa depan yang dilarang AGENTS.md §9, dan
API-nya kemungkinan berubah saat vLLM benar-benar masuk. Keduanya terbuka kuncinya oleh
backend vLLM (ROADMAP F11 item 7).

---

### ADR-011 — `QAResult.steps` sebagai daftar, sejak F3
**Status:** Diterima · 2026-11-08 · **Menggantikan** bentuk datar `QAResult` di PRD v0.1

`QAResult` menyimpan `steps: list[GenerationStep]`, bukan satu `retrieved`/`prompt`/`raw_output`
yang datar. `SequentialPipeline` di F3 sudah memakai bentuk ini walau selalu mengisi satu langkah.

**Mengapa:** loop dan branching memecahkan asumsi "satu query = satu kali retrieve + satu kali
generate". Loop juga **mengubah query** tiap iterasi, jadi `query` harus hidup di tingkat langkah
— tanpa itu, hasil antara tidak bisa dibaca dan G5 gagal justru di pipeline yang paling butuh.

**Mengapa sejak F3 dan bukan nanti:** alternatifnya memigrasikan format `predictions.jsonl`
tiga kali, di F7, F8, dan F9, berikut semua test dan metrik yang membacanya.

**Ini bukan pelanggaran "jangan bikin abstraksi untuk kebutuhan yang belum ada"** — kebutuhannya
tertulis di PRD §6.4 dan sudah disepakati masuk scope, bukan dugaan. Bedakan: antisipasi
terlarang adalah untuk kebutuhan yang *mungkin* muncul.

**Konsekuensi:** `QAResult` punya property turunan `last_step`, `retrieved`, `prompt`,
`raw_output` yang menunjuk langkah terakhir, supaya kode sequential tetap ringkas. Ini
**satu-satunya** logika yang diizinkan hidup di dalam model `schema/`.

---

### ADR-012 — `judger/` dan `aggregator/` jadi tipe komponen tersendiri
**Status:** Diterima · 2026-11-08

Dua direktori komponen baru, masing-masing dengan registry sendiri, sejajar `retriever/`
dan `generator/`.

**Mengapa dipisah dan tidak ditaruh di dalam pipeline-nya:** keduanya adalah titik variasi
nyata. Conditional punya minimal tiga judger (LLM, SKR, always/never); branching punya minimal
dua aggregator (majority vote, bobot logprob). Menanamkannya di dalam pipeline berarti
menyalin seluruh pipeline hanya untuk mengganti satu keputusan.

**Asimetri yang disengaja:** `judger/` berisi **class** turunan `BaseJudger` — ia punya
dependency (generator, atau index kNN untuk SKR) dan state. `aggregator/` berisi **fungsi murni**
— tidak ada state, tidak ada dependency. Keduanya memakai registry, tapi bentuknya beda sesuai
aturan Class vs Function di AGENTS.md §3. Jangan "diseragamkan" jadi sama-sama class.

**Konsekuensi:** `pipeline/` boleh import keduanya. Keduanya tidak boleh import `pipeline/`.

---

### ADR-013 — Metrik biaya adalah warga kelas satu
**Status:** Diterima · 2026-11-08 · **Mekanismenya direvisi oleh ADR-014**

`num_llm_calls` adalah field pada `QAResult`, dan `avg_llm_calls` selalu ada di `metrics.json` —
sejak F6, saat nilainya masih selalu 1.0.

**Mengapa:** branching memanggil LLM `top_k` kali dan loop `n_iterations` kali. Melaporkan EM/F1
tanpa biaya membuat branching tampak unggul tanpa memperlihatkan bahwa ia lima kali lebih mahal.
Perbandingan antar-arsitektur (PRD G8) tidak bermakna tanpa angka ini.

**Konsekuensi:** satu panggilan `generate()` berisi `n` prompt dihitung `n`, bukan 1 — yang
diukur adalah kerja model, bukan jumlah request HTTP. Termasuk panggilan judger.

**Catatan:** versi awal ADR ini menyuruh *pipeline* menghitung sendiri. ADR-014 memindahkannya
ke boundary generator.

---

### ADR-014 — `num_llm_calls` dihitung di boundary generator, bukan oleh pipeline
**Status:** Diterima · 2026-11-08 · **Merevisi** mekanisme ADR-013

`CountingGenerator` membungkus `BaseGenerator` apa pun dan menghitung prompt yang benar-benar
lewat. Pipeline **tidak** menaikkan counter sendiri dan tidak punya akses untuk mengubahnya.

**Mengapa:** angka biaya yang dilaporkan oleh pihak yang sedang dinilai bukan angka yang bisa
diaudit. Dengan penghitungan di dalam pipeline, pipeline yang lupa menambah counter akan
terlihat **lebih murah** daripada yang teliti — insentifnya terbalik. Lebih buruk lagi,
jalur yang jarang terpakai (retry, cabang error, panggilan judger) justru yang paling mungkin
terlewat dihitung.

**Mengapa decorator dan bukan inspeksi log:** menghitung di satu titik yang dilewati semua
panggilan membuat angkanya benar secara konstruksi, bukan secara kedisiplinan.

**Konsekuensi:** pipeline menerima generator yang sudah terbungkus; pembungkusan terjadi saat
perakitan komponen, bukan di dalam pipeline. Reviewer boleh menolak PR pipeline yang mengandung
`num_llm_calls +=`. Ini juga satu-satunya alasan sah `BaseGenerator` punya implementasi
pembungkus — bukan backend baru.

---

### ADR-015 — Kontrak determinisme hanya mengikat generator deterministik
**Status:** Diterima · 2026-11-08

Uji reduksi (ADR-009) dan uji reprodusibilitas (NFR-5) dijalankan dengan `EchoGenerator` +
`RandomRetriever` ber-seed. Dengan LLM nyata, perbandingan memakai metrik agregat, **bukan**
kesamaan per-item.

**Mengapa:** "hasil identik" adalah inti uji reduksi, tapi properti itu tidak bisa dijamin
oleh LLM mana pun — bahkan `temperature: 0` tidak bit-identik antar-versi model. Kalau uji
reduksi dijalankan terhadap Ollama, ia akan flaky, lalu akan di-`skip`, lalu hilang. Test
terpenting di repo ini tidak boleh bergantung pada hal yang tidak bisa dijamin.

**Mengapa `seed` jadi field config wajib:** `RandomRetriever` dipakai di hampir semua test
pipeline. Tanpa seed, seluruh uji reduksi flaky karena alasan yang sama sekali tidak ada
hubungannya dengan orkestrasi yang sedang diuji.

**Konsekuensi:** `RunConfig.seed` wajib dengan default eksplisit; tidak ada pemanggilan
`random` global di mana pun. `temperature` dan `seed` masuk `provenance` supaya terlihat
apakah sebuah run memang berhak diharapkan reprodusibel.

---

### ADR-016 — `provenance` harus cukup untuk membedakan dua run mana pun
**Status:** Diterima · 2026-11-08

`provenance` di `metrics.json` memuat hash config yang sudah di-resolve **plus** parameter
pipeline (`n_iterations`, `top_k`, `aggregator.type`, `judger.type`), `temperature`, `seed`,
dan nama + hash template prompt.

**Mengapa:** versi awal (FR-8.7 lama) hanya mencatat tipe pipeline, tipe retriever, nama model,
dan jumlah item. Dengan itu, dua run loop `n_iterations: 2` dan `n_iterations: 5` menghasilkan
`provenance` identik dengan angka berbeda — dan `compare`, satu-satunya mekanisme untuk tujuan
G8, menampilkan dua baris yang tidak bisa dibedakan. Lubang ini ditemukan saat mengaudit
dokumen sendiri, sebelum ada kode.

**Uji kelayakan:** kalau dua run bisa menghasilkan `provenance` identik tapi angka berbeda,
`provenance`-nya belum lengkap. Ini kriteria yang bisa diperiksa, bukan selera.

**Mengapa template prompt di-hash:** prompt adalah data (ADR-002 semangatnya sama), dan
mengubah satu kata di template bisa menggeser EM beberapa poin. Tanpa hash-nya, run lama
tidak bisa dibandingkan dengan run baru secara sah.

---

### ADR-017 — `develop` adalah working branch, `main` hanya rilis
**Status:** Diterima · 2026-11-08

`main` tidak pernah menerima commit langsung atau merge branch fitur. Seluruh pengerjaan fase
ROADMAP (F1 dst.) terjadi di `develop` atau branch fitur yang dicabang dari `develop`.
`main` diperbarui hanya lewat merge `develop` di titik rilis/milestone.

**Mengapa:** `main` harus selalu mencerminkan keadaan yang terbukti benar (gate §6
CONTRIBUTING sudah hijau), supaya bisa dirujuk atau di-deploy kapan saja tanpa memeriksa
commit mana yang "aman". Pekerjaan fase yang sedang berjalan butuh tempat untuk commit
bertahap — termasuk commit yang belum lolos gate — tanpa mengganggu itu.

**Konsekuensi:** lihat [CONTRIBUTING.md §3.1](../CONTRIBUTING.md#31-strategi-branch-adr-017)
untuk mekanika alurnya. F0 sudah merge ke `main` sebelum ADR ini ditulis; mulai F1, seluruh
fase dikerjakan di `develop`.

---

## 3. Pelajaran & jebakan

Diisi selama pengerjaan. Format: gejala → sebab → cara hindari.

### L1 — Klaim auditabilitas gagal justru di dokumennya sendiri
**2026-11-08, saat mengaudit PRD v0.2 sebelum ada kode**

**Gejala:** PRD menyatakan tiap arsitektur "bisa diaudit" dan menjadikan uji reduksi sebagai
test terpenting, tapi enam hal yang membuat klaim itu bisa dieksekusi tidak ada: generator
mana yang dipakai saat uji reduksi, `seed` yang disebut NFR-5 tapi tak pernah jadi field,
`provenance` yang tidak merekam parameter pipeline, tally voting, syarat hitung-ulang offline,
dan `num_llm_calls` yang dilaporkan sendiri oleh pihak yang dinilai.

**Sebab:** kata "bisa diaudit" dipakai tanpa definisi. Selama tidak ada pertanyaan konkret
yang harus dijawab oleh artefak, setiap dokumen akan lulus audit terhadap dirinya sendiri.

**Cara hindari:** jangan menulis properti kualitas sebagai kata sifat. Tulis sebagai
**pertanyaan yang harus bisa dijawab oleh artefak**, lalu sebutkan artefak mana yang
menjawabnya (PRD §6.5). Kalau sebuah baris di dokumen tidak bisa dipetakan ke artefak,
baris itu aspirasi, bukan kebutuhan.

**Pola yang berulang di keenam lubang:** semuanya adalah properti yang diasumsikan "akan
terjadi dengan sendirinya kalau kodenya rapi". Tidak ada yang begitu. Determinisme, biaya,
dan keterlacakan harus dipaksa oleh satu titik di konstruksi — ber-seed, terbungkus counter,
tersimpan — bukan oleh kedisiplinan tiap penulis pipeline.

---

### L2 — Nama modul yang menyadur stdlib gagal secara tersembunyi
**2026-11-08, F0**

**Gejala:** `utils/logging.py` lolos `ruff check` dan lolos `pytest` (7/7 hijau), tapi
`python -m doctest src/rag_finetune/utils/logging.py` meledak dengan
`AttributeError: module 'logging' has no attribute 'basicConfig'`.

**Sebab:** saat file bernama `logging.py` dimuat sebagai modul top-level, ia terdaftar di
`sys.modules` dengan nama `logging` dan menimpa stdlib. `import logging` di dalamnya lalu
resolve ke dirinya sendiri. Di dalam paket, absolute import tetap menemukan stdlib dengan
benar — itulah yang membuat bug ini tidak terlihat dari unit test.

**Cara hindari:** jangan pernah memberi nama modul yang sama dengan modul stdhlib —
`logging.py`, `io.py`, `json.py`, `types.py`, `copy.py`, `select.py`. Dinamai apa yang
dilakukannya: `logging_config.py`, `jsonl.py`. Sudah masuk tabel review Zen di AGENTS.md §2.

**Yang membuatnya ketangkap:** menjalankan doctest dan blok `__main__` sebagai verifikasi
terpisah, bukan hanya `pytest`. Langkah 5 Zen Workflow ("sertakan contoh penggunaan") bukan
formalitas dokumentasi — ia jalur eksekusi berbeda yang menangkap kelas bug yang tidak
terlihat dari test suite. Deliverable F2 `utils/io.py` sudah ikut di-rename jadi
`utils/jsonl.py` sebelum ditulis.

---

## 4. Pertanyaan terbuka

| # | Pertanyaan | Dibutuhkan saat |
|---|---|---|
| Q1 | Dataset mana yang jadi benchmark utama — NQ atau HotpotQA? NQ lebih sederhana (single-hop) untuk MVP | F2 |
| Q2 | Corpus apa yang dipakai untuk pengembangan? Subset Wikipedia pra-index FlashRAG (1 juta dokumen) atau subset lebih kecil yang dibangun sendiri? | F2 |
| Q3 | Model Ollama apa yang jadi default di `base.yaml`? Perlu yang cukup patuh untuk QA ekstraktif | F4 |
| Q4 | Model embedding default — `intfloat/e5-base-v2` (seperti FlashRAG) atau `BAAI/bge-small-en-v1.5` yang lebih ringan? | F5b |
| Q5 | Prediksi diambil dari output LLM dengan cara apa — ambil apa adanya, atau parse pola seperti `Answer:`? Mempengaruhi skor EM secara signifikan | F6 |
| Q6 | `n_iterations` default untuk Iter-RetGen — 2 atau 3? Dan `MAX_ITERATIONS` dipatok berapa? Biayanya linear terhadap angka ini | F7 |
| Q7 | Di loop, dokumen tiap iterasi **diakumulasi** atau **diganti**? Iter-RetGen asli mengganti; akumulasi memberi konteks lebih kaya tapi prompt membengkak tiap iterasi | F7 |
| Q8 | Branching memakai `top_k` yang sama dengan sequential, atau `top_k` sendiri yang lebih kecil? `top_k=10` berarti 10× panggilan LLM | F8 |
| Q9 | Kalau `MajorityVoteAggregator` menemukan `top_k` jawaban yang semuanya berbeda (tidak ada mayoritas sama sekali), ambil skor retrieval tertinggi atau tandai sebagai abstain? | F8 |

Agent **tidak boleh** menebak jawaban pertanyaan-pertanyaan ini. Tanyakan ke pengguna
saat fase yang bersangkutan dimulai, lalu catat jawabannya di sini sebagai ADR.

---

## 5. Glosarium

| Istilah | Arti di repo ini |
|---|---|
| **Corpus** | Kumpulan dokumen yang dicari. Tidak sama dengan dataset |
| **Dataset** | Kumpulan pertanyaan + jawaban gold untuk evaluasi |
| **Index** | Artefak di disk hasil pra-proses corpus (FAISS atau BM25) |
| **Run** | Satu eksekusi pipeline atas satu config; punya `run_id` |
| **Hasil antara** | Dokumen terambil + prompt + output mentah yang disimpan per item |
| **Walking skeleton** | Pipeline lengkap dengan komponen palsu; buktikan alur dulu (ADR-005) |
| **Judger** | Komponen yang memutuskan perlu-tidaknya retrieval, dipakai pipeline conditional (ADR-012) |
| **Aggregator** | Fungsi murni yang menggabungkan banyak kandidat jawaban jadi satu, dipakai pipeline branching (ADR-012) |
| **Langkah / `GenerationStep`** | Satu putaran retrieve+generate. Sequential punya 1, loop punya 1 per iterasi, branching 1 per cabang (ADR-011) |
| **Fan-out** | Pola branching: satu item jadi `top_k` prompt, hasilnya dipetakan balik ke item asalnya |
| **Uji reduksi** | Konfigurasi yang membuat pipeline kompleks runtuh jadi sequential dan menghasilkan hasil identik. Wajib di F7–F9 (ADR-009) |
| **Refiner** | Komponen yang memadatkan konteks sebelum masuk prompt (future scope) |
