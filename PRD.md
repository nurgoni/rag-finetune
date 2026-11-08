# PRD — rag-finetune

**Status:** Draft disetujui · **Versi:** 0.2 · **Terakhir diubah:** 2026-11-08
**Pemilik:** @nurgoni

> **Perubahan v0.2:** scope dinaikkan dari satu pipeline (`Sequential`) menjadi
> **keempat arsitektur pipeline** FlashRAG — sequential, loop, branching, conditional —
> masing-masing diisi **satu algoritma termudah** lebih dulu. Lihat §6.4 dan ADR-009.

---

## 1. Ringkasan

`rag-finetune` adalah **rewrite terdidik** dari [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG):
sebuah toolkit RAG modular yang ditulis ulang dari nol agar alur end-to-end-nya
bisa dipahami, dijalankan, dan dimodifikasi oleh satu orang.

FlashRAG sangat lengkap (23 algoritma, 36 dataset, 4 arsitektur pipeline), tapi kelengkapan itu
membuat alurnya sulit diikuti: banyak indireksi lewat `dict` config, factory dinamis, dan
registry yang membuat jawaban atas pertanyaan *"apa yang sebenarnya terjadi antara pertanyaan
masuk dan jawaban keluar?"* tersebar di belasan file.

Repo ini menukar **jumlah algoritma** dengan kejelasan, tapi **tidak** menukar cakupan
arsitekturnya. Keempat pola orkestrasi RAG diimplementasikan — karena justru di situ
pelajarannya: bedanya Standard RAG, Iter-RetGen, SuRe, dan Adaptive-RAG bukan di
komponennya (retriever dan generator sama saja), melainkan **di cara komponen itu
dirangkai**. Satu algoritma termudah per arsitektur cukup untuk menunjukkan polanya;
algoritma sisanya jadi latihan mengisi pola yang sudah ada.

## 2. Masalah

1. **Alur tidak terlihat.** Di FlashRAG, konfigurasi dilewatkan sebagai `dict` ke setiap komponen, sehingga tidak ada kontrak eksplisit tentang apa yang dibutuhkan sebuah retriever.
2. **Tidak bisa dijalankan bertahap.** Sulit mengeksekusi hanya tahap retrieval tanpa menyiapkan seluruh stack.
3. **Barrier eksperimen tinggi.** Mengganti generator berarti memahami hierarki class yang mengasumsikan GPU dan `transformers`.
4. **Sulit di-debug.** Kalau skor evaluasi jelek, tidak jelas apakah penyebabnya retrieval, prompt, atau generasi.

## 3. Tujuan

| # | Tujuan | Ukuran keberhasilan |
|---|---|---|
| G1 | Satu perintah menjalankan RAG end-to-end | `rag-finetune run configs/experiments/nq_sequential.yaml` menghasilkan file skor |
| G2 | Setiap layer bisa dijalankan sendiri | Setiap modul utama punya blok `__main__` yang jalan tanpa layer di atasnya |
| G3 | Jalan di laptop tanpa GPU | Generator default = Ollama lokal; retrieval dense pakai index FAISS CPU |
| G4 | Kontrak antar-komponen eksplisit | Semua data lintas-layer berupa model Pydantic; semua komponen yang bisa ditukar punya ABC |
| G5 | Observabilitas per tahap | Hasil antara (dokumen terambil, prompt final, output mentah) tersimpan per query |
| G6 | Komponen baru murah ditambahkan | Retriever baru = 1 file + 1 entri registry + 1 test, tanpa menyentuh pipeline |
| G7 | Keempat pola orkestrasi RAG terwakili | `sequential`, `loop`, `branching`, `conditional` masing-masing punya minimal 1 algoritma yang jalan dan terukur |
| G8 | Pola orkestrasi bisa dibandingkan langsung | Keempatnya jalan di dataset & retriever yang sama; `metrics.json` membuat trade-off skor vs jumlah panggilan LLM terlihat |

## 4. Non-tujuan (fase ini)

| Non-tujuan | Alasan |
|---|---|
| **Fine-tuning / training loop** | Meski repo bernama `rag-finetune`, fase ini **hanya inference RAG**. Training masuk future scope (§11) |
| Parity 23 algoritma FlashRAG | Target: **4 arsitektur × 1 algoritma termudah**, bukan 23 algoritma. Sisanya mengisi pola yang sudah ada |
| Algoritma yang butuh akses logprob token | REPLUG dan FLARE butuh probabilitas token; Ollama tidak menjaminnya. Ditunda sampai ada backend vLLM (§11, ADR-010) |
| Algoritma yang butuh model khusus | Self-RAG butuh model yang di-fine-tune dengan reflection token; Adaptive-RAG butuh classifier terlatih. Di luar scope inference (§11) |
| Web UI | CLI cukup untuk eksperimen |
| Distributed / multi-node | Satu mesin |
| Mengalahkan performa FlashRAG | Target: jelas, bukan cepat |
| Dukungan 36 dataset | 2 dataset cukup untuk membuktikan alur |

## 5. Pengguna & use case

**Persona tunggal: ML engineer yang bereksperimen sendiri.**

| ID | Use case | Alur |
|---|---|---|
| UC1 | Jalankan baseline RAG di dataset QA | Tulis YAML → `run` → dapat EM/F1 + hasil antara |
| UC2 | Bandingkan dense vs BM25 | Dua YAML beda `retriever.type` → bandingkan skor |
| UC3 | Debug retrieval buruk | Buka hasil antara, lihat dokumen yang terambil untuk query yang salah jawab |
| UC4 | Ganti model generator | Ubah satu field YAML (`generator.model`) |
| UC5 | Index corpus sendiri | `rag-finetune index` atas file JSONL milik sendiri |
| UC6 | Tambahkan metrik baru | Tulis satu fungsi, daftarkan, muncul di laporan |
| UC7 | Bandingkan 4 arsitektur pipeline | Satu YAML, `--pipeline` berbeda → `compare` menampilkan skor vs biaya berdampingan |
| UC8 | Pahami beda pola orkestrasi | Baca `predictions.jsonl`: `steps` memperlihatkan query tiap iterasi (loop) atau tiap cabang (branching) |

## 6. Arsitektur

### 6.1 Alur end-to-end

```
                   configs/*.yaml
                        │
                   ┌────▼─────┐
                   │  Config  │  pydantic: validasi + default
                   └────┬─────┘
       ┌────────────────┼──────────────────┐
       │                │                  │
 ┌─────▼─────┐   ┌──────▼──────┐    ┌──────▼──────┐
 │  Dataset  │   │  Retriever  │    │  Generator  │
 │  (QAItem) │   │ dense/bm25  │    │   Ollama    │
 └─────┬─────┘   └──────┬──────┘    └──────┬──────┘
       │                │                  │
       └────────────┬───┴──────────────────┘
                    │
            ┌───────▼─────────────────────────────────┐
            │              Pipeline                   │
            │  orkestrasi komponen; 4 pola (§6.4):    │
            │  sequential · loop · branching · cond.  │
            └───────┬─────────────────────────────────┘
                    │  list[QAResult]
            ┌───────▼────────┐
            │   Evaluator    │  EM, F1, Recall@k, biaya
            └───────┬────────┘
                    │
        results/<run_id>/{metrics.json, predictions.jsonl}
```

**Komponen tetap, orkestrasi berubah.** Ini inti arsitekturnya: keempat pola di §6.4
memakai `Retriever`, `PromptTemplate`, dan `Generator` yang sama persis. Yang berbeda hanya
*berapa kali* dan *dalam urutan apa* ketiganya dipanggil. Konsekuensi desainnya:
pipeline boleh memanggil komponen, tapi **tidak boleh** mengandung logika retrieval
atau pemanggilan HTTP sendiri.

### 6.2 Tanggung jawab modul

| Modul | Tanggung jawab | Tidak bertanggung jawab atas |
|---|---|---|
| `config/` | Baca & validasi YAML jadi objek tipe-aman | Membuat komponen |
| `schema/` | Model data lintas-layer | Logika apa pun |
| `dataset/` | JSONL → `list[QAItem]` | Retrieval |
| `corpus/` | JSONL → `Iterator[Document]` | Embedding |
| `index/` | Bangun & muat artefak index | Scoring saat query |
| `retriever/` | Query → `list[RetrievedDocument]` | Format prompt |
| `prompt/` | Query + dokumen → string prompt | Memanggil LLM |
| `generator/` | Prompt → teks | Retrieval, parsing jawaban |
| `judger/` | Query → perlu retrieval atau tidak | Melakukan retrieval itu sendiri |
| `aggregator/` | Banyak kandidat jawaban → satu jawaban | Menghasilkan kandidat |
| `pipeline/` | Orkestrasi urutan tahap | Implementasi tahap |
| `evaluation/` | Prediksi + gold → skor | Menjalankan pipeline |

### 6.3 Kontrak data inti

Semua model Pydantic, immutable (`frozen=True`) kecuali disebut lain:

```python
Document            : id: str, text: str, title: str | None, metadata: dict[str, str]
RetrievedDocument   : document: Document, score: float, rank: int
QAItem              : id: str, question: str, golden_answers: list[str]

GenerationStep      : index: int, label: str, query: str,
                      retrieved: list[RetrievedDocument],
                      prompt: str, raw_output: str, answer: str
JudgeDecision       : needs_retrieval: bool, reason: str, raw_output: str | None
VoteTally           : answer: str, normalized: str, votes: int, best_score: float
Aggregation         : strategy: str, tally: list[VoteTally],
                      resolved_by: Literal["majority", "tie_break"]

QAResult            : item: QAItem, steps: list[GenerationStep],
                      prediction: str, judge: JudgeDecision | None,
                      aggregation: Aggregation | None,
                      num_llm_calls: int, elapsed_s: float, error: str | None
MetricReport        : run_id: str, scores: dict[str, float], num_items: int,
                      avg_llm_calls: float, provenance: dict[str, str]
```

Tiga field opsional pada `QAResult` adalah **jejak audit khas per arsitektur** (§6.5.1):
`judge` hanya terisi di conditional, `aggregation` hanya di branching, dan `steps` yang
panjangnya > 1 dengan `query` berbeda hanya muncul di loop dan branching.
`resolved_by` ada supaya tie-break tidak bisa menyamar sebagai konsensus.

### 6.3.1 Mengapa `steps: list[GenerationStep]`

Pada v0.1 `QAResult` menyimpan satu `retrieved` + satu `prompt` + satu `raw_output`, karena
hanya ada `SequentialPipeline`. Begitu loop dan branching masuk, asumsi "satu query =
satu kali retrieve + satu kali generate" **pecah**:

| Arsitektur | Isi `steps` |
|---|---|
| sequential | tepat 1 langkah |
| loop | 1 langkah per iterasi, `label="iter-0"`, `"iter-1"`, … |
| branching | 1 langkah per cabang, `label="doc-0"`, `"doc-1"`, … + 1 langkah agregasi |
| conditional | 1 langkah; `judge` terisi. Cabang no-retrieval → `retrieved` kosong |

`label` dan `query` ada di tingkat langkah karena loop **mengubah query** tiap iterasi —
tanpa itu, hasil antara tidak bisa dibaca (G5).

Agar kode sequential tidak jadi bertele-tele, `QAResult` menyediakan property turunan
`last_step`, `retrieved`, `prompt`, dan `raw_output` yang menunjuk langkah terakhir.
Ini satu-satunya logika yang diizinkan hidup di dalam model schema.

`num_llm_calls` adalah field pertama kelas, bukan metrik sampingan: branching dan loop
menukar **biaya** dengan akurasi, dan perbandingan antar-arsitektur (G8) tidak bermakna
tanpa angka ini.

## 6.4 Empat arsitektur pipeline

### 6.4.1 Kriteria "paling mudah diimplementasikan"

Urutan pengerjaan ditentukan oleh tiga kriteria, diperiksa berurutan:

1. **Berapa abstraksi baru yang dibutuhkan?** Pola yang hanya merangkai ulang komponen yang
   sudah ada lebih murah daripada pola yang menuntut ABC baru.
2. **Apakah butuh kemampuan model di luar "prompt masuk, teks keluar"?** Butuh logprob token
   atau bobot yang di-fine-tune = langsung ditunda, bukan sekadar belakangan (ADR-010).
3. **Apakah kondisi berhentinya statis?** Iterasi dengan jumlah tetap lebih mudah
   daripada iterasi yang berhenti berdasarkan isi output.

### 6.4.2 Urutan yang dipilih

| # | Arsitektur | Algoritma pertama | Abstraksi baru | Mengapa di posisi ini |
|---|---|---|---|---|
| 1 | **sequential** | Standard RAG | `BasePipeline`, `PromptTemplate` | Baseline. Semua pola lain adalah variasi darinya |
| 2 | **loop** | Iter-RetGen (iterasi tetap) | *tidak ada* | Hanya `for` di sekeliling sequential. Jumlah iterasi dari config → tidak ada logika kondisi berhenti |
| 3 | **branching** | Per-doc generate + majority vote | `aggregator/` (fungsi murni) | Reuse `generate(prompts)` yang sudah batch → fan-out gratis. Tambahannya cuma voting |
| 4 | **conditional** | LLM-prompted judger | `BaseJudger` (ABC) | Satu-satunya yang menambah **tipe komponen baru** yang bisa ditukar, sejajar retriever/generator |

**Mengapa loop sebelum branching:** Iter-RetGen tidak menambah abstraksi apa pun — ia
memanggil ulang jalur sequential dengan query yang sudah diperkaya jawaban iterasi
sebelumnya. Branching menambah satu hal yang kelihatan kecil tapi nyata: fan-out memecah
pemetaan 1-item-1-generasi, jadi hasil harus dipetakan balik ke item asalnya.

**Mengapa conditional terakhir meski konsepnya paling sepele.** Idenya satu kalimat
("kalau butuh retrieval, retrieve"), tapi implementasinya menuntut `BaseJudger` + registry
+ routing ke dua sub-pipeline. Konsep sederhana ≠ implementasi murah — dan kriteria di
§6.4.1 mengurutkan berdasarkan implementasi.

**Mengapa majority vote, bukan REPLUG.** REPLUG adalah algoritma branching kanonik di
FlashRAG, tapi ia membobot keluaran per-dokumen dengan probabilitas token — Ollama tidak
menjamin akses logprob. Majority vote adalah REPLUG tanpa pembobotan: pola fan-out dan
agregasinya identik, jadi menukar `MajorityVoteAggregator` dengan bobot logprob nanti
adalah perubahan satu file (ADR-010).

### 6.4.3 Algoritma lanjutan per arsitektur

Mengisi pola yang sudah ada, dikerjakan setelah keempat pola di atas jalan:

| Arsitektur | Lanjutan | Tambahan yang dibutuhkan | Status |
|---|---|---|---|
| loop | IRCoT | Kondisi berhenti dari isi output (deteksi `"answer is"`), retrieve per kalimat CoT | Direncanakan |
| loop | Self-Ask | Dekomposisi sub-pertanyaan + parsing | Direncanakan |
| loop | FLARE | Logprob token untuk mendeteksi kalimat ber-keyakinan rendah | 🔒 Butuh vLLM |
| loop | Self-RAG | Model ter-fine-tune dengan reflection token | 🔒 Di luar scope |
| branching | SuRe | Prompt-only: kandidat → validasi → ranking. Murni prompt berantai | Direncanakan |
| branching | REPLUG | Logprob token untuk bobot ensemble | 🔒 Butuh vLLM |
| conditional | SKR | Judger kNN atas koleksi pertanyaan yang diketahui | Direncanakan |
| conditional | Adaptive-RAG | Classifier kompleksitas query yang terlatih | 🔒 Butuh training |
| sequential | + Reranker / Refiner | Tahap antara retrieve dan prompt | Direncanakan |

## 6.5 Auditabilitas

"Bisa diaudit" dipakai di dokumen ini dengan arti sempit dan bisa diperiksa:
**seorang pembaca yang hanya memegang isi `results/<run_id>/` dapat menjawab enam pertanyaan
di bawah tanpa menjalankan ulang pipeline dan tanpa membaca kode.**

| # | Pertanyaan audit | Dijawab oleh |
|---|---|---|
| A1 | Apa yang terjadi pada satu query tertentu? | `QAResult.steps` — query, dokumen, prompt, output mentah tiap langkah |
| A2 | Run ini pakai setelan apa? | `config.yaml` tersalin + `provenance` (FR-8.7) |
| A3 | Berapa biayanya? | `num_llm_calls` dari boundary generator (FR-7.7) |
| A4 | Angka metriknya benar? | Hitung ulang dari `predictions.jsonl` (FR-8.8) |
| A5 | Orkestrasinya benar? | Uji reduksi (FR-7.8) |
| A6 | Dua run ini beda di mana? | `provenance` cukup diskriminatif (FR-8.7) |

### 6.5.1 Audit khusus per arsitektur

Tiap arsitektur punya satu **cara gagal yang khas**. Kolom terakhir adalah artefak yang
membuat kegagalan itu terlihat — tanpa itu, arsitektur tersebut tidak bisa diaudit.

| Arsitektur | Cara gagal yang khas | Artefak yang membuatnya terlihat |
|---|---|---|
| sequential | Dokumen relevan tidak terambil; prompt salah bentuk | `steps[0].retrieved` + `steps[0].prompt` |
| **loop** | Query iterasi tidak benar-benar berubah → loop hanya mengulang kerja yang sama dengan biaya `n` kali | `steps[i].query` per iterasi (FR-7b.5) + `num_llm_calls` |
| **branching** | Jawaban akhir hasil tie-break 1-1-1-1-1 tapi terbaca seolah konsensus | Tally agregasi (FR-7c.7) |
| **conditional** | Judger selalu menjawab "ya" → identik dengan sequential tapi lebih mahal | `JudgeDecision` tersimpan (FR-7d.5) + metrik routing (FR-8.6) + baseline `AlwaysRetrieveJudger` |

### 6.5.2 Uji reduksi sebagai alat audit utama

Tiap arsitektur punya konfigurasi yang membuatnya runtuh menjadi `SequentialPipeline`:

| Arsitektur | Konfigurasi reduksi | Harapan |
|---|---|---|
| loop | `n_iterations: 1` | `QAResult` identik dengan sequential |
| branching | `top_k: 1` | `QAResult` identik dengan sequential `top_k: 1` |
| conditional | `judger.type: always_retrieve` | `QAResult` identik dengan sequential |

Nilainya: ia memisahkan **bug orkestrasi** dari **kualitas algoritma**. Kalau uji reduksi
gagal, yang salah adalah kode perangkaiannya — tidak perlu mendebat apakah Iter-RetGen
memang membantu. Uji ini wajib dijalankan dengan generator deterministik (FR-7.9); dengan
LLM nyata, "identik" bukan properti yang bisa dijamin.

### 6.5.3 Batas yang diakui

Hal-hal yang **tidak** bisa diaudit dari artefak, dan diterima apa adanya:

1. **Counterfactual conditional.** "Apa jawabannya kalau judger memutuskan sebaliknya?" butuh run kedua. Mitigasinya bukan artefak, melainkan baseline `AlwaysRetrieveJudger`/`NeverRetrieveJudger` yang mengapit hasilnya.
2. **Determinisme LLM nyata.** Ollama dengan `temperature: 0` pun tidak dijamin bit-identik antar-versi. Karena itu kontrak determinisme (FR-7.9) hanya mengikat untuk generator deterministik.
3. **Apakah sebuah algoritma "benar" menurut papernya.** Yang diaudit repo ini adalah kebenaran implementasi terhadap PRD, bukan kesetiaan pada publikasi aslinya.

## 7. Kebutuhan fungsional

### FR-1 Konfigurasi
- FR-1.1 Config dibaca dari YAML, divalidasi model Pydantic; field tidak dikenal → error, bukan diabaikan diam-diam.
- FR-1.2 `configs/base.yaml` berisi default; YAML eksperimen hanya menuliskan selisihnya (merge dalam-dalam).
- FR-1.3 Secret (API key) dari env/`.env` sebagai `SecretStr`, tidak pernah dari YAML.
- FR-1.4 Pesan error validasi menyebut path field yang salah (`retriever.top_k`).
- FR-1.5 `RunConfig.seed: int` wajib ada dan punya default eksplisit. Semua sumber keacakan di repo ini (`RandomRetriever`, sampling dataset, tie-break yang tidak bisa dihindari) mengambil dari seed itu — tidak ada yang memanggil `random` global.
- FR-1.6 Config yang sudah di-resolve disalin ke `results/<run_id>/config.yaml`, dan hash-nya masuk ke `provenance` (FR-8.7). Sebuah run harus bisa diaudit tanpa menebak YAML mana yang dipakai.

### FR-2 Dataset
- FR-2.1 Baca JSONL berisi `id`, `question`, `golden_answers`.
- FR-2.2 Dukung `split` (train/dev/test) dan `limit` untuk smoke test cepat.
- FR-2.3 Baris cacat: default **gagal cepat** dengan nomor baris; mode `skip_malformed` opsional.

### FR-3 Corpus & index
- FR-3.1 Corpus JSONL berisi `id`, `contents` (opsional `title`).
- FR-3.2 Perintah `index` membangun index dense (FAISS) atau BM25 dan menyimpannya ke disk bersama metadata (nama model embedding, jumlah dokumen, dimensi).
- FR-3.3 Memuat index yang dibangun dengan model embedding berbeda dari config → error eksplisit, bukan skor sampah.
- FR-3.4 Pembangunan index menampilkan progres dan bisa diulang (idempoten atas folder output).

### FR-4 Retriever
- FR-4.1 ABC `BaseRetriever` dengan `search(query, top_k)` dan `batch_search(queries, top_k)`.
- FR-4.2 `DenseRetriever` (FAISS + embedding model) dan `BM25Retriever`.
- FR-4.3 Hasil terurut skor menurun, `rank` terisi mulai 0.
- FR-4.4 Dipilih dari config lewat registry, bukan `if/elif` di pipeline.
- FR-4.5 `RandomRetriever` **wajib ber-seed** dari `RunConfig.seed`. Ia dipakai di hampir semua test pipeline; kalau tidak deterministik, seluruh uji reduksi (FR-7.8) jadi flaky.

### FR-5 Prompt
- FR-5.1 `PromptTemplate` dengan placeholder `{question}` dan `{reference}`.
- FR-5.2 Punya system prompt terpisah dan format referensi yang bisa dikonfigurasi (`"Doc {rank} (Title: {title}) {text}"`).
- FR-5.3 Mendukung mode tanpa referensi (baseline no-retrieval) untuk ablasi.
- FR-5.4 Placeholder yang tidak terisi → error, bukan string literal `{question}` lolos ke LLM.
- FR-5.5 Template diberi nama per peran (`qa`, `judge`, `iter_followup`, `branch_single_doc`) dan dipilih berdasarkan kebutuhan pipeline. Pipeline merujuk template **dengan nama**, tidak menyusun string sendiri (FR-7.5).
- FR-5.6 Template tersimpan sebagai data (YAML/file teks), bukan literal panjang di dalam kode Python — supaya prompt bisa diubah tanpa menyentuh logika.

### FR-6 Generator
- FR-6.1 ABC `BaseGenerator` dengan `generate(prompts) -> list[str]`.
- FR-6.2 Backend default: **Ollama lewat API OpenAI-compatible** (`/v1/chat/completions`).
- FR-6.3 Parameter generasi (temperature, max_tokens, model) dari config.
- FR-6.4 Kegagalan jaringan → retry dengan backoff terbatas, lalu exception domain yang jelas.
- FR-6.5 Backend ditambahkan tanpa menyentuh pipeline (registry).

### FR-7 Pipeline — umum
- FR-7.1 ABC `BasePipeline` dengan `run(dataset) -> list[QAResult]`.
- FR-7.2 Pipeline dipilih dari config (`pipeline.type`) lewat registry. Mengganti arsitektur tidak boleh butuh perubahan kode (sejajar FR-4.4 dan FR-6.5).
- FR-7.3 Setiap langkah dicatat sebagai `GenerationStep` dengan `label` yang menerangkan perannya; durasi dan `num_llm_calls` terakumulasi di `QAResult`.
- FR-7.4 Satu item gagal tidak membatalkan seluruh run: `QAResult.error` terisi, run lanjut, dan ringkasan kegagalan dilaporkan di akhir.
- FR-7.5 Pipeline hanya **memanggil** komponen. Dilarang mengandung logika retrieval, pemanggilan HTTP, atau penyusunan string prompt sendiri — ketiganya milik `retriever/`, `generator/`, `prompt/`.
- FR-7.6 Keempat pipeline jalan di atas config dataset + retriever + generator yang sama, hanya beda blok `pipeline:` (G8).
- FR-7.7 **`num_llm_calls` dihitung di boundary generator, bukan dilaporkan sendiri oleh pipeline.** `CountingGenerator` membungkus generator mana pun dan menghitung prompt yang benar-benar lewat. Pipeline tidak punya akses untuk mengubah angkanya (ADR-014).
- FR-7.8 **Setiap pipeline wajib punya uji reduksi**: satu konfigurasi yang membuatnya runtuh menjadi `SequentialPipeline` dan menghasilkan `QAResult` identik — `n_iterations=1`, `top_k=1`, `AlwaysRetrieveJudger`. Dijalankan dengan generator deterministik (FR-7.9), bukan LLM nyata.
- FR-7.9 **Kontrak determinisme.** Uji reduksi dan uji reprodusibilitas dijalankan dengan `EchoGenerator` + `RandomRetriever` ber-seed. Dengan LLM nyata, "hasil identik" bukan properti yang bisa dijamin; perbandingan dengan LLM nyata memakai metrik agregat, bukan kesamaan per-item.

### FR-7a Sequential — Standard RAG
- FR-7a.1 Alur: retrieve → bangun prompt → generate → parse prediksi. Menghasilkan tepat 1 `GenerationStep`.
- FR-7a.2 Mode tanpa retrieval (`retrieval: false`) untuk baseline ablasi, memakai FR-5.3.

### FR-7b Loop — Iter-RetGen
- FR-7b.1 Jalankan `n_iterations` putaran retrieve→generate; jumlahnya dari config, **bukan** dari isi output.
- FR-7b.2 Query iterasi ke-`i+1` disusun dari query asli + jawaban iterasi ke-`i`. Strategi penyusunan ini eksplisit dan bisa diganti.
- FR-7b.3 Prediksi akhir = jawaban iterasi terakhir.
- FR-7b.4 `n_iterations` punya batas atas tervalidasi (`MAX_ITERATIONS`) supaya salah config tidak jadi loop mahal yang tak terdeteksi.
- FR-7b.5 Setiap iterasi tercatat sebagai satu `GenerationStep` berisi query yang **benar-benar dipakai** iterasi itu.

### FR-7c Branching — per-doc generate + majority vote
- FR-7c.1 Retrieve `top_k` dokumen, bangun **satu prompt per dokumen**, kirim semuanya lewat satu panggilan `generate(prompts)`.
- FR-7c.2 Agregator menggabungkan `top_k` kandidat jawaban jadi satu prediksi.
- FR-7c.3 `MajorityVoteAggregator`: voting atas jawaban yang sudah dinormalisasi (FR-8.2 dipakai ulang agar "Paris." dan "paris" tidak terhitung beda).
- FR-7c.4 Seri (tie) diputus deterministik: kandidat dengan skor retrieval tertinggi menang. Reprodusibilitas (NFR-5) tidak boleh bergantung pada urutan `dict`.
- FR-7c.5 Agregator adalah fungsi murni terdaftar di registry, bisa ditukar tanpa menyentuh pipeline.
- FR-7c.6 Setiap cabang tercatat sebagai `GenerationStep` beserta dokumen asalnya, supaya terlihat cabang mana yang benar.
- FR-7c.7 Langkah agregasi merekam **tally lengkap**: tiap jawaban kandidat, bentuk ternormalisasinya, jumlah vote, skor retrieval tertinggi pendukungnya, dan apakah hasilnya diputus mayoritas atau tie-break. Tanpa ini, mayoritas asli (4 dari 5) tidak bisa dibedakan dari tie-break (1-1-1-1-1) — padahal keduanya punya tingkat keyakinan yang jauh berbeda.
- FR-7c.8 Agregasi **bisa dijalankan ulang offline** dari `predictions.jsonl`: karena jawaban tiap cabang beserta skor dokumennya tersimpan, menukar aggregator tidak perlu memanggil LLM lagi. Ini sekaligus cara termurah mengevaluasi aggregator baru.

### FR-7d Conditional — LLM-prompted judger
- FR-7d.1 ABC `BaseJudger` dengan `judge(item) -> JudgeDecision`.
- FR-7d.2 `LLMJudger`: tanya ke LLM apakah pertanyaan butuh pengetahuan eksternal, parse jadi boolean.
- FR-7d.3 Output judger yang tak bisa di-parse → default **butuh retrieval**, dan dicatat di `JudgeDecision.reason`. Gagal ke arah yang aman, bukan ke arah yang menyembunyikan masalah.
- FR-7d.4 Routing: butuh retrieval → jalur sequential penuh; tidak butuh → generate langsung tanpa referensi.
- FR-7d.5 `JudgeDecision` tersimpan di `QAResult` termasuk output mentah judger, supaya kualitas routing-nya sendiri bisa dievaluasi.
- FR-7d.6 Judger dipilih dari config lewat registry (`judger.type`).

### FR-8 Evaluasi
- FR-8.1 Metrik: `exact_match`, `f1`, `precision`, `recall`, `retrieval_recall@k`.
- FR-8.2 Normalisasi jawaban (lowercase, buang artikel & tanda baca) mengikuti konvensi SQuAD.
- FR-8.3 Metrik adalah fungsi murni `(prediction, golden_answers) -> float`, bisa didaftarkan.
- FR-8.4 Output: `metrics.json` (agregat) + `predictions.jsonl` (per item, berisi `QAResult` penuh).
- FR-8.5 **Metrik biaya** dilaporkan bersama metrik kualitas: `avg_llm_calls` dan `avg_elapsed_s` per item. Tanpa ini, branching akan terlihat "lebih baik" dari sequential tanpa menunjukkan bahwa ia `top_k` kali lebih mahal (G8).
- FR-8.6 **Metrik routing** untuk pipeline conditional: proporsi item yang dirutekan ke retrieval, dan skor terpisah untuk masing-masing cabang. Judger yang selalu menjawab "ya" harus terlihat sebagai judger yang tidak berguna.
- FR-8.7 `metrics.json` mencatat `provenance` yang **cukup untuk membedakan dua run mana pun**: hash config yang sudah di-resolve, tipe + parameter pipeline (`n_iterations`, `top_k`, `aggregator.type`, `judger.type`), tipe retriever, nama + tag model generator, `temperature`, `seed`, nama + hash template prompt yang dipakai, dan jumlah item.
  *Uji kelayakannya:* kalau dua run bisa menghasilkan `provenance` identik tapi angka berbeda, `provenance`-nya belum lengkap. Tanpa ini `compare` (FR-9.5) menampilkan baris yang tidak bisa dibedakan, dan G8 gagal.
- FR-8.8 Seluruh metrik — termasuk metrik biaya dan routing — bisa **dihitung ulang dari `predictions.jsonl` saja**, tanpa index, tanpa model, tanpa jaringan.

### FR-9 CLI
- FR-9.1 `rag-finetune index <config>` — bangun index.
- FR-9.2 `rag-finetune run <config>` — jalankan pipeline + evaluasi.
- FR-9.3 `rag-finetune eval <predictions.jsonl>` — hitung ulang metrik tanpa inferensi lagi.
- FR-9.4 Flag `--limit N` untuk smoke test dan `--log-level`.
- FR-9.5 `rag-finetune compare <run_id> ...` — tabel perbandingan beberapa run berdampingan (skor + biaya), untuk menjawab G8 tanpa menyusun spreadsheet manual.
- FR-9.6 Flag `--pipeline <type>` menimpa `pipeline.type` di config, supaya keempat arsitektur bisa dijalankan dari satu YAML.

## 8. Kebutuhan non-fungsional

| ID | Kebutuhan |
|---|---|
| NFR-1 | Python >= 3.12, dependensi dikelola `uv` |
| NFR-2 | Jalan di CPU-only, RAM 16 GB, untuk corpus <= 1 juta dokumen |
| NFR-3 | Setiap file sumber < 300 baris; kalau lebih, pecah |
| NFR-4 | `ruff format` + `ruff check` bersih; type hint di semua signature |
| NFR-5 | Reprodusibel: run dengan config + seed sama menghasilkan skor sama |
| NFR-6 | Smoke test (50 item) selesai < 5 menit di laptop |
| NFR-7 | Dependensi berat (`faiss`, `torch`) adalah extras opsional, bukan dependensi inti |

## 9. Keputusan teknis

Rinciannya di [.agents/memory.md](.agents/memory.md). Ringkasan:

| Area | Keputusan | Alasan |
|---|---|---|
| Config | Pydantic + YAML | Type-safe, error jelas; perbaikan dari `dict` FlashRAG |
| Generator | Ollama (OpenAI-compatible) | Lokal, gratis, tanpa GPU; klien OpenAI-compatible juga langsung bisa dipakai untuk provider lain nanti |
| Retriever | FAISS (dense) + `bm25s` (sparse) | FAISS standar; `bm25s` murni Python, tanpa Java seperti Pyserini |
| Pipeline | 4 arsitektur, 1 algoritma termudah tiap arsitektur | Pelajaran utama RAG ada di pola orkestrasi, bukan di jumlah algoritma (§6.4) |
| Urutan pipeline | sequential → loop → branching → conditional | Diurutkan berdasarkan jumlah abstraksi baru, bukan kerumitan konsep (§6.4.1) |
| Algoritma logprob | Ditunda, tidak dibuang | REPLUG/FLARE perlu probabilitas token; polanya sudah disiapkan agar jadi perubahan 1 file saat vLLM ada (ADR-010) |
| Jejak eksekusi | `steps: list[GenerationStep]` | Loop & branching memecahkan asumsi 1-retrieve-1-generate (§6.3.1) |
| Scope | Inference saja | Fine-tuning butuh fondasi evaluasi yang bisa dipercaya dulu |

## 10. Acceptance criteria (MVP)

MVP dianggap selesai bila **semuanya** terpenuhi:

1. `uv sync` lalu `rag-finetune index configs/experiments/nq_sequential.yaml` membangun index dari corpus contoh.
2. `rag-finetune run configs/experiments/nq_sequential.yaml --limit 50` selesai tanpa intervensi manual dan menulis `metrics.json` + `predictions.jsonl`.
3. `metrics.json` memuat EM dan F1 yang masuk akal (EM > 0 pada dataset NQ contoh).
4. Mengganti `retriever.type` dari `dense` ke `bm25` berhasil tanpa perubahan kode.
5. Mengganti `generator.model` ke model Ollama lain berhasil tanpa perubahan kode.
6. `predictions.jsonl` berisi dokumen terambil dan prompt final untuk setiap item (G5).
7. **Keempat arsitektur jalan** dari config yang sama, hanya beda `--pipeline`:
   `sequential`, `loop`, `branching`, `conditional` — masing-masing menghasilkan `metrics.json` yang valid.
8. **`QAResult` sesuai arsitekturnya**: `loop` menghasilkan `n_iterations` langkah dengan query yang berubah tiap iterasi; `branching` menghasilkan satu langkah per dokumen; `conditional` mengisi `judge`.
9. **Trade-off terlihat**: `rag-finetune compare` menampilkan EM/F1 berdampingan dengan `avg_llm_calls`, dan `branching` memang tercatat lebih mahal daripada `sequential`.
10. Menukar `aggregator.type` dan `judger.type` berhasil tanpa perubahan kode.
11. **Keempat uji reduksi hijau** (FR-7.8): `n_iterations=1`, `top_k=1`, dan `always_retrieve` masing-masing menghasilkan `QAResult` identik dengan `SequentialPipeline`.
12. **Enam pertanyaan audit §6.5 terjawab** hanya dari isi `results/<run_id>/`, tanpa menjalankan ulang dan tanpa membaca kode. Diperiksa dengan benar-benar mencobanya, bukan dengan menyatakannya.
13. **`provenance` cukup diskriminatif**: dua run yang berbeda hanya di `n_iterations` menghasilkan `metrics.json` yang bisa dibedakan (ADR-016).
14. `pytest` hijau; coverage komponen inti >= 70%. Keempat pipeline punya test yang jalan **tanpa jaringan** (pakai `EchoGenerator` + `RandomRetriever` ber-seed).
15. Modul `retriever`, `generator`, `prompt`, `judger`, `aggregator`, `evaluation`, dan keempat pipeline masing-masing bisa dijalankan sendiri lewat `python -m`.
16. `docs/` menerangkan alur end-to-end dengan rujukan nama file sebenarnya, plus satu diagram per arsitektur.

## 11. Future scope (belum dikerjakan)

Dicatat agar arsitektur tidak menutup jalan ke sana, **bukan** untuk diimplementasikan sekarang:

1. **Provider LLM via API** (Anthropic/OpenAI) — sudah diantisipasi oleh ABC `BaseGenerator`.
2. **Backend vLLM** untuk inferensi batch throughput tinggi. Ini juga **pembuka kunci** untuk REPLUG dan FLARE, karena membawa akses logprob token (ADR-010).
3. **Algoritma lanjutan per arsitektur** — IRCoT, Self-Ask, SuRe, SKR. Masing-masing mengisi pola yang sudah ada; lihat tabel §6.4.3.
4. **Reranker** (cross-encoder) sebagai tahap antara retrieve dan prompt.
5. **Refiner** (kompresi konteks) — relevan khusus untuk branching, yang paling boros token.
6. **Algoritma yang butuh model khusus** — Self-RAG (reflection token), Adaptive-RAG (classifier kompleksitas query).
7. **Fine-tuning** — retriever embedding (hard negatives dari pipeline ini) dan generator (LoRA). Ini yang akhirnya membenarkan nama repo.

Catatan: keempat **arsitektur** sudah pindah dari daftar ini ke scope aktif (§6.4).
Yang masih di sini adalah algoritma-algoritma tambahan di dalam arsitektur tersebut.

## 12. Risiko

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Ollama lambat untuk evaluasi batch besar | Iterasi lambat | `--limit` untuk smoke test; vLLM di roadmap |
| Branching & loop mengalikan jumlah panggilan LLM (`top_k`× atau `n_iter`×) | Smoke test menembus anggaran NFR-6 | `--limit` lebih kecil untuk keduanya; `MAX_ITERATIONS` tervalidasi (FR-7b.4); `avg_llm_calls` selalu dilaporkan |
| Model Ollama kecil gagal mengikuti format judger | Routing conditional jadi acak | FR-7d.3 default ke "butuh retrieval"; metrik routing FR-8.6 membuat judger buruk kelihatan |
| Loop/branching menghasilkan `predictions.jsonl` berkali-kali lebih besar | Disk membengkak | Konsekuensi diterima (ADR-007); `--limit` untuk pengembangan |
| Build index FAISS boros memori | Tidak jalan di laptop | Embed per-batch streaming, tulis index bertahap |
| Zoo kualitas model Ollama bervariasi | Skor sulit dibandingkan | Catat nama + tag model di `metrics.json` |
| Scope creep ke 23 algoritma | MVP tidak pernah selesai | ROADMAP mengunci urutan; apa pun di luar fase saat ini masuk future scope |
