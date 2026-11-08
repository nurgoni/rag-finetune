# ROADMAP

Urutan pengerjaan `rag-finetune`. Dibaca bersama [PRD.md](../PRD.md).

**Aturan main:** fase dikerjakan berurutan. Sebuah fase baru boleh dimulai setelah
**exit criteria** fase sebelumnya terpenuhi. Kalau tergoda mengerjakan fase di depan,
catat idenya di [.agents/memory.md](../.agents/memory.md) dan lanjutkan fase sekarang.

Prinsip urutannya: **bangun dari bawah ke atas, tapi buktikan dari ujung ke ujung sedini mungkin.**
Fase 3 sengaja memakai retriever dan generator palsu supaya alur pipeline terbukti
sebelum ada model berat yang terlibat.

| Fase | Judul | Status |
|---|---|---|
| F0 | Fondasi repo | ✅ Selesai |
| F1 | Config & schema | ✅ Selesai |
| F2 | Data: dataset & corpus | ✅ Selesai |
| F3 | Pipeline walking skeleton (sequential) | ⬜ Belum |
| F4 | Generator Ollama | ⬜ Belum |
| F5 | Retriever nyata (BM25 → dense) | ⬜ Belum |
| F6 | Evaluasi & observabilitas | ⬜ Belum |
| **F7** | **Loop — Iter-RetGen** | ⬜ Belum |
| **F8** | **Branching — majority vote** | ⬜ Belum |
| **F9** | **Conditional — LLM judger** | ⬜ Belum |
| F10 | CLI, compare & dokumentasi | ⬜ Belum |
| F11+ | Future scope | 🔒 Terkunci |

**Mengapa F7–F9 setelah F6 (evaluasi), bukan sebelumnya.** Tiga arsitektur tambahan itu
masing-masing mengklaim "lebih baik dari Standard RAG". Klaim itu tidak bisa diperiksa
tanpa harness evaluasi yang sudah jalan. Mengerjakan loop sebelum ada EM/F1 berarti
menulis kode yang kebenarannya tidak bisa dibuktikan.

Urutan F7 → F8 → F9 mengikuti kriteria kesulitan di [PRD §6.4.1](../PRD.md), yaitu
**jumlah abstraksi baru** — bukan kerumitan konsep. Conditional idenya paling sepele
tapi implementasinya paling mahal karena menambah tipe komponen baru.

---

## F0 — Fondasi repo

**Tujuan:** repo bisa di-install, di-lint, dan di-test walau belum ada fitur.

Deliverable:
- `pyproject.toml` (`uv`, Python >= 3.12, ruff, pytest) dengan extras `[dense]`, `[sparse]`, `[dev]`
- Layout paket `src/rag_finetune/` sesuai [AGENTS.md §4](../AGENTS.md)
- `rag_finetune/exceptions.py` — hierarki `RagFinetuneError`
- `rag_finetune/utils/logging_config.py` — `setup_logging(level)` satu tempat. **Bukan** `logging.py`: nama itu menyadur modul stdlib dan merusak modul saat dimuat berdiri sendiri
- `.gitignore`, `.env.example`, `README.md` kerangka
- Satu test sanity (`test_import.py`)

Exit criteria:
- `uv sync --extra dev` sukses
- `ruff check .` dan `pytest` hijau
- `python -c "import rag_finetune"` tidak melakukan I/O apa pun

---

## F1 — Config & schema

**Tujuan:** kontrak data dan konfigurasi final sebelum ada logika yang bergantung padanya.

Deliverable:
- `schema/` — `Document`, `RetrievedDocument`, `QAItem`, `QAResult`, `MetricReport` (Pydantic, `frozen=True`)
- `config/schema.py` — `Config` dengan sub-model `DatasetConfig`, `CorpusConfig`, `IndexConfig`, `RetrieverConfig`, `GeneratorConfig`, `PromptConfig`, `EvaluationConfig`, `RunConfig`
- `config/loader.py` — `ConfigLoader` yang merge `base.yaml` + YAML eksperimen (deep merge) lalu validasi
- `configs/base.yaml` dan `configs/experiments/nq_sequential.yaml`
- Secret lewat `SecretStr` dari `.env`
- `RunConfig.seed` dengan default eksplisit (FR-1.5) — prasyarat seluruh uji reduksi di F7–F9
- `config_hash()` atas config yang sudah di-resolve (FR-1.6) — prasyarat `provenance` di F6

Exit criteria:
- Field tak dikenal di YAML → `ValidationError` yang menyebut path field (FR-1.1, FR-1.4)
- Dua config yang berbeda **di field apa pun** menghasilkan `config_hash` berbeda; dua config identik menghasilkan hash sama (ADR-016)
- `python -m rag_finetune.config.loader configs/experiments/nq_sequential.yaml` mencetak config tervalidasi
- Test: valid, field asing, tipe salah, deep merge

**Catatan desain:** di sinilah repo ini menyimpang dari FlashRAG. FlashRAG melewatkan `dict`
ke mana-mana; kita melewatkan sub-model yang sudah tervalidasi. Komponen hanya menerima
bagian config yang ia butuhkan, bukan seluruh config.

---

## F2 — Data: dataset & corpus

**Tujuan:** data masuk ke sistem dengan tipe yang benar dan error yang jelas.

Deliverable:
- `dataset/loader.py` — `QADatasetLoader`: JSONL → `list[QAItem]`, dukung `split` + `limit`
- `corpus/loader.py` — `CorpusLoader`: JSONL → `Iterator[Document]` (streaming, bukan list)
- `utils/jsonl.py` — `read_jsonl`, `write_jsonl` dengan nomor baris pada error. **Bukan** `io.py`: alasan yang sama dengan `logging_config.py` di F0 (pelajaran L2)
- Data contoh kecil di `tests/fixtures/` (± 20 dokumen, 10 pertanyaan) agar test tidak butuh download

Exit criteria:
- Baris cacat → error menyebut nomor baris (FR-2.3)
- Corpus 100 ribu baris bisa diiterasi tanpa memuat semuanya ke memori
- Kedua loader punya blok `__main__` yang mencetak ringkasan isi

---

## F3 — Pipeline walking skeleton

**Tujuan:** **alur end-to-end terbukti jalan sebelum ada model berat.**
Ini fase terpenting untuk tujuan "paham end-to-end".

Deliverable:
- `prompt/template.py` — `PromptTemplate` (FR-5)
- `generator/base.py` — ABC `BaseGenerator`
- `retriever/base.py` — ABC `BaseRetriever`
- `generator/echo.py` — `EchoGenerator`: mengembalikan potongan dokumen pertama. Bukan sekadar stub test, tapi baseline yang berguna
- `retriever/random_retriever.py` — `RandomRetriever`: ambil dokumen acak dari corpus. Berfungsi sebagai **lower bound** untuk membandingkan retriever nyata nanti
- `pipeline/base.py` — ABC `BasePipeline`
- `pipeline/sequential.py` — `SequentialPipeline`, Standard RAG (FR-7a)
- `registry.py` — registry generik berbasis decorator untuk retriever/generator/pipeline/judger/aggregator

Exit criteria:
- Satu script menjalankan: dataset → random retrieve → prompt → echo generate → `list[QAResult]`
- `QAResult` terisi penuh: satu `GenerationStep` berisi dokumen, prompt, output mentah; plus prediksi
- Satu item gagal tidak menghentikan run (FR-7.4), dan dilaporkan di ringkasan
- Test pipeline jalan **tanpa jaringan dan tanpa model**

**Penting untuk fase F7–F9:** `QAResult` sudah memakai `steps: list[GenerationStep]` sejak
fase ini, walau sequential hanya pernah mengisi satu langkah. Ini **bukan** antisipasi
berlebihan yang dilarang [AGENTS.md §9](../AGENTS.md) — ia kebutuhan tertulis di
[PRD §6.3.1](../PRD.md) untuk arsitektur yang sudah disepakati masuk scope. Alternatifnya
adalah memigrasikan format `predictions.jsonl` tiga kali di F7, F8, dan F9.

---

## F4 — Generator Ollama

**Tujuan:** ganti generator palsu dengan LLM nyata. Alurnya sudah tidak berubah lagi.

Deliverable:
- `generator/openai_compatible.py` — `OpenAICompatibleGenerator`: base URL + model + params. Ollama adalah satu instansiasi darinya (`http://localhost:11434/v1`)
- Retry dengan exponential backoff, cap percobaan (FR-6.4)
- `GeneratorError` beserta konteks (status code, potongan body)
- `generator/counting.py` — `CountingGenerator`, pembungkus yang menghitung prompt yang benar-benar lewat (FR-7.7, ADR-014)
- Mark `@pytest.mark.integration` untuk test yang benar-benar memanggil Ollama

Exit criteria:
- Pipeline F3 jalan utuh dengan Ollama lokal, `RandomRetriever` masih dipakai
- Ollama mati → pesan error yang bisa dipahami manusia, bukan traceback `httpx` mentah
- Unit test tidak menyentuh jaringan (HTTP client di-fake)
- `CountingGenerator` menghitung `n` untuk satu `generate()` berisi `n` prompt, dan tetap benar saat ada retry — retry **bukan** panggilan baru, yang diukur kerja model (ADR-013)

**Catatan desain:** class-nya dinamai `OpenAICompatibleGenerator`, bukan `OllamaGenerator`,
karena permukaan API yang sama nanti melayani provider lain. Ini satu-satunya antisipasi
masa depan yang diizinkan di fase ini, dan biayanya nol.

---

## F5 — Retriever nyata

**Tujuan:** retrieval berkualitas. BM25 dulu karena tanpa GPU, tanpa embedding, tanpa build lama.

### F5a — BM25
- `index/bm25_builder.py` — bangun & simpan index `bm25s`
- `retriever/bm25.py` — `BM25Retriever`
- `IndexMetadata` disimpan bersama artefak (jumlah dokumen, tanggal, versi tokenizer)

### F5b — Dense
- `index/embedder.py` — `TextEmbedder` (sentence-transformers, batching, normalisasi)
- `index/faiss_builder.py` — bangun index FAISS secara streaming per batch
- `retriever/dense.py` — `DenseRetriever`
- Validasi: nama model embedding di metadata index harus sama dengan di config (FR-3.3)

Exit criteria:
- `rag-finetune index` menghasilkan index yang bisa dimuat ulang
- Mengganti `retriever.type` antara `bm25`/`dense`/`random` hanya lewat YAML (AC-4)
- Keduanya mengalahkan `RandomRetriever` pada `retrieval_recall@5` — ini validasi kebenaran, bukan sekadar benchmark
- Build index 100 ribu dokumen tidak melebihi anggaran memori NFR-2

---

## F6 — Evaluasi & observabilitas

**Tujuan:** angka yang bisa dipercaya dan jejak yang bisa di-debug.

Deliverable:
- `evaluation/metrics.py` — fungsi murni: `exact_match`, `f1`, `precision`, `recall`, `retrieval_recall_at_k`, plus `normalize_answer` (konvensi SQuAD)
- `evaluation/evaluator.py` — `Evaluator`: metrik dipilih dari config, agregasi, bangun `MetricReport`
- `evaluation/registry.py` — metrik baru didaftarkan lewat decorator (FR-8.3)
- Penulis hasil: `results/<run_id>/{config.yaml, metrics.json, predictions.jsonl}`
- `run_id` deterministik dari hash config + timestamp

Exit criteria:
- Metrik diuji terhadap kasus yang dihitung tangan (termasuk jawaban berganda, tanda baca, artikel)
- `metrics.json` mencatat `provenance` yang **cukup diskriminatif**: `config_hash`, parameter pipeline, tipe retriever, nama + tag model, `temperature`, `seed`, nama + hash template (FR-8.7, ADR-016)
- Uji `provenance`: dua run yang berbeda hanya di `n_iterations` harus menghasilkan `provenance` berbeda. Ini test, bukan konvensi
- Semua metrik bisa dihitung ulang dari `predictions.jsonl` saja — tanpa index, tanpa model, tanpa jaringan (FR-8.8)
- **Metrik biaya** `avg_llm_calls` dan `avg_elapsed_s` sudah dilaporkan (FR-8.5), walau sequential selalu bernilai 1.0. Ini prasyarat F7–F9: tanpa angka biaya, "loop lebih akurat" adalah klaim setengah
- `rag-finetune eval` menghitung ulang skor dari `predictions.jsonl` tanpa inferensi (FR-9.3)
- `normalize_answer` diekspor sebagai fungsi publik — F8 memakainya ulang untuk voting (FR-7c.3)

---

## F7 — Loop: Iter-RetGen

**Tujuan:** arsitektur kedua, dengan **nol abstraksi baru**. Murni orkestrasi.

Ide algoritmanya: jawaban sementara adalah query yang lebih baik. Iterasi ke-2 mencari
dengan "pertanyaan asli + jawaban iterasi ke-1", sehingga dokumen yang ditemukan bergeser
ke arah yang relevan. Jumlah iterasinya **tetap**, dari config — tidak ada kondisi berhenti
yang perlu ditebak.

Deliverable:
- `pipeline/loop.py` — `IterRetGenPipeline` (FR-7b)
- `LoopConfig`: `n_iterations` dengan validasi batas atas `MAX_ITERATIONS` (FR-7b.4)
- Strategi penyusunan query lanjutan sebagai unit yang bisa diganti — template `iter_followup` (FR-5.5), bukan f-string di tengah pipeline
- Template `iter_followup` di `prompt/templates/`

Exit criteria:
- `QAResult.steps` berisi tepat `n_iterations` langkah, berlabel `iter-0`…`iter-N`
- Setiap langkah mencatat **query yang benar-benar dipakai** iterasi itu — query iterasi ke-2 harus terlihat berbeda dari iterasi ke-1 (FR-7b.5)
- **Uji reduksi:** `n_iterations: 1` menghasilkan `QAResult` identik dengan `SequentialPipeline`. Dijalankan dengan `EchoGenerator` + `RandomRetriever` ber-seed (FR-7.9, ADR-015) — bukan dengan Ollama, karena "identik" tidak bisa dijamin LLM mana pun
- `n_iterations` di luar batas → `ValidationError` saat start, bukan tagihan token
- `avg_llm_calls` == `n_iterations` tepat, diukur `CountingGenerator`, bukan dilaporkan pipeline
- **Uji anti-loop-mandek:** `steps[1].query != steps[0].query`. Loop yang mengulang query yang sama persis membayar `n` kali untuk kerja yang sama — kegagalan khas arsitektur ini (PRD §6.5.1) dan tidak terlihat dari skor
- Test jalan tanpa jaringan

---

## F8 — Branching: per-doc generate + majority vote

**Tujuan:** arsitektur ketiga. Satu abstraksi baru, dan itu fungsi murni.

Ide algoritmanya: daripada menjejalkan `top_k` dokumen ke satu prompt, bangun `top_k`
prompt masing-masing dengan satu dokumen, lalu vote jawabannya. Keuntungannya: satu
dokumen yang menyesatkan tidak mencemari seluruh konteks.

Deliverable:
- `pipeline/branching.py` — `BranchingPipeline` (FR-7c)
- `aggregator/base.py` + registry; `aggregator/majority_vote.py` — fungsi murni (FR-7c.5)
- Template `branch_single_doc`
- Pemetaan balik fan-out: `top_k` prompt per item dalam satu panggilan `generate()`, hasil dipetakan kembali ke item asalnya
- `Aggregation` + `VoteTally` di `schema/` — tally lengkap beserta `resolved_by` (FR-7c.7)

Exit criteria:
- `QAResult.steps` berisi satu langkah per dokumen (`doc-0`…`doc-k`) plus satu langkah agregasi
- Voting memakai `normalize_answer` dari F6 — "Paris." dan "paris" terhitung satu (FR-7c.3)
- **Tie diputus deterministik** oleh skor retrieval tertinggi; dua run berurutan menghasilkan prediksi identik (FR-7c.4, NFR-5)
- **Uji reduksi:** `top_k: 1` menghasilkan `QAResult` identik dengan sequential `top_k: 1`, dengan `EchoGenerator` (ADR-015)
- **Tally terekam dan `resolved_by` benar:** kasus 4-dari-5 tercatat `majority`, kasus 1-1-1-1-1 tercatat `tie_break`. Tanpa ini, tie-break menyamar jadi konsensus — kegagalan khas arsitektur ini (PRD §6.5.1)
- **Agregasi bisa dihitung ulang offline** dari `predictions.jsonl`: menukar aggregator tidak memanggil LLM (FR-7c.8). Diuji dengan menjalankan ulang aggregator atas file hasil dan membandingkan prediksinya
- `avg_llm_calls` == `top_k` tepat, dan `compare` memperlihatkannya sebagai biaya nyata
- Satu cabang gagal tidak membatalkan item; voting jalan atas cabang yang berhasil, dan tally mencatat jumlah cabang yang gagal

**Catatan desain:** `MajorityVoteAggregator` adalah REPLUG tanpa pembobotan logprob.
Pola fan-out + agregasinya identik, jadi REPLUG nanti = satu aggregator baru, bukan
pipeline baru (ADR-010).

---

## F9 — Conditional: LLM judger

**Tujuan:** arsitektur keempat. Satu **tipe komponen baru** yang bisa ditukar.

Ide algoritmanya: tidak semua pertanyaan butuh retrieval. "Siapa penulis Hamlet?" sudah ada
di parameter model; retrieval hanya menambah bising dan biaya. Judger memutuskan rutenya.

Deliverable:
- `judger/base.py` — ABC `BaseJudger` + registry (FR-7d.1, FR-7d.6)
- `judger/llm_judger.py` — `LLMJudger` (FR-7d.2)
- `judger/always.py` — `AlwaysRetrieveJudger` / `NeverRetrieveJudger`: baseline sepele yang membuat terlihat apakah judger nyata benar-benar menambah nilai
- `pipeline/conditional.py` — `ConditionalPipeline` (FR-7d.4)
- Template `judge`
- Metrik routing (FR-8.6): proporsi ke retrieval + skor terpisah per cabang

Exit criteria:
- Output judger yang tak terparse → default butuh retrieval, alasannya tercatat di `JudgeDecision.reason` (FR-7d.3)
- `JudgeDecision` beserta output mentah judger tersimpan di `QAResult` (FR-7d.5)
- **Uji reduksi:** `judger.type: always_retrieve` menghasilkan `QAResult` identik dengan `SequentialPipeline`, dengan `EchoGenerator` (ADR-015)
- Metrik routing membuat judger yang selalu menjawab "ya" terlihat tidak berguna — skor `LLMJudger` dibandingkan dengan `AlwaysRetrieveJudger` dan `NeverRetrieveJudger` yang mengapitnya (PRD §6.5.3)
- `num_llm_calls` **termasuk panggilan judger** — conditional tidak boleh terlihat gratis. Terjamin karena penghitungan ada di boundary generator (ADR-014)
- Cabang no-retrieval memakai mode tanpa referensi dari FR-5.3, bukan prompt duplikat
- `steps[0].retrieved` kosong pada cabang no-retrieval, dan itu bisa dibedakan dari "retrieval gagal menemukan apa pun" lewat `JudgeDecision`

---

## F10 — CLI, compare & dokumentasi

**Tujuan:** orang lain (dan Anda tiga bulan dari sekarang) bisa memakainya.

Deliverable:
- `cli.py` — subcommand `index`, `run`, `eval`, `compare` (`typer`), flag `--limit`, `--log-level`, `--pipeline`
- `--pipeline` menimpa `pipeline.type` supaya keempat arsitektur jalan dari satu YAML (FR-9.6)
- `compare` — tabel beberapa run berdampingan: skor + `avg_llm_calls` (FR-9.5)
- Broad exception catch **hanya di sini**, dengan exit code dan pesan rapi
- `docs/ARCHITECTURE.md` — penjelusuran satu query melewati seluruh sistem, merujuk file:baris nyata
- `docs/PIPELINES.md` — satu diagram + satu contoh `steps` nyata per arsitektur, plus tabel kapan memakai yang mana
- `docs/QUICKSTART.md` — dari nol sampai skor pertama
- `docs/EXTENDING.md` — cara menambah retriever, generator, judger, aggregator, metrik, pipeline
- `README.md` final

Exit criteria:
- Seluruh **acceptance criteria MVP** di [PRD.md §10](../PRD.md) terpenuhi, termasuk AC-7 sampai AC-16 (empat arsitektur, uji reduksi, dan enam pertanyaan audit §6.5)
- Satu perintah menjalankan keempat arsitektur berurutan dan menghasilkan satu tabel perbandingan
- Orang yang belum pernah melihat repo ini bisa mendapat skor pertama hanya dari QUICKSTART

---

## F11+ — Future scope (terkunci)

Jangan dikerjakan sebelum F10 tuntas. Keempat **arsitektur** sudah masuk scope aktif
(F3, F7, F8, F9); yang tersisa di sini adalah algoritma tambahan yang mengisi pola itu,
plus hal-hal di luar orkestrasi.

| Prioritas | Item | Arsitektur | Prasyarat |
|---|---|---|---|
| 1 | IRCoT — berhenti berdasarkan isi output, retrieve per kalimat CoT | loop | F7 |
| 2 | SuRe — kandidat → validasi → ranking, murni prompt | branching | F8 |
| 3 | SKR — judger kNN atas pertanyaan yang diketahui | conditional | F9 |
| 4 | Reranker cross-encoder | sequential | F5b |
| 5 | Self-Ask — dekomposisi sub-pertanyaan | loop | F7 |
| 6 | Provider LLM via API (Anthropic/OpenAI) | — | F4, tinggal base URL + auth |
| 7 | Backend vLLM | — | F4 + GPU |
| 8 | **REPLUG** — bobot ensemble dari logprob | branching | item 7 |
| 9 | **FLARE** — deteksi keyakinan rendah dari logprob | loop | item 7 |
| 10 | Refiner (kompresi konteks) | semua, terutama branching | F10 |
| 11 | Adaptive-RAG — classifier kompleksitas query | conditional | butuh training |
| 12 | Self-RAG — model dengan reflection token | loop | butuh training |
| 13 | **Fine-tuning retriever** (hard negatives dari pipeline ini) | — | F6, butuh evaluasi tepercaya |
| 14 | **Fine-tuning generator** (LoRA) | — | item 13 |

Item 8–9 terkunci di balik vLLM, bukan di balik kesulitan: keduanya butuh akses logprob
token yang tidak dijamin Ollama (ADR-010). Begitu backend-nya ada, keduanya adalah
pengisian pola yang sudah berdiri — REPLUG satu aggregator baru, FLARE satu kondisi
berhenti baru.

Item 13–14 adalah yang akhirnya membenarkan nama repo. Keduanya sengaja ditaruh paling akhir:
fine-tuning tanpa harness evaluasi yang tepercaya hanya menghasilkan angka yang tidak bisa dipercaya.
