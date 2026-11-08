# AGENTS.md

Panduan operasional untuk agent (dan manusia) yang menulis kode di repo ini.
Baca file ini **sebelum** menulis baris kode pertama.

Proyek ini adalah **rewrite terdidik** dari [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG).
Tujuannya bukan parity fitur, tapi **kejelasan alur end-to-end**: setiap layer harus bisa
dibaca, dijalankan, dan dipahami sendiri-sendiri.

Dokumen pendamping:

| File | Isi |
|---|---|
| [PRD.md](PRD.md) | Apa yang dibangun dan mengapa, acceptance criteria |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Urutan milestone dan exit criteria |
| [.agents/memory.md](.agents/memory.md) | Keputusan arsitektur yang sudah final (ADR) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, tooling, workflow PR |

---

## 1. Prinsip utama

> *Simple is better than complex. Readability counts. Explicit is better than implicit.*

Repo ini dioptimalkan untuk **dibaca sekali dan dimengerti**, bukan untuk benchmark tercepat.
Kalau harus memilih antara 10 baris yang jelas dan 3 baris yang pintar, pilih yang 10 baris.

Tiga aturan yang tidak bisa dinegosiasikan:

1. **Satu modul, satu tanggung jawab.** `retriever` tidak tahu soal prompt. `generator` tidak tahu soal FAISS.
2. **Boundary lewat abstraksi eksplisit.** Komponen saling bicara lewat ABC/Protocol dan model Pydantic, bukan lewat `dict` bebas.
3. **Tidak ada side effect saat import.** Tidak ada model yang di-load, tidak ada file yang dibuka, tidak ada koneksi HTTP di level modul.

---

## 2. Alur kerja wajib (Zen Workflow)

Setiap kali menambah fungsi/class baru, ikuti **5 langkah ini secara berurutan**.
Jangan lompat ke langkah 3.

### Langkah 1 — Klarifikasi Input/Output

Tentukan tipe data masuk dan keluar sebelum apa pun.
Kalau ambigu (format dataset, bentuk skor, satuan, apa yang terjadi saat kosong) → **tanya pengguna**, jangan menebak.

Checklist:
- Tipe konkret masuk/keluar? (`list[Document]`, bukan "list of docs")
- Apa yang terjadi kalau input kosong / `None`?
- Exception apa yang mungkin dilempar?

### Langkah 2 — Tulis Signature + Docstring dulu

Definisikan permukaan API sebelum isinya. Type hint lengkap, docstring Google-style ringkas.

```python
class DenseRetriever(BaseRetriever):
    """Retriever dense berbasis FAISS atas corpus yang sudah di-index."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Ambil `top_k` dokumen paling mirip dengan `query`.

        Args:
            query: Pertanyaan pengguna dalam natural language.
            top_k: Jumlah dokumen yang dikembalikan. Harus >= 1.

        Returns:
            Dokumen terurut skor menurun. List kosong kalau index kosong.

        Raises:
            ValueError: Kalau `top_k` < 1.
            IndexNotBuiltError: Kalau index belum pernah dibangun.
        """
```

Docstring minimal berisi: satu baris ringkasan, `Args`, `Returns`, `Raises` (kalau ada).

### Langkah 3 — Implementasi paling sederhana

Tulis versi yang paling mudah dibaca, **walau sedikit lebih panjang**.
Jangan optimasi, jangan caching, jangan batching, jangan `async` — kecuali PRD memintanya secara eksplisit.
Optimasi datang setelah ada benchmark yang membuktikan perlunya.

### Langkah 4 — Review mental Zen

Baca ulang kode sendiri dan periksa empat hal ini:

| Gejala | Tindakan |
|---|---|
| Nama membingungkan (`data`, `tmp`, `res`, `proc`) | Ganti jadi nama yang menyebut isinya (`retrieved_docs`, `score_matrix`) |
| Nama modul menyadur stdlib (`logging.py`, `json.py`, `types.py`, `io.py`) | Ganti (`logging_config.py`). Di dalam paket absolute import tetap benar, jadi kegagalannya baru muncul saat modul dimuat berdiri sendiri — lihat pelajaran L2 di [.agents/memory.md](.agents/memory.md) |
| Nesting > 2 level | Refactor: early return, guard clause, atau extract method |
| `except Exception` / `except:` | Persempit ke exception konkret. Broad catch **hanya** di boundary sistem (CLI entrypoint, HTTP handler) dan wajib di-log |
| Magic number (`0.75`, `512`, `100`) | Jadikan konstanta bernama `UPPER_CASE` di level modul, atau field config |

### Langkah 5 — Sertakan contoh penggunaan

Setiap modul yang bisa dijalankan sendiri **wajib** punya demo:

```python
if __name__ == "__main__":
    retriever = DenseRetriever.from_config(load_config("configs/base.yaml"))
    for doc in retriever.search("Siapa penemu bola lampu?", top_k=3):
        print(f"{doc.score:.3f}  {doc.text[:80]}")
```

Untuk fungsi murni kecil, doctest lebih baik daripada blok `__main__`.

---

## 3. Class vs Function

**Default: class.** Pakai class kalau komponen punya salah satu dari ini:

- **State** — index FAISS yang sudah dimuat, tokenizer, cache
- **Dependency** — butuh config, butuh retriever lain, butuh HTTP client
- **Lifecycle** — perlu setup/teardown, `build()` lalu `search()`

Contoh di repo ini: `Config`, `DenseRetriever`, `OpenAICompatibleGenerator`, `SequentialPipeline`,
`Evaluator`, `LLMJudger` (punya dependency ke generator).

**Pakai function kalau:**

- Transformasi stateless murni → `normalize_answer(text: str) -> str`
- Metrik murni → `exact_match(pred: str, golds: list[str]) -> float`
- Agregator → `majority_vote(candidates: list[Candidate]) -> str`
- Entrypoint CLI dan factory tingkat atas → `def main() -> None`
- pytest fixture (framework mengharapkan function)

**Asimetri yang disengaja:** `judger/` berisi class, `aggregator/` berisi function, walau
keduanya titik ekstensi yang sejajar dan sama-sama memakai registry. Alasannya: judger punya
dependency (generator, atau index kNN untuk SKR) dan state; aggregator tidak punya keduanya —
ia hanya memetakan daftar kandidat ke satu jawaban. Jangan "diseragamkan" (ADR-012).

**Aturan pemutus:** kalau sebuah class hanya berisi `__init__` + satu method publik dan tidak pernah
mengubah state-nya sendiri, **itu function yang dipakaikan kostum class**. Ubah jadi function.

**Aturan sebaliknya:** kalau sebuah function punya >3 parameter yang selalu datang bersama-sama
(`model`, `tokenizer`, `device`, `max_len`), itu class yang belum lahir. Jadikan class.

---

## 4. Struktur proyek & batas dependensi

```
rag-finetune/
├── configs/                      # YAML eksperimen (data, bukan kode)
│   ├── base.yaml
│   └── experiments/
├── src/rag_finetune/
│   ├── config/                   # Pydantic settings + loader YAML
│   ├── schema/                   # Model data lintas-layer (Document, QAItem, ...)
│   ├── dataset/                  # Baca dataset QA (JSONL) -> QAItem
│   ├── corpus/                   # Baca corpus dokumen -> Document
│   ├── index/                    # Bangun & simpan index (FAISS, BM25)
│   ├── retriever/                # BaseRetriever + dense/bm25
│   ├── prompt/                   # PromptTemplate
│   │   └── templates/            # prompt sebagai data: qa, judge, iter_followup, ...
│   ├── generator/                # BaseGenerator + backend Ollama
│   ├── judger/                   # BaseJudger (class) — untuk pipeline conditional
│   ├── aggregator/               # penggabung kandidat (fungsi murni) — untuk branching
│   ├── pipeline/                 # BasePipeline + 4 arsitektur
│   │   ├── base.py
│   │   ├── sequential.py         # Standard RAG
│   │   ├── loop.py               # Iter-RetGen
│   │   ├── branching.py          # per-doc generate + vote
│   │   └── conditional.py        # judger routing
│   ├── evaluation/               # metrik + Evaluator
│   ├── utils/                    # logging_config, jsonl, timer
│   └── cli.py                    # entrypoint
├── scripts/                      # skrip sekali-pakai (download, preprocess)
├── tests/
└── docs/
```

Arah dependensi **satu arah saja**:

```
cli → pipeline → {retriever, generator, prompt, judger, aggregator, evaluation}
                       ↓
              {index, dataset, corpus}
                       ↓
              {config, schema} → utils
```

Aturan keras:

- **`schema/` hanya boleh import Pydantic + stdlib.** Tidak ada `torch`, `faiss`, `transformers`, `httpx`.
- **`utils/` tidak boleh import modul lain dari repo ini.**
- `retriever/` tidak boleh import `generator/`, dan sebaliknya. Yang menyatukan mereka adalah `pipeline/`.
- `judger/` dan `aggregator/` **tidak boleh** import `pipeline/`. Arahnya satu arah: pipeline memakai mereka, bukan sebaliknya.
- Import lintas-layer ke atas (misal `config` import `pipeline`) = circular dependency. Tolak.

### Aturan khusus `pipeline/`

Pipeline adalah lapisan **orkestrasi**. Ia memanggil komponen; ia tidak menjadi komponen.

Yang **dilarang** hidup di dalam `pipeline/`:

- Pemanggilan HTTP atau SDK model apa pun → milik `generator/`
- Logika skoring atau akses index → milik `retriever/`
- Penyusunan string prompt (f-string dengan instruksi di dalamnya) → milik `prompt/`
- Normalisasi atau perhitungan metrik → milik `evaluation/`

Uji cepatnya: kalau sebuah pipeline tidak bisa dites dengan `EchoGenerator` +
`RandomRetriever` tanpa jaringan, berarti ia menyimpan logika yang bukan miliknya.

### Empat arsitektur pipeline

Keempatnya memakai retriever, prompt, dan generator yang **sama persis**. Yang berbeda hanya
berapa kali dan dalam urutan apa ketiganya dipanggil.

| Tipe | Algoritma pertama | Bentuk `QAResult.steps` |
|---|---|---|
| `sequential` | Standard RAG | 1 langkah |
| `loop` | Iter-RetGen | 1 per iterasi, query berubah tiap langkah |
| `branching` | per-doc generate + majority vote | 1 per dokumen + 1 agregasi |
| `conditional` | LLM judger | 1 langkah, `judge` terisi |

Tiga aturan yang berlaku untuk setiap pipeline baru:

1. **Wajib ada uji reduksi.** Harus ada konfigurasi yang membuatnya runtuh jadi sequential
   dan menghasilkan hasil **identik** (`n_iterations=1`, `top_k=1`, `AlwaysRetrieveJudger`).
   Ini pertahanan termurah terhadap bug orkestrasi: ia memisahkan bug perangkaian dari
   kualitas algoritma. **Dijalankan dengan `EchoGenerator` + `RandomRetriever` ber-seed**,
   bukan dengan Ollama — "identik" bukan properti yang bisa dijamin LLM mana pun (ADR-015).
2. **Jangan menghitung `num_llm_calls` sendiri.** Penghitungan terjadi di `CountingGenerator`
   yang membungkus generator. Kode pipeline yang mengandung `num_llm_calls +=` akan ditolak
   review (ADR-014).
3. **Wajib menyimpan jejak audit khas arsitekturnya.** Tiap arsitektur punya satu cara gagal
   yang tidak terlihat dari skor, dan satu artefak yang membuatnya terlihat:

   | Arsitektur | Cara gagal yang tak terlihat dari skor | Artefak wajib |
   |---|---|---|
   | loop | Query tiap iterasi tidak berubah → bayar `n` kali untuk kerja yang sama | `steps[i].query` |
   | branching | Tie-break 1-1-1-1-1 terbaca seolah konsensus | `Aggregation.tally` + `resolved_by` |
   | conditional | Judger selalu bilang "ya" → sama dengan sequential tapi lebih mahal | `JudgeDecision` + metrik routing |

Definisi "bisa diaudit" yang dipakai repo ini — enam pertanyaan yang harus bisa dijawab hanya
dari isi `results/<run_id>/` — ada di [PRD §6.5](PRD.md). Kalau sebuah properti kualitas tidak
bisa dipetakan ke artefak yang menjawabnya, ia aspirasi dan bukan kebutuhan (pelajaran L1 di
[.agents/memory.md §3](.agents/memory.md)).

Rinciannya di [PRD §6.4–6.5](PRD.md) dan ADR-009 sampai ADR-016 di [.agents/memory.md](.agents/memory.md).

---

## 5. Code style

| Aturan | Nilai |
|---|---|
| Python | >= 3.12 |
| Formatter / linter | `ruff format`, `ruff check` |
| Panjang baris | 100 karakter |
| Type hint | Wajib di semua signature, publik maupun privat |
| Class | `PascalCase` |
| Function / method / variabel | `snake_case` |
| Konstanta | `UPPER_CASE` di level modul |
| Privat | prefiks `_` |
| Mutable default | Dilarang. Pakai `None` atau `field(default_factory=list)` |
| Logging | `logger = logging.getLogger(__name__)` satu per modul. **Tidak ada `print()`** kecuali di blok `__main__` dan CLI output |
| String formatting | f-string. Pengecualian: logging pakai lazy `%s` → `logger.info("loaded %d docs", n)` |
| Docstring | Google-style |
| Path | `pathlib.Path`, bukan `os.path` atau string |

### Error handling

Definisikan exception domain sendiri di `rag_finetune/exceptions.py`, turunan dari satu base:

```python
class RagFinetuneError(Exception):
    """Base untuk semua error yang berasal dari repo ini."""


class IndexNotBuiltError(RagFinetuneError):
    """Index diminta tapi belum pernah dibangun."""
```

Alasannya: pemanggil bisa `except RagFinetuneError` di boundary tanpa menelan `KeyboardInterrupt`
atau bug `AttributeError` kita sendiri.

---

## 6. Testing

- Framework: `pytest`.
- Struktur `tests/` mencerminkan `src/`: `tests/retriever/test_dense.py`.
- Setiap komponen baru butuh minimal: **satu happy path + satu edge case** (input kosong atau invalid).
- Jangan panggil Ollama atau load model asli di unit test. Pakai fake yang mengimplementasikan ABC yang sama:

```python
class FakeGenerator(BaseGenerator):
    """Generator deterministik untuk test; selalu mengembalikan echo prompt."""

    def generate(self, prompts: list[str]) -> list[str]:
        return [f"echo: {p}" for p in prompts]
```

- Test yang butuh service eksternal diberi mark `@pytest.mark.integration` dan tidak jalan secara default.

---

## 7. Konfigurasi

Satu sumber kebenaran: **model Pydantic di `config/schema.py`**. YAML hanya data yang di-validate masuk.

- Jangan baca `os.environ` di tengah-tengah kode. Semua env var masuk lewat `Settings` di `config/`.
- Jangan pernah meneruskan `dict` konfigurasi mentah ke dalam komponen. Teruskan objek config yang sudah tervalidasi, atau lebih baik: hanya field yang komponen itu butuhkan.
- Secret (API key) tidak pernah masuk ke YAML yang di-commit. Pakai `.env` + `SecretStr`.

---

## 8. Yang harus dilakukan agent sebelum menyerahkan pekerjaan

1. `ruff format . && ruff check .` bersih.
2. `pytest` hijau. Kalau ada yang gagal, **laporkan apa adanya** beserta outputnya — jangan dilewati diam-diam.
3. Komponen baru punya blok `__main__` atau doctest yang benar-benar jalan.
4. Keputusan arsitektur baru dicatat di [.agents/memory.md](.agents/memory.md).
5. Kalau menyimpang dari PRD atau ROADMAP, sebutkan secara eksplisit di ringkasan.

## 9. Yang dilarang

- Menambah dependensi baru tanpa alasan tertulis di `.agents/memory.md`.
- Menulis abstraksi untuk kebutuhan yang belum ada ("nanti mungkin perlu multi-tenant").
- Menyalin-tempel kode FlashRAG apa adanya. Pahami, lalu tulis ulang dengan nama dan struktur repo ini.
- `except Exception: pass`.
- Membuat file `utils.py` raksasa sebagai tempat sampah.
