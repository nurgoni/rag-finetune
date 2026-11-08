# CONTRIBUTING

Cara bekerja di `rag-finetune`. Baca [AGENTS.md](AGENTS.md) lebih dulu — di situ aturan
gaya dan alur kerjanya; file ini soal mekanikanya.

---

## 1. Setup

Prasyarat: Python >= 3.12, [`uv`](https://docs.astral.sh/uv/), dan [Ollama](https://ollama.com) untuk generasi lokal.

```bash
git clone <repo-url> rag-finetune
cd rag-finetune

# dependensi inti + tooling dev
uv sync --extra dev

# tambahkan saat mengerjakan retrieval (lihat ADR-008)
uv sync --extra dev --extra sparse     # BM25
uv sync --extra dev --extra dense      # FAISS + embedding (unduh torch, besar)

cp .env.example .env
```

Siapkan model Ollama:

```bash
ollama pull <model>        # nama model lihat configs/base.yaml
ollama list                # pastikan server jalan
```

Verifikasi:

```bash
uv run ruff check .
uv run pytest
uv run python -c "import rag_finetune; print('ok')"
```

---

## 2. Perintah sehari-hari

```bash
uv run ruff format .                  # format
uv run ruff check . --fix             # lint + autofix
uv run pytest                         # unit test (cepat, tanpa jaringan)
uv run pytest -m integration          # test yang butuh Ollama / model nyata
uv run pytest --cov=rag_finetune      # coverage
uv run rag-finetune --help            # CLI
```

Unit test **tidak boleh** butuh jaringan, GPU, atau index yang sudah dibangun.
Apa pun yang butuh itu diberi mark `@pytest.mark.integration`.

---

## 3. Alur kerja perubahan

1. **Cek fase sekarang** di [docs/ROADMAP.md](docs/ROADMAP.md) dan [.agents/memory.md §1](.agents/memory.md). Pekerjaan di luar fase sekarang masuk ke "Pertanyaan terbuka" atau future scope, bukan langsung dikerjakan.
2. **Cek ADR** di [.agents/memory.md §2](.agents/memory.md). Kalau perubahan Anda melawan sebuah ADR, bahas dulu — jangan diam-diam membalikkannya.
3. **Ikuti Zen Workflow** 5 langkah dari [AGENTS.md §2](AGENTS.md): klarifikasi I/O → signature + docstring → implementasi paling sederhana → review mental → contoh penggunaan.
4. **Tulis test** bersamaan dengan kode, bukan sesudahnya. Minimal satu happy path + satu edge case.
5. **Perbarui dokumen** kalau perlu: ADR baru ke `.agents/memory.md`, perubahan kontrak ke `PRD.md`, status fase ke `ROADMAP.md`.
6. **Jalankan gate** di §6 sebelum menyerahkan.

Satu branch, satu perubahan logis. `feat/bm25-retriever`, bukan `feat/retrieval-stuff`.

### 3.1 Strategi branch (ADR-017)

Dua branch permanen:

| Branch | Peran | Siapa yang push langsung |
|---|---|---|
| `main` | **Produksi.** Mencerminkan rilis yang sudah stabil | Tidak ada — hanya lewat merge `develop` di penutupan fase/rilis |
| `develop` | **Integrasi.** Working branch default untuk seluruh pengerjaan fase ROADMAP | Agent dan kontributor, langsung atau lewat PR fitur |

Alur kerja:

1. Checkout dari `develop`, bukan `main`: `git checkout develop && git checkout -b feat/f1-config-schema`.
2. Kerjakan satu fase/perubahan logis di branch fitur itu, mengikuti §3 di atas.
3. Merge balik ke `develop` (PR atau merge langsung untuk kerja solo).
4. `main` hanya diperbarui dengan merge `develop` → `main` di titik rilis/milestone (misal penutupan satu fase ROADMAP yang sudah hijau gate §6), bukan per-commit.

**Mengapa:** `main` harus selalu bisa di-deploy atau dirujuk sebagai keadaan "terakhir yang terbukti benar", sementara pekerjaan fase yang sedang berjalan (F1 dst.) butuh tempat untuk commit bertahap tanpa mengganggu itu.

---

## 4. Cara menambah komponen

Tiga titik ekstensi utama. Pola yang sama berulang di ketiganya: **implementasikan ABC,
daftarkan, tes, demo.** Pipeline tidak pernah perlu diubah.

### Retriever baru

1. Buat `src/rag_finetune/retriever/<nama>.py`
2. Turunkan `BaseRetriever`, implementasikan `search()` dan `batch_search()`
3. Daftarkan: `@register_retriever("<nama>")`
4. Tambahkan field config yang dibutuhkan ke `RetrieverConfig`
5. Test di `tests/retriever/test_<nama>.py` — pakai fixture corpus kecil, jangan unduh apa pun
6. Tambahkan blok `__main__` yang menampilkan hasil pencarian satu query
7. Pakai lewat YAML: `retriever: {type: <nama>}`

### Generator baru

Sama polanya: turunkan `BaseGenerator`, implementasikan `generate(prompts) -> list[str]`,
`@register_generator("<nama>")`. **Fake HTTP client-nya di unit test**; test yang benar-benar
memanggil service diberi mark `integration`.

### Metrik baru

Metrik adalah **fungsi murni**, bukan class — tidak punya state maupun dependency
(lihat [AGENTS.md §3](AGENTS.md)):

```python
@register_metric("my_metric")
def my_metric(prediction: str, golden_answers: list[str]) -> float:
    """Ringkasan satu baris soal apa yang diukur.

    Args:
        prediction: Jawaban hasil generasi, sudah dinormalisasi pemanggil.
        golden_answers: Jawaban acuan; minimal satu.

    Returns:
        Skor dalam rentang [0.0, 1.0].
    """
```

Uji terhadap kasus yang dihitung tangan, termasuk tanda baca, artikel, dan jawaban berganda.

### Judger baru (pipeline conditional)

Judger adalah **class** — ia punya dependency dan state (ADR-012).

1. `src/rag_finetune/judger/<nama>.py`, turunkan `BaseJudger`
2. Implementasikan `judge(item: QAItem) -> JudgeDecision`
3. `@register_judger("<nama>")`, tambahkan field ke `JudgerConfig`
4. **Output yang tak bisa di-parse → default `needs_retrieval=True`**, alasannya ditulis ke `JudgeDecision.reason`. Gagal ke arah aman (FR-7d.3)
5. Bandingkan terhadap `AlwaysRetrieveJudger` dan `NeverRetrieveJudger` — judger yang tidak mengalahkan keduanya tidak menambah nilai apa pun

### Aggregator baru (pipeline branching)

Aggregator adalah **fungsi murni** — tanpa state, tanpa dependency (ADR-012):

```python
@register_aggregator("my_aggregator")
def my_aggregator(candidates: list[Candidate]) -> str:
    """Pilih satu jawaban akhir dari kandidat per-dokumen.

    Args:
        candidates: Satu kandidat per cabang; `score` adalah skor retrieval
            dokumen sumbernya. Minimal satu elemen.

    Returns:
        Jawaban terpilih. String kosong kalau semua kandidat kosong.
    """
```

Wajib **deterministik**: seri harus diputus oleh aturan eksplisit (skor retrieval tertinggi),
bukan oleh urutan iterasi `dict`. Dua run berurutan harus menghasilkan prediksi identik (NFR-5).

### Pipeline baru

Turunkan `BasePipeline`, implementasikan `run(dataset) -> list[QAResult]`.
Pipeline mengorkestrasi komponen; ia **tidak boleh** mengandung logika retrieval,
pemanggilan LLM, atau penyusunan string prompt sendiri — lihat
[AGENTS.md §4 "Aturan khusus `pipeline/`"](AGENTS.md).

Checklist tambahan khusus pipeline:

- [ ] Terdaftar lewat `@register_pipeline("<type>")`; bisa dipilih dari YAML tanpa perubahan kode
- [ ] Setiap putaran retrieve+generate tercatat sebagai `GenerationStep` dengan `label` bermakna dan `query` yang **benar-benar dipakai** putaran itu
- [ ] **Tidak ada `num_llm_calls +=` di kode pipeline** — penghitungan milik `CountingGenerator` (ADR-014)
- [ ] **Uji reduksi ada dan hijau** — satu konfigurasi yang membuat pipeline ini runtuh jadi sequential dan menghasilkan hasil identik, dijalankan dengan `EchoGenerator` + `RandomRetriever` ber-seed (ADR-015). Ini test terpenting di pipeline mana pun
- [ ] **Jejak audit khas arsitekturnya tersimpan** — lihat tabel di [AGENTS.md §4](AGENTS.md). Tanyakan: "kalau pipeline ini gagal dengan cara yang tidak menurunkan skor, artefak mana yang memperlihatkannya?" Kalau tidak ada jawabannya, jejaknya belum cukup
- [ ] Semua metrik dari pipeline ini bisa dihitung ulang dari `predictions.jsonl` saja (FR-8.8)
- [ ] Parameter barunya masuk `provenance` — dua run yang beda hanya di parameter itu harus bisa dibedakan dari `metrics.json` (ADR-016)
- [ ] Tidak ada pemanggilan `random` global; semua keacakan dari `RunConfig.seed`
- [ ] Jalan di test tanpa jaringan
- [ ] Punya diagram dan contoh `steps` nyata di `docs/PIPELINES.md`

Kalau pipeline baru Anda butuh akses logprob token, **berhenti** — baca ADR-010 dulu.
Ollama tidak menjaminnya, dan `BaseGenerator` sengaja belum punya parameter itu.

---

## 5. Konvensi commit

[Conventional Commits](https://www.conventionalcommits.org/). Subjek dalam imperatif, <= 72 karakter.

```
feat(retriever): add BM25 retriever backed by bm25s
fix(config): report full field path on validation error
docs(roadmap): mark phase F2 as complete
refactor(pipeline): extract prompt building from SequentialPipeline
test(evaluation): cover multi-answer exact match
chore(deps): add bm25s to sparse extras
```

Scope yang dipakai: `config`, `schema`, `dataset`, `corpus`, `index`, `retriever`,
`prompt`, `generator`, `judger`, `aggregator`, `pipeline`, `evaluation`, `cli`, `docs`, `deps`.

Body menjelaskan **mengapa**, bukan apa — diff sudah menunjukkan apanya.

---

## 6. Gate sebelum menyerahkan pekerjaan

Semua harus terpenuhi:

- [ ] `uv run ruff format .` dan `uv run ruff check .` bersih
- [ ] `uv run pytest` hijau. **Kalau ada yang gagal, laporkan beserta outputnya** — jangan dilewati diam-diam
- [ ] Type hint lengkap di semua signature baru
- [ ] Docstring Google-style di semua class/fungsi publik baru
- [ ] Review mental Zen dijalankan: nama jelas, nesting <= 2, tidak ada broad `except`, tidak ada magic number
- [ ] Komponen baru punya blok `__main__` atau doctest yang **benar-benar dijalankan**, bukan hanya ditulis
- [ ] File sumber < 300 baris (NFR-3)
- [ ] Dependensi baru punya alasan tertulis di `.agents/memory.md`
- [ ] Penyimpangan dari PRD/ROADMAP disebutkan eksplisit di ringkasan PR

---

## 7. Review checklist

Yang ditanyakan reviewer, kurang-lebih dalam urutan ini:

**Kejelasan** — Bisakah dibaca sekali dan dimengerti? Nama menyebut isinya? Alurnya lurus?

**Boundary** — Arah dependensi masih satu arah ([AGENTS.md §4](AGENTS.md))? `schema/` masih hanya Pydantic + stdlib? Retriever masih tidak tahu soal generator?

**Kontrak** — Data lintas-layer berupa model Pydantic, bukan `dict`? Komponen menerima sub-model yang ia butuhkan, bukan seluruh config?

**Error** — Exception konkret, bukan `Exception`? Pesan error menyebut konteks yang cukup untuk bertindak (nomor baris, nama field, nama model)?

**Test** — Ada happy path dan edge case? Jalan tanpa jaringan? Deterministik?

**Orkestrasi** — Kalau menyentuh `pipeline/`: masih bebas dari logika HTTP, retrieval, dan penyusunan prompt? Uji reduksinya ada dan pakai generator deterministik? Tidak menghitung `num_llm_calls` sendiri?

**Auditabilitas** — Bisa menjawab: kalau perubahan ini gagal dengan cara yang **tidak** menurunkan skor, artefak mana yang memperlihatkannya? Parameter baru masuk `provenance`?

**Scope** — Hanya mengerjakan fase sekarang? Tidak ada abstraksi untuk kebutuhan yang belum ada?

Perubahan ditolak kalau: menyalin kode FlashRAG apa adanya, memasukkan logika training
sebelum F10 (ADR-004), menambah `dict` config mentah (ADR-002), menaruh logika komponen
di dalam `pipeline/` (AGENTS.md §4), mengubah urutan F7→F8→F9 (ADR-009), menambah parameter
`logprobs` ke `BaseGenerator` (ADR-010), atau membuat `utils.py` sebagai tempat sampah.

---

## 8. Kalau macet

1. Baca ulang ADR terkait di [.agents/memory.md](.agents/memory.md) — mungkin sudah pernah diputuskan
2. Cek "Pertanyaan terbuka" §4 di file itu — mungkin ini memang belum diputuskan, dan **jawabannya harus ditanyakan, bukan ditebak**
3. Lihat FlashRAG untuk memahami *konsepnya*, lalu tutup tab-nya dan tulis sendiri (ADR-001)
4. Kalau ambiguitasnya soal tipe input/output, itu Langkah 1 Zen Workflow: berhenti dan tanya
