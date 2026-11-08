# rag-finetune

Rewrite terdidik [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG): toolkit RAG modular yang
ditulis ulang dari nol supaya alur end-to-end-nya bisa dipahami, dijalankan, dan dimodifikasi
oleh satu orang.

Yang ditukar bukan cakupan arsitekturnya, melainkan jumlah algoritmanya. Keempat pola
orkestrasi RAG diimplementasikan — karena di situ pelajarannya: bedanya Standard RAG,
Iter-RetGen, dan Adaptive-RAG bukan di komponennya (retriever dan generator sama saja),
melainkan **di cara komponen itu dirangkai**.

> **Status: F3 — pipeline walking skeleton.** Belum ada pipeline yang bisa dijalankan.
> Lihat [docs/ROADMAP.md](docs/ROADMAP.md) untuk posisi pengerjaan saat ini.

## Empat arsitektur pipeline

| Tipe | Algoritma pertama | Fase |
|---|---|---|
| `sequential` | Standard RAG | F3 |
| `loop` | Iter-RetGen | F7 |
| `branching` | per-doc generate + majority vote | F8 |
| `conditional` | LLM judger | F9 |

Urutannya ditentukan oleh **jumlah abstraksi baru** yang dibutuhkan, bukan kerumitan konsep —
conditional dikerjakan terakhir meski idenya paling sepele. Rinciannya di [PRD.md §6.4](PRD.md).

## Setup

Prasyarat: Python >= 3.12, [`uv`](https://docs.astral.sh/uv/), dan
[Ollama](https://ollama.com) untuk generasi lokal.

```bash
uv sync --extra dev
uv run pytest
```

Dependensi berat adalah extras opsional — repo ini bisa di-install dan dites tanpa mengunduh
PyTorch (ADR-008):

```bash
uv sync --extra dev --extra sparse   # BM25
uv sync --extra dev --extra dense    # FAISS + embedding
```

## Dokumen

| File | Isi |
|---|---|
| [AGENTS.md](AGENTS.md) | Aturan gaya, Zen Workflow, batas dependensi antar-layer |
| [PRD.md](PRD.md) | Apa yang dibangun dan mengapa; definisi auditabilitas (§6.5) |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Fase F0–F10 beserta exit criteria |
| [.agents/memory.md](.agents/memory.md) | ADR-001..016 dan pertanyaan yang masih terbuka |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, workflow, cara menambah komponen |

## Lisensi

MIT
