# CLAUDE.md

Repo ini sudah punya dokumentasi operasionalnya sendiri untuk agent. File ini hanya
penunjuk — **jangan duplikasi isinya di sini**, supaya tidak ada dua sumber kebenaran
yang bisa saling menyimpang.

Baca berurutan, sebelum menulis baris kode pertama di sesi ini:

1. [AGENTS.md](AGENTS.md) — prinsip utama, Zen Workflow 5 langkah, class vs function,
   struktur proyek, dan batas dependensi antar-layer. Ini sumber kebenaran untuk *gaya kode*.
2. [.agents/memory.md](.agents/memory.md) — status fase sekarang, seluruh ADR yang sudah
   final, pelajaran dari bug sebelumnya, dan pertanyaan terbuka yang **harus ditanyakan ke
   pengguna, bukan ditebak**. Ini sumber kebenaran untuk *keputusan yang sudah diambil*.
3. [docs/ROADMAP.md](docs/ROADMAP.md) — fase mana yang sedang dikerjakan dan exit criteria-nya.
   Jangan mengerjakan fase di depan fase sekarang.
4. [CONTRIBUTING.md](CONTRIBUTING.md) — mekanika branch, commit, dan gate sebelum menyerahkan
   pekerjaan. §3.1 soal strategi branch: `develop` adalah working branch, `main` hanya rilis.
5. [PRD.md](PRD.md) — apa yang dibangun dan mengapa, kontrak data, acceptance criteria.

## Memori lintas-sesi

Selain dokumen di atas, memori lintas-sesi tentang pengguna (preferensi gaya kode,
koreksi yang pernah diberikan) tersimpan di luar repo ini dan dimuat otomatis di awal
tiap sesi lewat `MEMORY.md`. Dokumen di atas tetap yang mengikat untuk apa pun yang
spesifik ke repo ini — kalau ada konflik, menang isi repo, bukan memori sesi lama.
