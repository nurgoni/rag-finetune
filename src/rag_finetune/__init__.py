"""rag-finetune — rewrite terdidik FlashRAG.

Paket ini sengaja tidak meng-import submodul apa pun. Mengimpornya harus murah dan
bebas I/O: tidak ada model yang dimuat, tidak ada file yang dibuka, tidak ada koneksi
jaringan (AGENTS.md §1). Itu juga sebabnya `__version__` ditulis literal alih-alih
dibaca lewat `importlib.metadata`, yang akan menyentuh disk.

Versi di sini harus sejalan dengan `version` di `pyproject.toml`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
