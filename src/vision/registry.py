"""Registry wajah: baca, validasi, ganti nama, dan simpan secara atomik."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class RegistryError(RuntimeError):
    """Registry rusak atau strukturnya tidak valid."""


def validate_registry(registry) -> dict:
    """Validasi struktur registry minimal: {'people': [{person_id, name, embeddings}]}."""
    if not isinstance(registry, dict):
        raise RegistryError("Registry harus berupa objek JSON.")

    people = registry.get("people")

    if people is None or not isinstance(people, list):
        raise RegistryError("Registry tidak memiliki daftar 'people' yang valid.")

    for person in people:
        if not isinstance(person, dict):
            raise RegistryError("Entri 'people' harus berupa objek.")

        if not person.get("person_id") or not person.get("name"):
            raise RegistryError("Entri people tanpa person_id atau name.")

    return registry


def load_registry(registry_path: Path) -> dict:
    """Muat dan validasi registry. Melempar FileNotFoundError / RegistryError."""
    registry_path = Path(registry_path)

    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Registry tidak ditemukan: {registry_path}"
        )

    try:
        with registry_path.open("r", encoding="utf-8") as file:
            registry = json.load(file)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Registry JSON rusak: {exc}") from exc

    return validate_registry(registry)


def save_registry_atomic(registry_path: Path, registry: dict) -> None:
    """Simpan registry secara atomik: tulis ke file sementara lalu replace."""
    registry_path = Path(registry_path)
    validate_registry(registry)

    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Tulis ke file sementara satu folder agar replace tetap di partisi sama.
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=registry_path.name,
        suffix=".tmp",
        dir=str(registry_path.parent),
    )

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(registry, file, ensure_ascii=False, indent=2)
        os.replace(temp_name, registry_path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def rename_person(registry_path: Path, person_id: str, new_name: str) -> dict:
    """
    Ganti nama satu identitas tanpa mengubah embedding.

    Aturan:
    - Jika person_id tidak ada: raise KeyError (tidak membuat duplikat).
    - Embedding yang valid tidak disentuh sama sekali.
    - Registry disimpan kembali secara atomik setelah nama diganti.
    """
    if not person_id:
        raise ValueError("person_id tidak boleh kosong.")

    if not new_name or not new_name.strip():
        raise ValueError("Nama baru tidak boleh kosong.")

    registry = load_registry(registry_path)

    target = None

    for person in registry["people"]:
        if person.get("person_id") == person_id:
            target = person

    if target is None:
        raise KeyError(
            f"Identitas {person_id} tidak ada di registry; "
            "nama tidak diganti."
        )

    before = {
        "name": target.get("name"),
        "n_embeddings": len(target.get("embeddings", [])),
    }

    if target.get("name") == new_name:
        return {
            "changed": False,
            "person_id": person_id,
            "name": new_name,
            **before,
        }

    target["name"] = new_name
    save_registry_atomic(registry_path, registry)

    return {
        "changed": True,
        "person_id": person_id,
        "old_name": before["name"],
        "name": new_name,
        "n_embeddings": before["n_embeddings"],
    }
