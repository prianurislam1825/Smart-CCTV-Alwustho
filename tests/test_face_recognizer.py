
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.vision.face_recognizer import FaceRecognizer


def main():
    registry_path = BASE_DIR / "data" / "face_registry.json"

    recognizer = FaceRecognizer(
        registry_path=registry_path,
        similarity_threshold=0.45,
    )

    print("=== FACE RECOGNIZER TEST ===")
    print("Jumlah identitas:", len(recognizer.people))

    for person_id, person in recognizer.people.items():
        print(
            f"- ID: {person_id} | "
            f"Nama: {person['name']} | "
            f"Jumlah embedding: {len(person['embeddings'])}"
        )

    # Uji embedding dari registry terhadap dirinya sendiri.
    first_person_id = next(iter(recognizer.people))
    first_person = recognizer.people[first_person_id]
    sample_embedding = first_person["embeddings"][0]

    result = recognizer.match_embedding(sample_embedding)

    print("\n=== SELF-MATCH TEST ===")
    print("Hasil:", result)

    assert result["recognized"], "Embedding tidak dikenali!"
    assert result["person_id"] == first_person_id, (
        "Identitas hasil pencocokan tidak sesuai!"
    )

    print("\nLULUS: modul dapat membaca registry dan mencocokkan embedding.")


if __name__ == "__main__":
    main()
