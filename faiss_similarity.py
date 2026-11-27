import numpy as np
import faiss
from collections import defaultdict
import csv


# ---------- Load ----------
snomed_data = np.load('snomed_MultiLangSapBERT.npy', allow_pickle=True).item()
snomed_concept_ids = np.asarray(snomed_data['concept_id'])
snomed_terms = np.asarray(snomed_data['term'], dtype=object)
snomed_embeddings = np.asarray(snomed_data['embedding'], dtype=np.float32)

patient_data = np.load('patient_descriptions_MultiLangSapBERT.npy', allow_pickle=True).item()
patient_record_index = np.asarray(patient_data['record_index'], dtype=object)  # per-mention
patient_mentions = np.asarray(patient_data['mention'],       dtype=object)  # per-mention
patient_embeddings = np.asarray(patient_data['embedding'],     dtype=np.float32)

assert snomed_embeddings.shape[1] == patient_embeddings.shape[1]


# ---------- FAISS ----------
index = faiss.IndexFlatIP(snomed_embeddings.shape[1])
index.add(snomed_embeddings)
print(f"Index: {index.ntotal} SNOMED vectors | {len(patient_mentions)} patient mentions")


K = 100
D, I = index.search(patient_embeddings, K)   # [M, K], [M, K]


THRESHOLD = 0.3
TOP_N = 1      # max matches per mention


aggregate = defaultdict(list)
mention_seen = defaultdict(set)

for m in range(len(patient_mentions)):
    rid = patient_record_index[m]
    ment = patient_mentions[m]

    # Skip empty mentions
    if not isinstance(ment, str) or not ment.strip():
        continue

    if ment not in mention_seen[rid]:
        mention_seen[rid].add(ment)
        aggregate[rid].append({'mention': ment, 'matches': []})

    # Find the mention entry
    entry = next(e for e in aggregate[rid] if e['mention'] == ment)

    # Collect top-N matches above threshold for this mention
    collected = 0
    for rank in range(K):
        j = I[m, rank]
        if j == -1:
            continue
        score = float(D[m, rank])
        if score < THRESHOLD:
            break
        entry['matches'].append({
            'concept_id': snomed_concept_ids[j],
            'term':       snomed_terms[j],
            'score':      score,
        })
        collected += 1
        if collected >= TOP_N:
            break

# ---------- Sort matches inside each mention by score ----------
for patient_id in aggregate:
    for entry in aggregate[patient_id]:
        entry['matches'].sort(key=lambda x: x['score'], reverse=True)


# ---------- Inspect ----------
def show(patient_id):
    print(f"\n=== Patient {patient_id} ===")
    for entry in aggregate[patient_id]:
        ment = entry['mention']
        matches = entry['matches']
        if not matches:
            print(f"\n  [mention] {ment!r}")
            print(f"     (no match ≥ {THRESHOLD})")
            continue
        print(f"\n  [mention] {ment!r}")
        for i, mt in enumerate(matches, 1):
            # print(f"     {i}. {mt['score']:.4f}  {mt['concept_id']}  {mt['term']}")
            print(f"    Score: {mt['score']:.2f}  snomed: {mt['concept_id']}  {mt['term']}")


# Show first 3 patients
for patient_mentions in list(aggregate)[:3]:
    show(patient_mentions)


def save_to_csv(aggregate, filename="patient_mentions.csv"):
    fieldnames = ["patient index", "mention", "score", "snomed id", "snomed description"]

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for patient_index in aggregate:
            for entry in aggregate.get(patient_index, []):
                ment = entry["mention"]
                matches = entry["matches"]

                if not matches:
                    writer.writerow({
                        "patient index": patient_index,
                        "mention": ment,
                        "score": "",
                        "snomed id": "",
                        "stomed description": "",
                    })
                    continue

                for mt in matches:
                    writer.writerow({
                        "patient index": patient_index,
                        "mention": ment,
                        "score": round(mt["score"], 4),
                        "snomed id": mt["concept_id"],
                        "snomed description": mt["term"],
                    })

save_to_csv(aggregate=aggregate, filename='MultiLangSAPBert_results.csv')