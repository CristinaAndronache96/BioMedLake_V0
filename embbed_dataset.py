import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


dataset_text = pd.read_csv(
    'Dataset_text_descriptions.csv',
    sep=';',
    keep_default_na=False,
    encoding='latin-1',
)


def split_multi(s: str, delims):
    if not s or not s.strip():
        return []
    if isinstance(delims, str):
        delims = (delims,)
    for d in delims:
        s = s.replace(d, '\x00')
    return [p.strip() for p in s.split('\x00') if p.strip()]


rows = []
for _, r in dataset_text.iterrows():
    rid = str(r['index'])

    # ---- DIAG_PRINCIPAL: treat as ONE mention ----
    dp = str(r['DIAG_PRINCIPAL']).strip()
    if dp:
        rows.append((rid, 'DIAG_PRINCIPAL', dp))

    # ---- DIAG_SECUNDARE: split on commas (your requirement) ----
    for m in split_multi(str(r['DIAG_SECUNDARE']), ','):
        rows.append((rid, 'DIAG_SECUNDARE', m))

    # ---- DIAGNOSTICE: comma-split (typically one per comma) ----
    for m in split_multi(str(r['DIAGNOSTICE']), delims=(',', ';', '.')):
        rows.append((rid, 'DIAGNOSTICE', m))

    # ---- EPICRIZA: free text, keep whole ----
    for m in split_multi(str(r['EPICRIZA']), delims=(',', ';', '.')):
        rows.append((rid, 'EPICRIZA', m))

    # ---- proceduri: comma-split (toggle off if not desired) ----
    for m in split_multi(str(r['proceduri']), delims=(',', ';', '.')):
        rows.append((rid, 'proceduri', m))

    # ---- INTERVENTIE_CHIRURGICALA: comma-split (toggle off if not desired) ----
    for m in split_multi(str(r['INTERVENTIE_CHIRURGICALA']), delims=(',', ';', '.')):
        rows.append((rid, 'INTERVENTIE_CHIRURGICALA', m))

mentions_df = pd.DataFrame(rows, columns=['record_index', 'source_field', 'mention'])

# ---------------------------------------------------------------
# 3. Embed every mention with SapBERT (same as before, just per-mention)
# ---------------------------------------------------------------

models = {'SapBERT': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
          'Clinical ModernBERT': 'Simonlee711/Clinical_ModernBERT',
          'BioClinical MordernBERT': 'thomas-sounack/BioClinical-ModernBERT-large',
          'PubMedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',  # SapBERT precursor
          'MultiLanguage SapBERT': 'cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR'
          }

tokenizer = AutoTokenizer.from_pretrained(models['MultiLanguage SapBERT'])
model = AutoModel.from_pretrained(models['MultiLanguage SapBERT']).cuda().eval()

mention_texts = mentions_df['mention'].tolist()

print(len(mention_texts))  # 107506

bs = 128
all_embs = []

with torch.no_grad():
    for i in tqdm(range(0, len(mention_texts), bs)):
        batch = mention_texts[i: i + bs]
        toks = tokenizer(
            batch,
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        cls_rep = model(**toks).last_hidden_state[:, 0, :]   # [B, 768]
        cls_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
        all_embs.append(cls_rep.cpu().numpy())

embeddings = np.vstack(all_embs).astype(np.float32)   # [M, 768]


# ---------------------------------------------------------------
# 4. Save long-format dict
# ---------------------------------------------------------------
np.save(
    'patient_descriptions_MultiLanguageSapBERT.npy',
    {
        'record_index':  mentions_df['record_index'].to_numpy(dtype=object),  # [M]
        'source_field':  mentions_df['source_field'].to_numpy(dtype=object),  # [M]
        'mention':       mentions_df['mention'].to_numpy(dtype=object),       # [M]
        'embedding':     embeddings,                                          # [M, 768] float32
    },
    allow_pickle=True,
)
