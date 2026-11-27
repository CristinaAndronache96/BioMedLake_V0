import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


active_snomed = pd.read_csv('Active_SnomedINTL_GPSRelease_PRODUCTION.csv',
                            sep=';',
                            keep_default_na=False,
                            usecols=['ConceptID', 'USPreferredTerm'],   # only load specific columns
                            )

names = active_snomed['USPreferredTerm'].astype(str).tolist()
concept_ids = active_snomed['ConceptID'].values
number_of_snomed_codes = len(names)


models = {'SapBERT': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
          'Clinical ModernBERT': 'Simonlee711/Clinical_ModernBERT',
          'BioClinical MordernBERT': 'thomas-sounack/BioClinical-ModernBERT-large',
          'PubMedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',  # PubMedBERT is precursor to SapBERT
          'MultiLanguage SapBERT': 'cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR'
          }

tokenizer = AutoTokenizer.from_pretrained(models['MultiLanguage SapBERT'])
model = AutoModel.from_pretrained(models['MultiLanguage SapBERT']).cuda().eval()


bs = 128
all_embs = []

with torch.no_grad():
    for i in tqdm(range(0, number_of_snomed_codes, bs)):
        print(i)
        batch = names[i:i + bs]
        toks = tokenizer(
            batch,
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        cls_rep = model(**toks).last_hidden_state[:, 0, :]   # [B, 768]
        # L2-normalize
        cls_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)

        all_embs.append(cls_rep.cpu().numpy())

embeddings = np.vstack(all_embs).astype(np.float32)   # [N, 768]

# Save as a single .npy holding a dict
np.save(
    f'snomed_MultiLanguageSapBERT.npy',
    {'concept_id': concept_ids,          # [N]
        'term': np.array(names, dtype=object),  # [N]
        'embedding': embeddings,            # [N, 768] float32
    },
    allow_pickle=True,
)
