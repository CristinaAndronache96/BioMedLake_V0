# Clinical Condition Mapping with BERT Embeddings & SNOMED CT

A pipeline for mapping patient condition mentions to active SNOMED CT concepts using BERT embeddings and evaluating mapping accuracy against existing ICD-10 mappings.

## Pipeline overview

The workflow runs in three stages:

1. **Generate BERT Embeddings:** for patient clinical records/ for each mention of a condition and active SNOMED CT concept descriptions.
2. **Match Best SNOMED Codes:** retrieve closest active SNOMED CT codes by embedding similarity.
3. **Approximate Error via ICD-10 Mappings:** compare predicted SNOMED codes with existing SNOMED-to-ICD-10 mappings to error rates.


## [Google doc spreadsheets with sheets for:](https://docs.google.com/spreadsheets/d/13hXBUul-_x67IWhZwNl4CWjcI1swWH-LdXj0HGgryYw/edit?usp=sharing)

 - Active Snomed Codes

 - Patient dataset

 - ICD-10 to Snomed mappings

 - Results with ICD-10 comparison
