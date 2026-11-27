import re
import pandas as pd


assignments_df = pd.read_csv("MultiLangSAPBert_results.csv", sep=';')
map_df = pd.read_csv("snomed_icd_10_map.csv", sep=';')


assignments_df["snomed id"] = (
    assignments_df["snomed id"].astype(str).str.strip()
)
assignments_df["mention"] = assignments_df["mention"].astype(str).str.strip()

snomed_col = "SNOMED CT"
icd_col = "ICD-10"
reliability_col = "Reliability"

map_df[snomed_col] = map_df[snomed_col].astype(str).str.strip()
map_df[icd_col] = map_df[icd_col].astype(str).str.strip().str.upper()
map_df[reliability_col] = map_df[reliability_col].astype(str).str.strip()


conn_clean = map_df["Connection reliability"].astype(str).str.strip()


'''ALWAYS CONNECTION'''
ground_truth_df = map_df[
    conn_clean.str.contains(r"\balways\b", case=False, regex=True, na=False)
].copy()

ground_truth_df["icd_clean"] = ground_truth_df[icd_col].str.replace(".", "", regex=False)
ground_truth_df["icd_3char"] = ground_truth_df["icd_clean"].str[:3]


valid_pairs = set(zip(ground_truth_df["icd_clean"], ground_truth_df[snomed_col])) # includes 1 icd to many correct snomed
valid_3char_pairs = set(zip(ground_truth_df["icd_3char"], ground_truth_df[snomed_col]))

icd_to_snomeds = (
    ground_truth_df.groupby("icd_clean")[snomed_col].apply(set).to_dict()
)

def extract_icd_code(mention: str):
    if not isinstance(mention, str):
        return None

    # Matches:
    # - 3-char ICD-10 category not immediately followed by more digits (e.g., "I45 ", "A41-")
    # - Dotted ICD-10 code up to 4 additional chars after '.' (e.g., "S42.41", "E11.12", "I45.10")
    match = re.match(
        r"^\s*([A-Za-z]\d{2}(?:\.\d{1,4})?)(?![A-Za-z0-9.])",
        mention,
    )
    if not match:
        return None

    return match.group(1).upper().replace(".", "")


assignments_df["extracted_icd"] = assignments_df["mention"].apply(extract_icd_code)

icd_like_df = assignments_df[assignments_df["extracted_icd"].notna()].copy()

def check_classification(row, fallback=True):
    code = row["extracted_icd"]
    snomed_id = row["snomed id"]

    # Exact (normalized) ICD code + SNOMED ID match in ground truth
    if (code, snomed_id) in valid_pairs:
        return True
    if fallback:
        # Fallback check on 3-character category if needed (optional: remove if strict match is required)
        if (code[:3], snomed_id) in valid_3char_pairs:
            return True
    return False


icd_like_df["properly_classified"] = icd_like_df.apply(check_classification, axis=1)
icd_like_df["expected_snomed_ids"] = icd_like_df["extracted_icd"].map(
    lambda c: sorted(icd_to_snomeds.get(c, []))
)

valid_exact_icds = set(ground_truth_df["icd_clean"])
valid_3char_icds = set(ground_truth_df["icd_3char"])


def has_ground_truth_connection(code: str) -> bool:
    return (code in valid_exact_icds) or (code[:3] in valid_3char_icds)


icd_like_df["has_gt_connection"] = icd_like_df["extracted_icd"].apply(
    has_ground_truth_connection
)

connectible_df = icd_like_df[icd_like_df["has_gt_connection"]]
unconnectible_df = icd_like_df[~icd_like_df["has_gt_connection"]]

connectible_proper = connectible_df[connectible_df["properly_classified"]]
connectible_not_proper = connectible_df[~connectible_df["properly_classified"]]

print(f"Total ICD-like mentions:               {len(icd_like_df)}")
print(f"  - Exist in ground truth (connectible): {len(connectible_df)}")
print(f"      * Properly classified:             {len(connectible_proper)}")
print(f"      * Not properly classified:         {len(connectible_not_proper)}")
print(f"  - Missing from ground truth:           {len(unconnectible_df)}")

connectible_df.to_csv("connectible_icd_mentions_3char.csv", sep=";", index=False)