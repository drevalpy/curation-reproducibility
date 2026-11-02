"""Curated PubChem CID overrides for drug names PubChem cannot resolve.

These were established by manual review (see the legacy preprocess_drp_data
notebooks) for drug names that carry vendor codes, typos, or synonyms that the
PubChem name endpoint does not match. Applied before falling back to a
dataset-prefixed pseudo-ID.

Combination compounds are deliberately absent: they have no single CID and are
left as pseudo-IDs.
"""

PDX_BRUNA_CID_OVERRIDES: dict[str, int] = {
    "GW843682X (AN-13)": 9826308,
}

#: Barretina et al. 2012 name their 24 compounds by development code as often as
#: by generic name. Measured against PubChem's name endpoint, all 24 resolve
#: without help, so this table is a pinned fallback for the day a lookup goes
#: flaky rather than a set of corrections: without it a transient failure would
#: silently bake a pseudo-ID into the output. Keys are the exact strings in
#: table S11; CIDs are the ones the original notebook's manual review recorded.
CCLE_CID_OVERRIDES: dict[str, int] = {
    "17-AAG": 6505803,  # Tanespimycin
    "AEW541": 11476171,  # NVP-AEW541
    "AZD0530": 10302451,  # Saracatinib
    "AZD6244": 10127622,  # Selumetinib
    "Erlotinib": 176870,
    "Irinotecan": 60838,
    "L-685458": 5479543,
    "LBW242": 11503417,
    "Lapatinib": 208908,
    "Nilotinib": 644241,
    "Nutlin-3": 216345,
    "PD-0325901": 9826528,
    "PD-0332991": 5330286,  # Palbociclib
    "PF2341066": 11626560,  # Crizotinib
    "PHA-665752": 10461815,
    "PLX4720": 24180719,
    "Paclitaxel": 36314,
    "Panobinostat": 6918837,
    "RAF265": 11656518,
    "Sorafenib": 216239,
    "TAE684": 16038120,  # NVP-TAE684
    "TKI258": 135398510,  # Dovitinib
    "Topotecan": 60700,
    "ZD-6474": 3081361,  # Vandetanib
}

BEATAML2_CID_OVERRIDES: dict[str, int] = {
    "Pazopanib (GW786034)": 10113978,
    "Roscovitine (CYC-202)": 160355,
    "Afatinib (BIBW-2992)": 10184653,
    "Axitinib (AG-013736)": 6450551,
    "Foretinib (XL880)": 42642645,
    "Lestaurtinib (CEP-701)": 126565,
    "Masitinib (AB-1010)": 10074640,
    "Vargetef": 135423438,  # typo for Vargatef (Nintedanib)
    "Vemurafenib (PLX-4032)": 42611257,
    "Volasertib (BI-6727)": 10461508,
    "AT-101": 3503,  # Gossypol; "AT 101" is a synonym
    "Baiclein": 5281605,  # typo for Baicalein
    "Dovitinib (CHIR-258)": 135398510,
    "AZD1152-HQPA (AZD2811)": 16007391,
}
