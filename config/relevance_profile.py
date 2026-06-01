"""
config/relevance_profile.py

Updated with domain-expert keyword list.
Two distinct relevance clusters identified from mentor input:
  1. Electroporation delivery — catheter-based, intraductal, ERCP-guided
  2. Diagnostics / biomarkers — pancreatic juice, liquid biopsy, early detection

Both matter to Embio. EMBIO_REFERENCE and queries reflect both.
"""

# ---------------------------------------------------------------------------
# Semantic reference text
# ---------------------------------------------------------------------------
# This is the anchor for ALL semantic scoring. Every document is compared
# against this vector. The more specific and accurate this is to Embio's
# actual focus, the better the scoring separates relevant from noise.

EMBIO_REFERENCE = """
Embio Medical AB is a Swedish medtech startup developing a flexible catheter
platform for electroporation and electrochemotherapy, with a primary focus on
intraductal and ERCP-guided delivery in pancreatic cancer and biliary tract
disease.

Core technical areas: catheter-based electroporation, intraductal
electroporation, ERCP electroporation catheter, bipolar and ring electrode
catheter design, low-voltage electroporation, reversible and irreversible
electroporation, non-thermal ablation, localized drug delivery, endoluminal
drug delivery, intratumoral chemotherapy, microcatheter drug delivery.

Clinical focus: pancreatic ductal adenocarcinoma (PDAC), pancreatic duct
intervention, bile duct electroporation, cholangiocarcinoma, early pancreatic
cancer detection, pancreatic intraepithelial neoplasia (PanIN), endoscopic
ablation, biliary tract ablation, ERCP-guided procedures.

Diagnostic angle: pancreatic juice biomarkers, pancreatic liquid biopsy,
KRAS and TP53 mutations in pancreatic juice, circulating tumor DNA, exosomes
in pancreatic cancer, pancreatic cyst fluid biomarkers, catheter aspiration
for biomarker collection, ERCP pancreatic juice collection, pancreatic duct
sampling for early detection.

Strategic context: regulatory pathway for active catheter devices, CE marking,
clinical feasibility of intraductal devices, first-in-human catheter studies,
organoid drug testing for pancreatic cancer.
""".strip()


# ---------------------------------------------------------------------------
# PubMed queries
# ---------------------------------------------------------------------------
# Organised into three clusters matching Embio's focus areas.
# Each query targets a distinct facet to maximise coverage without
# excessive overlap.

PUBMED_QUERIES = [
    # --- Cluster 1: Core electroporation ---
    '"electroporation"[Title/Abstract] AND "pancreatic cancer"[Title/Abstract]',
    '"irreversible electroporation"[Title/Abstract] AND pancreas[Title/Abstract]',
    '"electrochemotherapy"[Title/Abstract] AND catheter[Title/Abstract]',
    '"calcium electroporation"[Title/Abstract]',
    '"pulsed electric field"[Title/Abstract] AND cancer[Title/Abstract]',

    # --- Cluster 2: Intraductal / ERCP / catheter delivery ---
    '"intraductal"[Title/Abstract] AND "electroporation"[Title/Abstract]',
    '"ERCP"[Title/Abstract] AND ("electroporation"[Title/Abstract] OR "ablation"[Title/Abstract])',
    '"catheter-based"[Title/Abstract] AND "drug delivery"[Title/Abstract] AND pancrea*[Title/Abstract]',
    '"biliary"[Title/Abstract] AND "electroporation"[Title/Abstract]',
    '"cholangiocarcinoma"[Title/Abstract] AND ("ablation"[Title/Abstract] OR "electroporation"[Title/Abstract])',

    # --- Cluster 3: Biomarkers / diagnostics ---
    '"pancreatic juice"[Title/Abstract] AND ("biomarker"[Title/Abstract] OR "KRAS"[Title/Abstract])',
    '"pancreatic intraepithelial neoplasia"[Title/Abstract] OR "PanIN"[Title/Abstract]',
    '"early detection"[Title/Abstract] AND "pancreatic cancer"[Title/Abstract]',
    '"liquid biopsy"[Title/Abstract] AND "pancreatic cancer"[Title/Abstract]',
]


# ---------------------------------------------------------------------------
# ClinicalTrials queries
# ---------------------------------------------------------------------------

CLINICALTRIALS_QUERIES = [
    "electroporation pancreatic cancer",
    "irreversible electroporation pancreas",
    "electrochemotherapy catheter",
    "ERCP ablation pancreatic",
    "intraductal drug delivery pancreatic",
]


# ---------------------------------------------------------------------------
# Priority keywords — organised by cluster
# ---------------------------------------------------------------------------
# All clusters are weighted equally in scoring.
# Grouping is for human readability and maintenance only.

PRIORITY_KEYWORDS = [

    # --- Electroporation core ---
    "electroporation",
    "irreversible electroporation",
    "reversible electroporation",
    "electrochemotherapy",
    "calcium electroporation",
    "pulsed electric field",
    "pef",
    "non-thermal ablation",
    "low voltage electroporation",
    "electroporation parameters",
    "electroporation electrode design",

    # --- Catheter / device ---
    "electroporation catheter",
    "catheter-based electroporation",
    "flexible catheter",
    "intraductal catheter",
    "bipolar electrode catheter",
    "ring electrode catheter",
    "microcatheter",
    "endoluminal",

    # --- Intraductal / ERCP ---
    "intraductal electroporation",
    "pancreatic duct electroporation",
    "ercp electroporation",
    "ercp guided ablation",
    "intraductal ablation",
    "pancreatic duct ablation",
    "endoscopic ablation",
    "endoscopic irreversible electroporation",
    "biliary electroporation",
    "bile duct electroporation",
    "biliary tract ablation",
    "biliary catheter",

    # --- Drug delivery ---
    "localized drug delivery",
    "intratumoral drug delivery",
    "pancreatic duct drug delivery",
    "catheter-based drug delivery",
    "local chemotherapy pancreatic",
    "endoluminal drug delivery",

    # --- Oncology / clinical ---
    "pancreatic cancer",
    "pancreatic ductal adenocarcinoma",
    "pdac",
    "pancreatic adenocarcinoma",
    "locally advanced pancreatic cancer",
    "cholangiocarcinoma",
    "pancreatic intraepithelial neoplasia",
    "panin",
    "ablation",
    "tumor ablation",
    "minimally invasive",
    "first-in-human",
    "interventional oncology",
    "locoregional therapy",

    # --- Diagnostics / biomarkers ---
    "pancreatic juice biomarker",
    "pancreatic liquid biopsy",
    "pancreatic juice",
    "pancreatic cyst fluid",
    "pancreatic fluid biomarker",
    "kras pancreatic",
    "tp53 pancreatic",
    "circulating tumor dna",
    "ctdna",
    "exosomes pancreatic",
    "early pancreatic cancer detection",
    "pdac early diagnosis",
    "ercp pancreatic juice",
    "pancreatic duct sampling",
    "catheter aspiration biomarker",

    # --- Adjacent / strategic ---
    "organoid pancreatic cancer",
    "nanoknife",
    "eus-guided",
    "endoscopic ultrasound",
    "clinical feasibility",
]


# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# ---------------------------------------------------------------------------

SCORING_WEIGHTS = {
    "semantic":  0.50,
    "keyword":   0.30,
    "recency":   0.15,
    "feedback":  0.05,
}

DEFAULT_MIN_RELEVANCE = 0.55