"""
config/relevance_profile.py

Embio's relevance profile. Edit EMBIO_REFERENCE and PRIORITY_KEYWORDS
as Embio's scientific and commercial focus sharpens.

IMPORTANT: PRIORITY_KEYWORDS must stay in sync with KEYWORD_CLUSTERS
in dashboard/app.py. If you add a keyword here, add it to the
relevant cluster there too so it appears in the Relevance profile UI.
"""

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


PUBMED_QUERIES = [
    # Electroporation core
    '"electroporation"[Title/Abstract] AND "pancreatic cancer"[Title/Abstract]',
    '"irreversible electroporation"[Title/Abstract] AND pancreas[Title/Abstract]',
    '"electrochemotherapy"[Title/Abstract] AND catheter[Title/Abstract]',
    '"calcium electroporation"[Title/Abstract]',
    '"pulsed electric field"[Title/Abstract] AND cancer[Title/Abstract]',
    # Intraductal / ERCP
    '"intraductal"[Title/Abstract] AND "electroporation"[Title/Abstract]',
    '"ERCP"[Title/Abstract] AND ("electroporation"[Title/Abstract] OR "ablation"[Title/Abstract])',
    '"catheter-based"[Title/Abstract] AND "drug delivery"[Title/Abstract] AND pancrea*[Title/Abstract]',
    '"biliary"[Title/Abstract] AND "electroporation"[Title/Abstract]',
    '"cholangiocarcinoma"[Title/Abstract] AND ("ablation"[Title/Abstract] OR "electroporation"[Title/Abstract])',
    # Biomarkers
    '"pancreatic juice"[Title/Abstract] AND ("biomarker"[Title/Abstract] OR "KRAS"[Title/Abstract])',
    '"pancreatic intraepithelial neoplasia"[Title/Abstract] OR "PanIN"[Title/Abstract]',
    '"early detection"[Title/Abstract] AND "pancreatic cancer"[Title/Abstract]',
    '"liquid biopsy"[Title/Abstract] AND "pancreatic cancer"[Title/Abstract]',
]

CLINICALTRIALS_QUERIES = [
    "electroporation pancreatic cancer",
    "irreversible electroporation pancreas",
    "electrochemotherapy catheter",
    "ERCP ablation pancreatic",
    "intraductal drug delivery pancreatic",
]

EUROPEPMC_QUERIES = [
    'TITLE:electroporation ABSTRACT:"pancreatic cancer"',
    'TITLE:"irreversible electroporation" ABSTRACT:pancreas',
    'ABSTRACT:electrochemotherapy ABSTRACT:catheter',
    'ABSTRACT:"calcium electroporation" ABSTRACT:cancer',
    'ABSTRACT:"pulsed electric field" ABSTRACT:ablation',
]

# ---------------------------------------------------------------------------
# PRIORITY_KEYWORDS
# Must stay in sync with KEYWORD_CLUSTERS in dashboard/app.py.
# ---------------------------------------------------------------------------

PRIORITY_KEYWORDS = [
    # Electroporation core
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
    "nanoknife",
    # Catheter / intraductal
    "electroporation catheter",
    "catheter-based electroporation",
    "flexible catheter",
    "catheter",
    "intraductal catheter",
    "bipolar electrode catheter",
    "ring electrode catheter",
    "microcatheter",
    "intraductal electroporation",
    "pancreatic duct electroporation",
    "ercp electroporation",
    "ercp guided ablation",
    # Oncology / clinical
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
    "endoscopic ablation",
    "biliary electroporation",
    # Drug delivery
    "drug delivery",
    "localized drug delivery",
    "intratumoral drug delivery",
    "pancreatic duct drug delivery",
    "catheter-based drug delivery",
    "endoluminal drug delivery",
    # Biomarkers / diagnostics
    "pancreatic juice biomarker",
    "pancreatic liquid biopsy",
    "pancreatic juice",
    "pancreatic cyst fluid",
    "kras pancreatic",
    "tp53 pancreatic",
    "circulating tumor dna",
    "ctdna",
    "exosomes pancreatic",
    "early pancreatic cancer detection",
    "pdac early diagnosis",
    "ercp pancreatic juice",
    "pancreatic duct sampling",
    # Adjacent / strategic
    "eus-guided",
    "endoscopic ultrasound",
    "clinical feasibility",
    "safety",
    "organoid pancreatic cancer",
]

SCORING_WEIGHTS = {
    "semantic":  0.50,
    "keyword":   0.30,
    "recency":   0.15,
    "feedback":  0.05,
}

DEFAULT_MIN_RELEVANCE = 0.55
