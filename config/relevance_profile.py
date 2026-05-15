EMBIO_REFERENCE = """
Embio Medical develops a flexible catheter platform for electroporation and
electrochemotherapy, focused on pancreatic cancer. Key topics include
irreversible electroporation, flexible catheter design, pancreatic
adenocarcinoma, minimally invasive ablation, drug delivery enhancement,
electroporation parameters, clinical feasibility, safety, and regulatory
strategy for medtech devices.
""".strip()

PUBMED_QUERIES = [
    '"electroporation"[Title/Abstract] AND "pancreatic cancer"[Title/Abstract]',
    '"electrochemotherapy"[Title/Abstract] AND catheter[Title/Abstract]',
    '"irreversible electroporation"[Title/Abstract] AND pancreas[Title/Abstract]',
    '"percutaneous ablation"[Title/Abstract] AND "pancreatic adenocarcinoma"[Title/Abstract]',
    '"electroporation"[Title/Abstract] AND "drug delivery"[Title/Abstract] AND tumor[Title/Abstract]',
    '"calcium electroporation"[Title/Abstract]',
    '"pulsed electric field"[Title/Abstract] AND cancer[Title/Abstract]',
    '"EUS-guided"[Title/Abstract] AND ablation[Title/Abstract]',
]

CLINICALTRIALS_QUERIES = [
    "electroporation pancreatic cancer",
    "irreversible electroporation pancreas",
    "electrochemotherapy catheter",
]

PRIORITY_KEYWORDS = [
    "electroporation",
    "electrochemotherapy",
    "pancreatic cancer",
    "flexible catheter",
    "irreversible electroporation",
    "ire",
    "ablation",
    "pancreatic adenocarcinoma",
    "drug delivery",
    "electroporation parameters",
    "tumor ablation",
    "minimally invasive",
    "catheter",
    "clinical feasibility",
    "safety",
    "pulsed electric field", "pef", "nanoknife",
    "calcium electroporation", "eus-guided", "endoscopic ultrasound",
    "pancreatic ductal adenocarcinoma", "pdac",
    "locoregional therapy", "interventional oncology",
    "first-in-human", "locally advanced pancreatic cancer",
]

SCORING_WEIGHTS = {
    "semantic": 0.50,
    "keyword": 0.30,
    "recency": 0.15,
    "feedback": 0.05,
}

DEFAULT_MIN_RELEVANCE = 0.55
