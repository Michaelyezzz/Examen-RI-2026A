# =============================================================================
# app.py  -  GENERADO desde examen1bim.ipynb (no editar a mano)
# Sistema de recuperación léxico sobre BeIR/scifact: preprocesamiento,
# índice invertido, modelos TF-IDF (coseno) y BM25, y API FastAPI.
# =============================================================================
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np
from nltk.stem import PorterStemmer

# -----------------------------------------------------------------------------
# 1) Preprocesamiento
# -----------------------------------------------------------------------------

# Stopwords inglesas (lista de NLTK embebida como literal: no requiere
# nltk.download). Se omiten las formas con apóstrofo ("don't"...), porque el
# tokenizador parte por el apóstrofo y nunca las produce.
STOPWORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were be been
being have has had having do does did doing a an the and but if or because as
until while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some
such no nor not only own same so than too very s t can will just don should now
d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn
needn shan shouldn wasn weren won wouldn
""".split())

# Letras griegas frecuentes en biomedicina -> nombre en inglés (TNF-α == TNF-alpha)
GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "κ": "kappa", "λ": "lambda", "μ": "mu", "µ": "mu",
    "π": "pi", "σ": "sigma", "τ": "tau", "ω": "omega",
}
_GREEK_RE = re.compile("|".join(GREEK))
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")       # 1,000 -> 1000
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")        # alfanumérico + guiones internos
MIN_LEN = 2                                                # descarta tokens de 1 carácter

_stemmer = PorterStemmer()


def normalize(text):
    """Minúsculas, griegas -> nombre, y quitar acentos/diacríticos (NFKD)."""
    text = text.lower()
    text = _GREEK_RE.sub(lambda m: GREEK[m.group(0)], text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokenize(text):
    """Tokeniza texto ya normalizado.

    Un token compuesto con guiones ("il-6", "cell-free") genera la forma unida
    ("il6", "cellfree") y además cada parte ("il", "6", "cell", "free").
    """
    text = _THOUSANDS_RE.sub("", text)
    tokens = []
    for tok in _TOKEN_RE.findall(text):
        if "-" in tok:
            parts = tok.split("-")
            tokens.append("".join(parts))
            tokens.extend(parts)
        else:
            tokens.append(tok)
    return tokens


def remove_stopwords(tokens):
    """Quita stopwords y tokens demasiado cortos."""
    return [t for t in tokens if len(t) >= MIN_LEN and t not in STOPWORDS]


@lru_cache(maxsize=None)
def stem(token):
    """Stemming Porter (cacheado: el vocabulario se repite mucho)."""
    return _stemmer.stem(token)


def preprocess(text):
    """Pipeline completo: normalize -> tokenize -> stopwords -> stem."""
    return [stem(t) for t in remove_stopwords(tokenize(normalize(text or "")))]


# -----------------------------------------------------------------------------
# 2) Índice invertido
# -----------------------------------------------------------------------------

def load_corpus(path="corpus.jsonl"):
    """Lee corpus.jsonl (una línea JSON por documento: _id, title, text)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class InvertedIndex:
    """Índice invertido en memoria.

    Atributos tras build():
      doc_ids  list[str]   _id original del documento en la posición doc_idx
      N        int         nº de documentos
      vocab    dict        término -> term_id
      terms    list[str]   term_id -> término
      df       np.ndarray  df[term_id] = nº de documentos que contienen el término
      postings dict        term_id -> (doc_idx: np.int32[], tf: np.int32[]), doc_idx ascendente
      doc_len  np.ndarray  nº de tokens (preprocesados) de cada documento
      avgdl    float       longitud media de documento
      titles, texts list[str]  para construir títulos y snippets
    """

    def __init__(self):
        self.doc_ids, self.titles, self.texts = [], [], []
        self.vocab, self.terms, self.postings = {}, [], {}
        self.df = np.zeros(0, dtype=np.int32)
        self.doc_len = np.zeros(0, dtype=np.int32)
        self.N, self.avgdl = 0, 0.0

    def build(self, docs):
        """Construye el índice a partir de dicts con _id, title y text."""
        tmp_docs, tmp_tfs = {}, {}            # term_id -> listas temporales
        doc_len = []
        for doc_idx, d in enumerate(docs):
            _id = str(d["_id"])
            title, text = d.get("title") or "", d.get("text") or ""
            self.doc_ids.append(_id)
            self.titles.append(title)
            self.texts.append(text)
            tokens = preprocess(title + " " + text)   # texto indexado = título + texto
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                tid = self.vocab.get(term)
                if tid is None:
                    tid = self.vocab[term] = len(self.terms)
                    self.terms.append(term)
                    tmp_docs[tid], tmp_tfs[tid] = [], []
                tmp_docs[tid].append(doc_idx)
                tmp_tfs[tid].append(tf)

        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("Hay _id de documento duplicados")

        # Listas -> arrays numpy (doc_idx ya sale ordenado por construcción)
        self.postings = {tid: (np.asarray(tmp_docs[tid], dtype=np.int32),
                               np.asarray(tmp_tfs[tid], dtype=np.int32))
                         for tid in tmp_docs}
        self.df = np.array([len(self.postings[t][0]) for t in range(len(self.terms))],
                           dtype=np.int32)
        self.doc_len = np.asarray(doc_len, dtype=np.int32)
        self.N = len(self.doc_ids)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        return self

    def get_postings(self, term):
        """Postings de un término YA preprocesado; arrays vacíos si no existe."""
        tid = self.vocab.get(term)
        if tid is None:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        return self.postings[tid]
