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


# -----------------------------------------------------------------------------
# 3) Modelos de recuperación: TF-IDF (coseno) y BM25
# -----------------------------------------------------------------------------
import math

class SearchEngine:
    """Motor de búsqueda term-at-a-time sobre un InvertedIndex.

    Para cada término de la consulta se recorren SOLO sus postings y se suman
    las contribuciones en un acumulador numpy de tamaño N; los candidatos son
    los documentos con acumulador > 0 (nunca se itera documento a documento).
    """

    MODELS = ("tfidf", "bm25")

    def __init__(self, index, k1=1.5, b=0.75, snippet_len=300):
        self.index = index
        self.k1, self.b, self.snippet_len = float(k1), float(b), int(snippet_len)
        N = index.N
        df = index.df.astype(np.float64)

        # idf de cada modelo, indexado por term_id
        self.idf_tfidf = np.log(N / df)                               # log(N/df)
        self.idf_bm25 = np.log(1.0 + (N - df + 0.5) / (df + 0.5))     # siempre > 0

        # Norma L2 de cada vector de documento TF-IDF (precalculada una vez).
        # En cada lista de postings un doc aparece una sola vez -> += es seguro.
        norm_sq = np.zeros(N, dtype=np.float64)
        for tid, (doc_idx, tf) in index.postings.items():
            w = (1.0 + np.log(tf)) * self.idf_tfidf[tid]
            norm_sq[doc_idx] += w * w
        self.doc_norm = np.sqrt(norm_sq)

        # Parte del denominador de BM25 que solo depende del documento
        self.bm25_len_norm = self.k1 * (1.0 - self.b + self.b * index.doc_len / index.avgdl)

    # ---------------------------------------------------------------- scoring
    def _query_terms(self, query):
        """Términos de la consulta presentes en el vocabulario -> Counter(term_id)."""
        vocab = self.index.vocab
        return Counter(vocab[t] for t in preprocess(query) if t in vocab)

    def _acc_tfidf(self, qtf):
        acc = np.zeros(self.index.N, dtype=np.float64)
        # Vector de consulta con el mismo esquema (1+log tf)*idf
        qw = {tid: (1.0 + math.log(c)) * self.idf_tfidf[tid] for tid, c in qtf.items()}
        qnorm = math.sqrt(sum(w * w for w in qw.values()))
        if qnorm == 0.0:                 # solo términos con df = N (idf = 0)
            return acc
        for tid, w_q in qw.items():
            doc_idx, tf = self.index.postings[tid]
            acc[doc_idx] += w_q * (1.0 + np.log(tf)) * self.idf_tfidf[tid]
        # coseno = producto escalar / (|q| |d|); docs con norma 0 se quedan en 0
        np.divide(acc, self.doc_norm * qnorm, out=acc, where=self.doc_norm > 0)
        return acc

    def _acc_bm25(self, qtf):
        acc = np.zeros(self.index.N, dtype=np.float64)
        for tid in qtf:                  # cada término distinto de la consulta, una vez
            doc_idx, tf = self.index.postings[tid]
            tf = tf.astype(np.float64)
            acc[doc_idx] += self.idf_bm25[tid] * tf * (self.k1 + 1.0) / (
                tf + self.bm25_len_norm[doc_idx])
        return acc

    def scores(self, query, model="bm25"):
        """Vector denso (N,) con la puntuación de cada documento (para pruebas)."""
        model = self._check_model(model)
        qtf = self._query_terms(query)
        if not qtf:
            return np.zeros(self.index.N, dtype=np.float64)
        return self._acc_tfidf(qtf) if model == "tfidf" else self._acc_bm25(qtf)

    def _check_model(self, model):
        model = str(model).lower().strip()
        if model not in self.MODELS:
            raise ValueError(f"model debe ser uno de {self.MODELS}, no {model!r}")
        return model

    # ----------------------------------------------------------------- search
    def _snippet(self, doc_idx):
        """≈ primeros snippet_len caracteres del texto, cortando en un espacio."""
        text = self.index.texts[doc_idx] or self.index.titles[doc_idx]
        if len(text) <= self.snippet_len:
            return text
        cut = text.rfind(" ", 0, self.snippet_len)
        return text[: cut if cut > 0 else self.snippet_len].rstrip() + "..."

    def search(self, query, model="bm25", top_k=10):
        """Devuelve los top_k documentos con score > 0, de mayor a menor score.

        Desempate determinista: a igual score, menor doc_idx (orden del corpus).
        Consulta vacía, solo stopwords o términos desconocidos -> [].
        """
        model = self._check_model(model)
        top_k = int(top_k)
        if top_k < 1:
            return []
        acc = self.scores(query, model)
        cand = np.flatnonzero(acc > 0)               # solo docs tocados por postings
        if cand.size == 0:
            return []
        s = acc[cand]
        if cand.size > top_k:
            # Umbral = k-ésimo mayor score; se conservan TODOS los empatados con él
            kth = s[np.argpartition(-s, top_k - 1)[top_k - 1]]
            keep = s >= kth
            cand, s = cand[keep], s[keep]
        order = np.lexsort((cand, -s))[:top_k]        # clave 1: -score, clave 2: doc_idx
        return [
            {
                "rank": r,
                "document_id": self.index.doc_ids[i],
                "score": float(acc[i]),
                "title": self.index.titles[i],
                "snippet": self._snippet(i),
            }
            for r, i in enumerate(cand[order].tolist(), start=1)
        ]


# -----------------------------------------------------------------------------
# 4) Servicio web (FastAPI, solo JSON)
#    Arranque:  uvicorn app:app --host 0.0.0.0 --port 8000
# -----------------------------------------------------------------------------
import os
import time
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

CORPUS_PATH = os.environ.get(
    "CORPUS_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.jsonl"))
MAX_QUERY_CHARS = 1000
MAX_TOP_K = 100

STATE = {}     # se rellena en el arranque: index, engine, tiempo de carga


@asynccontextmanager
async def lifespan(_app):
    """Al arrancar: lee corpus.jsonl y construye índice + motor (una sola vez)."""
    t0 = time.perf_counter()
    index = InvertedIndex().build(load_corpus(CORPUS_PATH))
    STATE["index"] = index
    STATE["engine"] = SearchEngine(index)
    STATE["startup_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[app] Índice listo: {index.N} documentos, {len(index.vocab)} términos "
          f"en {STATE['startup_seconds']} s", flush=True)
    yield
    STATE.clear()


class SearchRequest(BaseModel):
    """Cuerpo de POST /search."""
    query: str = Field(..., description="Texto libre de la consulta (no vacío, máx. 1000 caracteres)")
    model: str = Field("bm25", description="'bm25' o 'tfidf' (sin distinguir mayúsculas)")
    top_k: int = Field(10, ge=1, le=MAX_TOP_K, description="Número de resultados, 1..100")

    @field_validator("query")
    @classmethod
    def _query_ok(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("query no puede estar vacía")
        if len(v) > MAX_QUERY_CHARS:
            raise ValueError(f"query no puede superar {MAX_QUERY_CHARS} caracteres")
        return v

    @field_validator("model", mode="before")
    @classmethod
    def _model_ok(cls, v):
        if not isinstance(v, str):
            raise ValueError("model debe ser una cadena: 'bm25' o 'tfidf'")
        v = v.strip().lower()
        if v not in SearchEngine.MODELS:
            raise ValueError("model debe ser 'bm25' o 'tfidf'")
        return v


class SearchResult(BaseModel):
    rank: int
    document_id: str
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    model: Literal["bm25", "tfidf"]
    results: List[SearchResult]


# Sin /docs ni /redoc (páginas HTML): el servicio es solo JSON.
app = FastAPI(
    title="SciFact IR - TF-IDF y BM25",
    description="Recuperación léxica sobre BeIR/scifact. Solo JSON.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


@app.get("/")
def root():
    """Descripción mínima del servicio (JSON)."""
    return {"service": "SciFact IR (TF-IDF coseno y BM25)",
            "endpoints": {"GET /health": "disponibilidad",
                          "POST /search": {"query": "str", "model": "bm25 | tfidf",
                                           "top_k": "int 1..100"}}}


@app.get("/health")
def health():
    return {"status": "ok", "documents": STATE["index"].N, "models": ["bm25", "tfidf"]}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    # Consulta sin términos útiles (p. ej. solo stopwords) -> 200 con results []
    results = STATE["engine"].search(req.query, req.model, req.top_k)
    return {"model": req.model, "results": results}
