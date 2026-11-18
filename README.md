---
title: SciFact IR TF-IDF BM25
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SciFact IR: TF-IDF y BM25

Servicio web JSON de recuperación de información léxica sobre el corpus
[BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) (5183 resúmenes científicos),
con dos modelos implementados desde cero sobre un índice invertido:

- **TF-IDF** con similitud coseno: `(1 + ln tf) · ln(N/df)`, vectores normalizados L2.
- **BM25**: `k1 = 1.5`, `b = 0.75`, `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`.

Preprocesamiento: minúsculas, letras griegas → nombre, eliminación de acentos, tokenización
alfanumérica (los compuestos con guion generan forma unida y partes: `il-6` → `il6`, `il`),
stopwords inglesas y stemming Porter. Cada documento (título + texto) se indexa completo
con su `_id` original.

Todo el código (`app.py`) se genera desde el notebook `examen1bim.ipynb`, que contiene
además la evaluación (P@10, R@10, NDCG@10) sobre las 300 consultas de test.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Disponibilidad: `{"status":"ok","documents":5183,"models":["bm25","tfidf"]}` |
| POST | `/search` | Búsqueda: body `{"query": str, "model": "bm25" \| "tfidf", "top_k": 1..100}` |
| GET | `/` | Descripción del servicio (JSON) |

- `model` no distingue mayúsculas; por defecto `"bm25"`. `top_k` por defecto `10`.
- Entradas inválidas (query vacía o > 1000 caracteres, modelo desconocido, `top_k` fuera de rango) → **422** con JSON.
- Consulta sin términos indexables (p. ej. solo stopwords) → **200** con `"results": []`.

Respuesta de `/search`:

```json
{
  "model": "bm25",
  "results": [
    {"rank": 1, "document_id": "22843838", "score": 22.1988,
     "title": "...", "snippet": "primeros ~300 caracteres del texto..."}
  ]
}
```

## Ejemplos con curl

Sustituir `BASE` por la URL del Space (o `http://localhost:7860` en local).

```bash
BASE=https://<usuario>-<space>.hf.space

curl -s $BASE/health

curl -s -X POST $BASE/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Vitamin D deficiency is linked to multiple sclerosis", "model": "bm25", "top_k": 5}'

curl -s -X POST $BASE/search \
  -H "Content-Type: application/json" \
  -d '{"query": "aspirin lowers colorectal cancer incidence", "model": "tfidf", "top_k": 3}'

# Error de validación -> 422
curl -s -X POST $BASE/search \
  -H "Content-Type: application/json" \
  -d '{"query": "   ", "model": "bm25"}'
```

## Ejecución local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860

# o con Docker
docker build -t scifact-ir .
docker run -p 7860:7860 scifact-ir
```
