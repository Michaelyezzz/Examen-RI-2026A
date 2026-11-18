# Examen ICCD753 – Recuperación de Información (corpus SciFact)

## Qué se construye
Sistema de recuperación léxico sobre BeIR/scifact (docs: _id, title, text; qrels: BeIR/scifact-qrels, partición test) con dos modelos propios, TF-IDF con coseno y BM25, expuestos como servicio web JSON (FastAPI) publicado en internet.
Entregables: (1) URL pública del servicio, (2) URL pública del notebook ejecutado con outputs.

## Prohibido
- Embeddings, modelos neuronales, chunking. La unidad es el documento completo con su _id original (str).
- Página web / frontend. Solo API JSON.
- TfidfVectorizer, rank_bm25, Whoosh, Elasticsearch, etc. en los modelos finales (solo para contrastar resultados).
- Crear carpetas o archivos fuera de la estructura de abajo.

## Estructura (única permitida)
- examen1bim.ipynb: TODO el trabajo. Celdas del enunciado intactas; lo nuevo va debajo.
- app.py: único archivo de código, GENERADO desde el notebook (no editar a mano). Contiene preprocess, InvertedIndex, SearchEngine y la API FastAPI.
- corpus.jsonl (raíz): _id, title, text. Lo usa el servicio al arrancar.
- requirements.txt, Dockerfile, README.md (se crean en la fase 6).
- .venv/ y .git/ existen pero se ignoran (.gitignore); nunca se suben ni se despliegan. No crear otro entorno.

## Cómo se genera app.py
- Fase 2: primera celda con `%%writefile app.py` (imports + preprocess + InvertedIndex).
- Fase 3: `%%writefile -a app.py` (SearchEngine).
- Fase 5: `%%writefile -a app.py` (API FastAPI).
- Tras cada celda %%writefile, una celda con `import importlib, app; importlib.reload(app)` y reconstruir el índice.
- Contratos: preprocess(text)->list[str]; InvertedIndex.build(docs) con doc_ids (_id originales), N, vocab, df, postings, doc_len, avgdl, titles, texts; SearchEngine(index).search(query, model, top_k=10)->list[{rank, document_id, score, title, snippet}], model in {"tfidf","bm25"}.
- STOPWORDS inglesas como lista literal dentro de app.py (sin nltk.download). Stemmer Porter de nltk.

## Reglas del notebook
- Una sección por fase con encabezado "## N. Título". Editar SOLO la sección de la fase asignada.
- Editar con NotebookEdit o nbformat; nunca reescribir el JSON a mano.
- Markdown en español que explique las decisiones; código comentado.
- Tras cada cambio, `jupyter nbconvert --to notebook --execute --inplace examen1bim.ipynb` (con el python del .venv) debe correr de arriba a abajo sin errores y dejar los outputs guardados.
- Al terminar cada fase: git commit.