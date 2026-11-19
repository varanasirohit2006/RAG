"""Initialize native libraries and local inference on the main thread."""
import os
import time


def warm_runtime():
    started = time.perf_counter()
    print('Loading local search dependencies...', flush=True)
    import pandas
    from backend.rag_pipeline import get_embedding_model
    model = get_embedding_model()
    import langchain_chroma
    model.embed_query('Local search initialization')
    print(f'Local search ready in {time.perf_counter()-started:.2f}s.', flush=True)
    from backend.generation_runtime import get_generation_model, resolve_provider
    from backend.retrieval_runtime import get_reranker
    provider = resolve_provider()
    if provider == 'ollama' or (provider == 'google' and os.getenv('GOOGLE_API_KEY')) or (provider == 'openrouter' and os.getenv('OPENROUTER_API_KEY')):
        get_generation_model()
    try:
        get_reranker(top_n=int(os.getenv('RETRIEVAL_TOP_N', '3')))
    except Exception as exc:
        print(f'Optional reranker setup deferred ({type(exc).__name__}).', flush=True)
    print('Ready. Startup made no model API requests.', flush=True)
