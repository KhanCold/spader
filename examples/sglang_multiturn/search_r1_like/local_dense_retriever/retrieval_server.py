# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

import argparse
import json
import logging
import queue
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("retrieval_server")


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False, gpu_id: int = 0):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    if use_fp16 and "cuda" in device:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer, device


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, gpu_id=0):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer, self.device = load_model(model_path=model_path, use_fp16=use_fp16, gpu_id=gpu_id)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class Reranker:
    def __init__(self, model_path, param_device=None, use_fp16=True):
        if param_device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = param_device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)
        if use_fp16 and "cuda" in self.device:
            self.model.half()

    @torch.no_grad()
    def rerank(self, query: str, docs: list[str], batch_size: int = 16) -> list[float]:
        if len(docs) == 0:
            return []
        pairs = [[query, doc] for doc in docs]
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self.device)
            out = self.model(**inputs, return_dict=True)
            batch_scores = out.logits.view(-1).float()
            scores.extend(batch_scores.cpu().tolist())
        return scores


class RerankerPool:
    """Thread-safe pool of Reranker instances across multiple GPUs.

    Each GPU gets one Reranker. When a request needs reranking, it acquires
    a reranker from the pool (blocking if all are busy), uses it, then returns
    it. This allows N GPUs to serve N concurrent rerank requests in parallel.
    """

    def __init__(self, model_path: str, gpu_ids: list[int], use_fp16: bool = True):
        self.gpu_ids = gpu_ids
        self._pool = queue.Queue()
        logger.info(f"Initializing RerankerPool with {len(gpu_ids)} GPUs: {gpu_ids}")
        for gid in gpu_ids:
            device = f"cuda:{gid}" if torch.cuda.is_available() else "cpu"
            logger.info(f"  Loading reranker on {device}...")
            reranker = Reranker(model_path=model_path, param_device=device, use_fp16=use_fp16)
            self._pool.put(reranker)
        logger.info(f"RerankerPool ready: {len(gpu_ids)} rerankers available.")

    def acquire(self) -> Reranker:
        """Block until a reranker is available, then return it."""
        return self._pool.get()

    def release(self, reranker: Reranker):
        """Return a reranker back to the pool."""
        self._pool.put(reranker)

    @property
    def size(self) -> int:
        return len(self.gpu_ids)


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool, initial_k: int = None, use_reranker: bool = None):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False, use_reranker: bool = None):
        return self._search(query, num, return_score, use_reranker=use_reranker)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False, initial_k: int = None, use_reranker: bool = None):
        return self._batch_search(query_list, num, return_score, initial_k=initial_k, use_reranker=use_reranker)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        
        # Always use a thread pool for parallel BM25 searching, even without reranker
        self.search_executor = ThreadPoolExecutor(max_workers=32)

        self.use_reranker = config.use_reranker
        if self.use_reranker:
            # Auto-detect all available GPUs; user can limit via CUDA_VISIBLE_DEVICES
            num_gpus = max(torch.cuda.device_count(), 1)
            gpu_ids = list(range(num_gpus))
            self.reranker_pool = RerankerPool(
                model_path=config.reranker_model_path,
                gpu_ids=gpu_ids,
                use_fp16=True,
            )
            self._thread_pool = ThreadPoolExecutor(max_workers=num_gpus)
            self.initial_k = config.initial_retrieval_k

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            results = []
            for hit in hits:
                raw_json = json.loads(self.searcher.doc(hit.docid).raw())
                
                # Directly extract title from meta and contents from root
                title = raw_json.get("meta", {}).get("title", "")
                content = raw_json.get("contents", "")

                results.append({
                    "title": title,
                    "contents": content
                })
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _rerank_single_query(self, query: str, num: int, initial_k: int):
        """BM25 retrieve + rerank for a single query. Thread-safe via RerankerPool."""
        candidates, _ = self._search(query, initial_k, True)
        doc_texts = [d.get("contents", "") for d in candidates]

        reranker = self.reranker_pool.acquire()
        try:
            rerank_scores = reranker.rerank(query, doc_texts)
        finally:
            self.reranker_pool.release(reranker)

        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)[:num]
        return [r[0] for r in ranked], [r[1] for r in ranked]

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False, initial_k: int = None, use_reranker: bool = None):
        if num is None:
            num = self.topk

        # Only allow reranking if the server is configured with reranker support
        # We can dynamically DISABLE it, but we cannot dynamically ENABLE it if it wasn't loaded.
        do_rerank = self.use_reranker
        if use_reranker is not None:
             if use_reranker and not self.use_reranker:
                  logger.warning("Reranking requested (use_reranker=True) but server was not started with reranker support. Ignoring.")
             elif not use_reranker:
                  do_rerank = False

        if do_rerank:
            k = initial_k if initial_k is not None else self.initial_k
            futures = [self._thread_pool.submit(self._rerank_single_query, q, num, k) for q in query_list]
            pairs = [f.result() for f in futures]
            results, scores = [list(x) for x in zip(*pairs)] if pairs else ([], [])
        else:
            # Parallelize BM25 search
            futures = [self.search_executor.submit(self._search, q, num, True) for q in query_list]
            pairs = [f.result() for f in futures]
            results, scores = [list(x) for x in zip(*pairs)] if pairs else ([], [])
            scores = [[float(s) for s in score_list] for score_list in scores]

        return (results, scores) if return_score else results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu and torch.cuda.is_available():
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            try:
                # Use specific GPU if possible, otherwise falls back to all GPUs if complex
                # but for single GPU selection:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                warnings.warn(f"Failed to move FAISS index to GPU 0: {e}. Falling back to all GPUs if possible.", stacklevel=2)
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        elif config.faiss_gpu:
            warnings.warn("FAISS GPU requested but CUDA not available. Using CPU index.", stacklevel=2)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False, initial_k: int = None):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 5,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        use_reranker: bool = False,
        reranker_model_path: str = "./model",
        reranker_topk: int = 5,
        initial_retrieval_k: int = 50,
        max_chars: int = 10000,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.use_reranker = use_reranker
        self.reranker_model_path = reranker_model_path
        self.reranker_topk = reranker_topk
        self.initial_retrieval_k = initial_retrieval_k
        self.max_chars = max_chars


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False
    max_chars: Optional[int] = None
    initial_k: Optional[int] = None
    use_reranker: Optional[bool] = None


def truncate_results(results, scores, max_chars, return_scores, queries):
    # Compute fixed overhead: all query texts + empty document lists skeleton
    skeleton = [{"query": queries[i], "documents": []} for i in range(len(results))]
    base_overhead = len(json.dumps(skeleton, indent=2, ensure_ascii=False))

    candidates = []

    for q_idx, docs in enumerate(results):
        for r_idx, doc in enumerate(docs):
            score = scores[q_idx][r_idx] if return_scores and scores else 0.0
            # Compute full serialized size of the document entry (including title, contents, JSON structure)
            if return_scores:
                entry = {"document": doc, "score": score}
            else:
                entry = doc
            entry_len = len(json.dumps(entry, indent=2, ensure_ascii=False))
            candidates.append({
                "q_idx": q_idx,
                "r_idx": r_idx,
                "doc": doc,
                "score": score,
                "len": entry_len,
            })

    # Sort by rank asc, then q_idx asc
    candidates.sort(key=lambda x: (x["r_idx"], x["q_idx"]))

    selected_indices = {q: [] for q in range(len(results))}
    current_chars = base_overhead
    truncated = False

    for cand in candidates:
        if current_chars + cand["len"] > max_chars:
            truncated = True
            break
        selected_indices[cand["q_idx"]].append(cand)
        current_chars += cand["len"]

    new_results = []
    new_scores = [] if return_scores else None

    for q in range(len(results)):
        q_cands = selected_indices[q]
        q_cands.sort(key=lambda x: x["r_idx"])

        docs_for_q = [c["doc"] for c in q_cands]
        new_results.append(docs_for_q)

        if return_scores:
            scores_for_q = [c["score"] for c in q_cands]
            new_scores.append(scores_for_q)

    return new_results, new_scores, truncated


app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }

    Output format:
    {
        "result": [
            {
                "query": "query text",
                "documents": [
                    # If return_scores=True: {"document": doc, "score": score}
                    # If return_scores=False: doc
                ]
            },
            ...
        ]
    }
    """
    topk = request.topk or config.retrieval_topk
    max_chars = request.max_chars if request.max_chars is not None else config.max_chars
    initial_k = request.initial_k if request.initial_k is not None else config.initial_retrieval_k
    use_reranker = request.use_reranker if request.use_reranker is not None else None

    # Perform batch retrieval
    if request.return_scores:
        results, scores = retriever.batch_search(query_list=request.queries, num=topk, return_score=True, initial_k=initial_k, use_reranker=use_reranker)
    else:
        results = retriever.batch_search(query_list=request.queries, num=topk, return_score=False, initial_k=initial_k, use_reranker=use_reranker)
        scores = None

    if max_chars > 0:
        results, scores, truncated = truncate_results(results, scores, max_chars, request.return_scores, request.queries)
    else:
        truncated = False

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        query_text = request.queries[i]
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            if scores is not None and len(scores) > i:
                 current_scores = scores[i]
            else:
                 current_scores = []
            
            for doc, score in zip(single_result, current_scores, strict=False):
                combined.append({"document": doc, "score": score})
            resp.append({"query": query_text, "documents": combined})
        else:
            resp.append({"query": query_text, "documents": single_result})
    
    ret_dict = {"result": resp}
    if truncated:
        msg = f"System Note: Retrieval results truncated at {max_chars} characters."
        ret_dict["truncation_warning"] = msg
        
    return ret_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--use_reranker", action="store_true", help="Enable reranker.")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3", help="Path to reranker model.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
        use_reranker=args.use_reranker,
        reranker_model_path=args.reranker_model,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
