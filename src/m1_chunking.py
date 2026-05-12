"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    # 1. Split text into sentences:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast
    embeddings = model.encode(sentences)

    # 3. Compare consecutive sentences:
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(a, b): 
        n = norm(a) * norm(b)
        return dot(a, b) / n if n != 0 else 0

    # 4. Group sentences:
    chunks = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(text=" ".join(current_group), metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}))
            current_group = []
        current_group.append(sentences[i])
        
    # Don't forget last group
    if current_group:
        chunks.append(Chunk(text=" ".join(current_group), metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}))

    # 5. Return chunks
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents_list = []
    children_list = []
    
    # 1. Split text into parents:
    paragraphs = text.split("\n\n")
    current_parent_text = ""
    parent_index = 0
    
    for p in paragraphs:
        if len(current_parent_text) + len(p) > parent_size and current_parent_text:
            pid = f"parent_{parent_index}"
            parents_list.append(Chunk(text=current_parent_text.strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
            parent_index += 1
            current_parent_text = ""
        current_parent_text += p + "\n\n"
        
    if current_parent_text.strip():
        pid = f"parent_{parent_index}"
        parents_list.append(Chunk(text=current_parent_text.strip(), metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))

    # 2. Split each parent into children:
    for p_chunk in parents_list:
        pid = p_chunk.metadata["parent_id"]
        p_text = p_chunk.text
        start = 0
        while start < len(p_text):
            child_text = p_text[start:start+child_size]
            children_list.append(Chunk(text=child_text, metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
            start += child_size

    # 3. Return (parents_list, children_list)
    return parents_list, children_list


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    # 1. Split by markdown headers:
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    
    # 2. Pair headers with their content:
    chunks = []
    current_header = ""
    current_content = ""
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header, "strategy": "structure"}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
            
    # Don't forget last section
    if current_content.strip() or current_header.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={**metadata, "section": current_header, "strategy": "structure"}
        ))
        
    # 3. Return chunks
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {"basic": {"chunks": 0, "total_len": 0, "min": float('inf'), "max": 0},
               "semantic": {"chunks": 0, "total_len": 0, "min": float('inf'), "max": 0},
               "hierarchical": {"parents": 0, "children": 0, "parent_len": 0, "child_len": 0, "min": float('inf'), "max": 0},
               "structure": {"chunks": 0, "total_len": 0, "min": float('inf'), "max": 0}}
               
    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})
        
        # basic
        basic_chunks = chunk_basic(text, metadata=meta)
        results["basic"]["chunks"] += len(basic_chunks)
        for c in basic_chunks:
            l = len(c.text)
            results["basic"]["total_len"] += l
            results["basic"]["min"] = min(results["basic"]["min"], l)
            results["basic"]["max"] = max(results["basic"]["max"], l)
            
        # semantic
        sem_chunks = chunk_semantic(text, metadata=meta)
        results["semantic"]["chunks"] += len(sem_chunks)
        for c in sem_chunks:
            l = len(c.text)
            results["semantic"]["total_len"] += l
            results["semantic"]["min"] = min(results["semantic"]["min"], l)
            results["semantic"]["max"] = max(results["semantic"]["max"], l)
            
        # hierarchical
        parents, children = chunk_hierarchical(text, metadata=meta)
        results["hierarchical"]["parents"] += len(parents)
        results["hierarchical"]["children"] += len(children)
        for c in parents:
            results["hierarchical"]["parent_len"] += len(c.text)
        for c in children:
            l = len(c.text)
            results["hierarchical"]["child_len"] += l
            results["hierarchical"]["min"] = min(results["hierarchical"]["min"], l)
            results["hierarchical"]["max"] = max(results["hierarchical"]["max"], l)
            
        # structure
        struct_chunks = chunk_structure_aware(text, metadata=meta)
        results["structure"]["chunks"] += len(struct_chunks)
        for c in struct_chunks:
            l = len(c.text)
            results["structure"]["total_len"] += l
            results["structure"]["min"] = min(results["structure"]["min"], l)
            results["structure"]["max"] = max(results["structure"]["max"], l)
            
    # Calculate averages
    for k in ["basic", "semantic", "structure"]:
        if results[k]["chunks"] > 0:
            results[k]["avg_len"] = results[k]["total_len"] / results[k]["chunks"]
        else:
            results[k]["avg_len"] = 0
            results[k]["min"] = 0
            
    if results["hierarchical"]["children"] > 0:
        results["hierarchical"]["avg_child_len"] = results["hierarchical"]["child_len"] / results["hierarchical"]["children"]
    else:
        results["hierarchical"]["avg_child_len"] = 0
        results["hierarchical"]["min"] = 0

    print(f"{'Strategy':<15} | {'Chunks':<10} | {'Avg Len':<10} | {'Min':<5} | {'Max':<5}")
    print(f"{'basic':<15} | {results['basic']['chunks']:<10} | {results['basic']['avg_len']:<10.1f} | {results['basic']['min']:<5} | {results['basic']['max']:<5}")
    print(f"{'semantic':<15} | {results['semantic']['chunks']:<10} | {results['semantic']['avg_len']:<10.1f} | {results['semantic']['min']:<5} | {results['semantic']['max']:<5}")
    print(f"{'hierarchical':<15} | {str(results['hierarchical']['parents'])+'p/'+str(results['hierarchical']['children'])+'c':<10} | {results['hierarchical']['avg_child_len']:<10.1f} | {results['hierarchical']['min']:<5} | {results['hierarchical']['max']:<5}")
    print(f"{'structure':<15} | {results['structure']['chunks']:<10} | {results['structure']['avg_len']:<10.1f} | {results['structure']['min']:<5} | {results['structure']['max']:<5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
