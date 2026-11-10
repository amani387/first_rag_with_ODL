"""
Token-aware chunker for OpenDataLoader PDF JSON output.
Preserves bounding box (bbox), page numbers, element IDs, and heading metadata.
"""
from typing import List, Dict, Any, Optional
import uuid

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


def count_tokens(text: str) -> int:
    """Counts tokens using tiktoken cl100k_base or fallback estimation."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Fallback heuristic: ~0.75 words per token (1.33 tokens per word)
    return max(1, int(len(text.split()) * 1.33))


def extract_element_text(element: Dict[str, Any]) -> str:
    """Extracts plain text content from an OpenDataLoader JSON element."""
    content = element.get("content") or element.get("text") or element.get("description")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        return " ".join(str(c) for c in content if c).strip()
    
    # Check if element has child kids
    kids = element.get("kids")
    if isinstance(kids, list):
        kid_texts = [extract_element_text(k) for k in kids if isinstance(k, dict)]
        return " ".join(t for t in kid_texts if t).strip()
    
    return ""


def extract_element_bbox(element: Dict[str, Any]) -> List[float]:
    """Extracts bounding box [x0, y0, x1, y1] from an element."""
    bbox = element.get("bounding box") or element.get("bbox") or []
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            return [float(x) for x in bbox]
        except (ValueError, TypeError):
            return []
    return []


def extract_element_page(element: Dict[str, Any]) -> int:
    """Extracts 1-indexed page number from an element."""
    page = element.get("page number") or element.get("page") or element.get("page_number") or 1
    try:
        return int(page)
    except (ValueError, TypeError):
        return 1


def chunk_elements(
    elements: List[Dict[str, Any]],
    upload_id: str,
    source_file: str,
    max_tokens: int = 700,
    overlap_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    Groups ODL elements into token-bounded chunks preserving bounding boxes and element metadata.

    Args:
        elements: List of raw element dictionaries from ODL JSON.
        upload_id: Unique upload identifier.
        source_file: Name of the source PDF.
        max_tokens: Maximum number of tokens per chunk.
        overlap_tokens: Overlap tokens from the end of the previous chunk.

    Returns:
        List of chunk dictionaries suitable for embedding and indexing.
    """
    # Preprocess elements into normalized units
    items = []
    current_heading = ""

    for el in elements:
        if not isinstance(el, dict):
            continue
        text = extract_element_text(el)
        if not text:
            continue

        el_type = str(el.get("type", "paragraph")).lower()
        if "heading" in el_type or "title" in el_type:
            current_heading = text

        el_id = el.get("id")
        if el_id is None:
            el_id = len(items) + 1

        items.append({
            "id": el_id,
            "text": text,
            "tokens": count_tokens(text),
            "page": extract_element_page(el),
            "bbox": extract_element_bbox(el),
            "type": el_type,
            "heading": current_heading
        })

    if not items:
        return []

    chunks = []
    current_elements = []
    current_tokens = 0
    chunk_index = 0

    def make_chunk(el_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        nonlocal chunk_index
        chunk_index += 1
        chunk_id = f"{upload_id}_chunk_{chunk_index}"
        combined_text = "\n\n".join(e["text"] for e in el_list)
        pages = sorted(list(set(e["page"] for e in el_list)))
        primary_page = pages[0] if pages else 1

        # Primary bbox: first non-empty bbox
        bboxes = [e["bbox"] for e in el_list if e["bbox"]]
        primary_bbox = bboxes[0] if bboxes else [0.0, 0.0, 0.0, 0.0]

        element_ids = [e["id"] for e in el_list]
        heading = next((e["heading"] for e in reversed(el_list) if e["heading"]), "")

        return {
            "chunk_id": chunk_id,
            "upload_id": upload_id,
            "source_file": source_file,
            "text": combined_text,
            "page": primary_page,
            "pages": pages,
            "bbox": primary_bbox,
            "bboxes": bboxes,
            "element_ids": element_ids,
            "token_count": count_tokens(combined_text),
            "extra_metadata": {
                "heading": heading,
                "element_types": list(set(e["type"] for e in el_list))
            }
        }

    i = 0
    while i < len(items):
        item = items[i]

        # Case 1: Single element exceeds max_tokens on its own
        if item["tokens"] > max_tokens:
            if current_elements:
                chunks.append(make_chunk(current_elements))
                current_elements = []
                current_tokens = 0

            # Sub-chunk the oversized element text
            words = item["text"].split()
            sub_words = []
            sub_tokens = 0
            for w in words:
                w_tok = count_tokens(w)
                if sub_tokens + w_tok > max_tokens and sub_words:
                    sub_text = " ".join(sub_words)
                    chunks.append({
                        "chunk_id": f"{upload_id}_chunk_{chunk_index + 1}",
                        "upload_id": upload_id,
                        "source_file": source_file,
                        "text": sub_text,
                        "page": item["page"],
                        "pages": [item["page"]],
                        "bbox": item["bbox"] or [0.0, 0.0, 0.0, 0.0],
                        "bboxes": [item["bbox"]] if item["bbox"] else [],
                        "element_ids": [item["id"]],
                        "token_count": count_tokens(sub_text),
                        "extra_metadata": {
                            "heading": item["heading"],
                            "element_types": [item["type"]]
                        }
                    })
                    chunk_index += 1
                    sub_words = []
                    sub_tokens = 0
                sub_words.append(w)
                sub_tokens += w_tok

            if sub_words:
                sub_text = " ".join(sub_words)
                chunks.append({
                    "chunk_id": f"{upload_id}_chunk_{chunk_index + 1}",
                    "upload_id": upload_id,
                    "source_file": source_file,
                    "text": sub_text,
                    "page": item["page"],
                    "pages": [item["page"]],
                    "bbox": item["bbox"] or [0.0, 0.0, 0.0, 0.0],
                    "bboxes": [item["bbox"]] if item["bbox"] else [],
                    "element_ids": [item["id"]],
                    "token_count": count_tokens(sub_text),
                    "extra_metadata": {
                        "heading": item["heading"],
                        "element_types": [item["type"]]
                    }
                })
                chunk_index += 1

            i += 1
            continue

        # Case 2: Adding this item exceeds max_tokens
        if current_tokens + item["tokens"] > max_tokens and current_elements:
            chunks.append(make_chunk(current_elements))

            # Compute overlap from the end of current_elements
            overlap_elements = []
            overlap_accum = 0
            for prev_el in reversed(current_elements):
                if overlap_accum + prev_el["tokens"] <= overlap_tokens:
                    overlap_elements.insert(0, prev_el)
                    overlap_accum += prev_el["tokens"]
                else:
                    break

            current_elements = overlap_elements
            current_tokens = sum(e["tokens"] for e in current_elements)

        current_elements.append(item)
        current_tokens += item["tokens"]
        i += 1

    if current_elements:
        chunks.append(make_chunk(current_elements))

    return chunks
