import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples')))
import pytest
import numpy as np
from FINAL_CODE import (
    read_file, tokenize_text, create_tfidf_model, get_top_keywords,
    extract_keywords_from_files, calculate_cosine_similarity,
    group_similar_documents, extractive_summary,
    process_documents, save_results_to_file
)
import tempfile
import os

def test_read_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        tmp.write("Sample text content.")
        tmp_path = tmp.name
    content = read_file(tmp_path)
    os.remove(tmp_path)
    assert content == "Sample text content."

def test_tokenize_text():
    tokens = tokenize_text("This is a test.")
    assert isinstance(tokens, list)
    assert "test" in tokens

def test_create_tfidf_model():
    docs = ["This is a test.", "Another document."]
    tfidf_matrix, feature_names = create_tfidf_model(docs)
    assert tfidf_matrix.shape[0] == 2
    assert isinstance(feature_names, list)

def test_get_top_keywords():
    docs = ["cat sat mat", "dog bark park"]
    tfidf_matrix, feature_names = create_tfidf_model(docs)
    top_keywords = get_top_keywords(tfidf_matrix, feature_names, 2)
    assert isinstance(top_keywords, list)
    assert all(isinstance(tup, tuple) for tup in top_keywords)

def test_extract_keywords_from_files():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f1:
        f1.write("dog cat fish")
        fname1 = f1.name
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f2:
        f2.write("apple orange banana")
        fname2 = f2.name

    keywords = extract_keywords_from_files([fname1, fname2], 2)
    os.remove(fname1)
    os.remove(fname2)
    assert len(keywords) == 2
    assert isinstance(keywords[0], list)

def test_calculate_cosine_similarity():
    docs = ["dog cat", "cat fish", "apple orange"]
    tfidf_matrix, _ = create_tfidf_model(docs)
    similarity = calculate_cosine_similarity(tfidf_matrix)
    assert similarity.shape == (3, 3)
    assert np.allclose(np.diag(similarity), 1)

def test_group_similar_documents():
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    groups = group_similar_documents(sim, threshold=0.85)
    assert isinstance(groups, list)
    assert any(isinstance(g, list) for g in groups)

def test_extractive_summary():
    text = "Sentence one. Sentence two. Sentence three."
    summary = extractive_summary(text, 2)
    assert isinstance(summary, str)
    assert summary.count('.') <= 2

def test_process_documents():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f1:
        f1.write("wind turbine power energy generation.")
        fname1 = f1.name
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f2:
        f2.write("solar energy renewable source.")
        fname2 = f2.name

    results = process_documents([fname1, fname2], 2, 0.1, 1)
    os.remove(fname1)
    os.remove(fname2)
    assert isinstance(results, list)
    assert "keywords" in results[0]
    assert "summary" in results[0]

def test_save_results_to_file():
    results = [{"filename": "file1.txt", "keywords": ["wind"], "summary": "wind turbine"}]
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        filename = tmp.name
    save_results_to_file(results, filename)
    with open(filename, "r") as f:
        content = f.read()
    os.remove(filename)
    assert "wind" in content
