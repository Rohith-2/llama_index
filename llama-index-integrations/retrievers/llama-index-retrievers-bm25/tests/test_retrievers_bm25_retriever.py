from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.bm25.base import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
import os


def test_class():
    names_of_base_classes = [b.__name__ for b in BM25Retriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes


def test_scores():
    documents = [
        Document(text="Large Language Model llm",extra_info={'filter':'1'}),
        Document(text="LlamaIndex is a data framework for your LLM application",extra_info={'filter':'2'}),
        Document(text="How to use LlamaIndex",extra_info={'filter':'1'}),
    ]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
    result_nodes = retriever.retrieve("llamaindex llm")
    assert len(result_nodes) == 2
    for node in result_nodes:
        assert node.score is not None
        assert node.score > 0.0

def test_scores_with_filters():
    documents = [
        Document(text="Large Language Model llm",extra_info={'filter':'1'}),
        Document(text="LlamaIndex is a data framework for your LLM application",extra_info={'filter':'2'}),
        Document(text="How to use LlamaIndex",extra_info={'filter':'1'}),
    ]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
    retriever.set_filter({'filter':'2'})
    result_nodes = retriever.retrieve("llamaindex llm")
    assert len(result_nodes) == 1
    for node in result_nodes:
        assert node.score is not None
        assert node.score > 0.0

def test_saving_loading(tmp_path):
    documents = [
        Document(text="Large Language Model llm", extra_info={'filter': '1'}),
        Document(text="LlamaIndex is a data framework for your LLM application", extra_info={'filter': '2'}),
        Document(text="How to use LlamaIndex", extra_info={'filter': '1'}),
    ]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
    temp_file = os.path.join(tmp_path,"test_bm25_retriever")

    retriever.persist(temp_file)
    retriever = BM25Retriever.from_persist_dir(temp_file)
    result_nodes = retriever.retrieve("llamaindex llm")
    assert len(result_nodes) == 2
    for node in result_nodes:
        assert node.score is not None
        assert node.score > 0.0
