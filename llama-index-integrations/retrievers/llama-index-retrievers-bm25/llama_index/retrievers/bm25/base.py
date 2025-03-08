import logging

from typing import Any, Callable, Dict, List, Optional, cast

import bm25_fusion as bm25f
from nltk.stem import PorterStemmer

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import (
    TextNode,
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    MetadataMode,
)
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_ARGS = {"similarity_top_k": "similarity_top_k", "_verbose": "verbose"}

DEFAULT_PERSIST_FILENAME = "retriever.json"


class BM25Retriever(BaseRetriever):
    r"""A BM25 retriever that uses the BM25 algorithm to retrieve nodes.

    Args:
        nodes (List[BaseNode], optional):
            The nodes to index. If not provided, an existing BM25 object must be passed.
        stemmer (Stemmer.Stemmer, optional):
            The stemmer to use. Defaults to an english stemmer.
        language (str, optional):
            The language to use for stopword removal. Defaults to "en".
        existing_bm25 (bm25s.BM25, optional):
            An existing BM25 object to use. If not provided, nodes must be passed.
        similarity_top_k (int, optional):
            The number of results to return. Defaults to DEFAULT_SIMILARITY_TOP_K.
        callback_manager (CallbackManager, optional):
            The callback manager to use. Defaults to None.
        objects (List[IndexNode], optional):
            The objects to retrieve. Defaults to None.
        object_map (dict, optional):
            A map of object IDs to nodes. Defaults to None.
        token_pattern (str, optional):
            The token pattern to use. Defaults to (?u)\\b\\w\\w+\\b.
        skip_stemming (bool, optional):
            Whether to skip stemming. Defaults to False.
        verbose (bool, optional):
            Whether to show progress. Defaults to False.
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        stemmer: Optional[PorterStemmer] = None,
        language: str = "en",
        existing_bm25: Optional[bm25f.BM25] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        skip_stemming: bool = False,
        token_pattern: str = r"(?u)\b\w\w+\b",
        variant:Optional[str] = 'bm25+',
        do_keyword:Optional[str] = 'False'

    ) -> None:
        self.stemmer = stemmer or PorterStemmer()
        self.similarity_top_k = similarity_top_k
        self.token_pattern = token_pattern
        self.skip_stemming = skip_stemming
        self.do_keyword = do_keyword
        self.metadata_filter = {}

        if existing_bm25 is not None:
            self.bm25 = existing_bm25
            #self.corpus = existing_bm25.corpus
        else:
            if nodes is None:
                raise ValueError("Please pass nodes or an existing BM25 object.")

            self.corpus = [node_to_metadata_dict(node) for node in nodes]
            for i, node in enumerate(nodes):
                node.metadata["_id_no"] = i
                
            metadata = [node.metadata for node in nodes]

            self.bm25 = bm25f.BM25([node.get_content()
                                    for node in nodes],metadata=metadata, do_stem=skip_stemming,
                                    stemmer=self.stemmer if not skip_stemming\
                                    else None,variant=variant,do_keyword=self.do_keyword)

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        stemmer: Optional[PorterStemmer] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        skip_stemming: bool = False,
        variant:Optional[str]="bm25+",
        do_keyword:Optional[str] = True,

        # deprecated
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "BM25Retriever":
        if tokenizer is not None:
            logger.warning(
                "The tokenizer parameter is deprecated and will be removed in a future release. "
                "Use a stemmer from PyStemmer instead."
            )

        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        return cls(
            nodes=nodes,
            stemmer=stemmer,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            skip_stemming=skip_stemming,
            variant = variant,
            do_keyword = do_keyword
        )

    def get_persist_args(self) -> Dict[str, Any]:
        """Get Persist Args Dict to Save."""
        return {
            DEFAULT_PERSIST_ARGS[key]: getattr(self, key)
            for key in DEFAULT_PERSIST_ARGS
            if hasattr(self, key)
        }

    def persist(self, path: str) -> None:
        """Persist the retriever to a directory."""
        if 'h5' not in path:
            path+= '.h5'
        self.bm25.save_hdf5(path)

    @classmethod
    def from_persist_dir(cls, path: str) -> "BM25Retriever":
        """Load the retriever from a directory."""
        if 'h5' not in path:
            path +='.h5'
        bm25 = bm25f.BM25.load_hdf5(path)
        return cls(existing_bm25=bm25)
    
    def set_filter(self,metadata_filter: Optional[dict] = None,):
        self.metadata_filter = metadata_filter

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        results = self.bm25.query(
            query, metadata_filter=self.metadata_filter,top_k=self.similarity_top_k, do_keyword=self.do_keyword
        )
        print(query,results)

        nodes: List[NodeWithScore] = [NodeWithScore(node=TextNode(text=node_dict['text'], metadata=node_dict),\
                                                     score=float(node_dict['score']))\
                                      for node_dict in results]

        return nodes
