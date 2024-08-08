# ruff: noqa: PLR0913
# ruff: noqa: PLR0912
from abc import ABC, abstractmethod
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PointStruct,
    UpdateResult,
    VectorParams,
)

from emb_predict.utils import log


# Define an abstract class VectorDB
class VectorDB(ABC):
    def __init__(self, collections: list[dict[str, str | int]]):
        self.collections = collections
        pass

    @abstractmethod
    def add(self, collection_name: str, item_list: list[str], batch_size: int) -> None:
        pass

    @abstractmethod
    def get(
        self,
        collection_name: str,
        search_input: str | None = None,
        search_field: Optional[list[str]] = None,
        exact_match=True,
        limit: int = 5,
    ) -> list[Any]:
        pass

    @abstractmethod
    def get_similar(self, collection_name: str, vector: str) -> list[tuple[str, float]]:
        pass


# https://qdrant.tech/documentation/quick-start
# More config: https://qdrant.tech/documentation/concepts/collections/#create-a-collection
class QdrantDB(VectorDB):
    def __init__(
        self,
        url: str,
        collections: list[dict[str, str | int]],
        recreate: bool = False,
        api_key: str | None = None,
    ):
        super().__init__(collections)

        if "localhost" in url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            from urllib.parse import urlparse

            parsed_url = urlparse(url)
            scheme = parsed_url.scheme
            host = parsed_url.hostname
            port = (
                parsed_url.port
                if parsed_url.port is not None
                else "443"
                if scheme == "https"
                else "80"
            )
            self.client = QdrantClient(host=host, port=port, api_key=api_key)

        # make sure we can connect
        try:
            db_collections = self.client.get_collections()
            log.info(
                f"Connected to VectorDB at {url}. Found {len(db_collections.collections)} collections"
            )
        except Exception as e:
            log.error(f"Could not connect to VectorDB at {url} with error: {e}")
            raise e

        if len(collections) < 1:
            raise ValueError(
                'Provide at least 1 collection, e.g. [{"name": "my_collec", "size": 512}]'
            )
        else:
            for collection in collections:
                try:
                    if not recreate:
                        points = self.client.get_collection(
                            collection["name"]
                        ).points_count
                        log.info(f"Collection {collection['name']} has {points} points")
                    else:
                        raise Exception("Recreate")
                except Exception:
                    if recreate:
                        log.info(f"⚠️ Recreating {collection['name']} collection")
                    else:
                        log.info(
                            f"⚠️ Collection {collection['name']} not found: Recreating the collection"
                        )
                    self.client.recreate_collection(
                        collection_name=collection["name"],
                        vectors_config=VectorParams(
                            size=collection["size"], distance=Distance.COSINE
                        ),
                    )
                    self.client.create_payload_index(
                        collection["name"],
                        "id",
                        {
                            "type": "text",
                            "tokenizer": "word",
                            "min_token_len": 2,
                            "max_token_len": 30,
                            "lowercase": True,
                        },
                    )
                    self.client.create_payload_index(
                        collection["name"],
                        "name",
                        {
                            "type": "text",
                            "tokenizer": "word",
                            "min_token_len": 2,
                            "max_token_len": 30,
                            "lowercase": True,
                        },
                    )
            collection_names = [d["name"] for d in collections if "name" in d]
            log.info(
                "Vector DB initialized with collections: " + ", ".join(collection_names)
            )

    def add(
        self, collection_name: str, item_list: list[str], batch_size: int = 1000
    ) -> UpdateResult:
        """Add a list of entities and their vector to the database"""
        for i in range(0, len(item_list), batch_size):
            item_batch = item_list[i : i + batch_size]

            points_count = self.client.get_collection(collection_name).points_count
            points_list = [
                PointStruct(
                    id=points_count + i + 1,
                    vector=item["vector"],
                    payload=item["payload"],
                )
                for i, item in enumerate(item_batch)
            ]

            # PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
            operation_info = self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points_list,
            )
        return operation_info

    def get(
        self,
        collection_name: str,
        search_input: str | None = None,
        search_fields: Optional[list[str]] = None,
        exact_match=True,
        limit: int = 5,
    ) -> list[Any]:
        """Get the vector for a specific entity ID"""
        if search_fields is not None:
            search_filter = Filter(
                should=[
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=search_input)
                        if exact_match is True
                        else MatchText(text=search_input),
                    )
                    for field in search_fields
                ]
            )
        search_result = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter if search_input else None,
            with_vectors=True,
            with_payload=True,
            limit=limit,
        )
        return search_result[0]

    def get_similar(
        self,
        collection_name: str,
        vector: str,
        search_input: str | None = None,
        limit: int = 10,
    ) -> list[Any] | None:
        """Search for vectors similar to a given vector"""
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=Filter(
                must=[FieldCondition(key="id", match=MatchText(value=search_input))]
            )
            if search_input
            else None,
            # search_params=SearchParams(hnsw_ef=128, exact=False),
            with_vectors=True,
            with_payload=True,
            limit=limit,
        )
        # "strategy": "best_score"
        return search_result


def init_vectordb(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collections: Optional[list[dict[str, str]]] = None,
    recreate: bool = False,
):
    return QdrantDB(
        url=url,
        api_key=api_key,
        collections=collections,
        recreate=recreate,
    )
