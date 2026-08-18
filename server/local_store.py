"""
SQLite-backed stand-in for the Supabase client.

The original Supabase project is gone (its hostname no longer resolves) and a
replacement is being provisioned separately, so the app had no data layer at
all. Rather than rewrite ~30 call sites, this module reimplements the exact
slice of the supabase-py fluent API that app.py actually uses:

    .table(t).select(cols, count=...).eq/.in_/.ilike/.is_/.order(...).execute()
    .table(t).insert(rows).execute()
    .table(t).update(patch).eq(...).execute()
    .table(t).delete().eq(...).execute()
    .rpc("match_chunks", {...}).execute()
    .storage.from_(bucket).upload(path, bytes, opts) / .remove([paths])
    .auth.get_user(token)

`match_chunks` is the pgvector RPC; here it's brute-force cosine over numpy,
which is exact (no IVFFlat approximation) and fine at this corpus size.

Storage writes to server/local_data/storage/. Schema mirrors
supabase/migrations/001 + 002, minus RLS — ownership is enforced in the
queries themselves, same as the RPC's user filter.

Switch back to real Supabase by setting SUPABASE_URL + a key in config.env;
app.py prefers the real client whenever those are present.
"""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DATA_DIR = Path(os.getenv("LOCAL_DATA_DIR") or (Path(__file__).parent / "local_data"))
DB_PATH = DATA_DIR / "synergy.db"
STORAGE_DIR = DATA_DIR / "storage"

# Stable identity for single-user local mode. Shaped like a UUID because the
# rest of the code (and the future Supabase schema) treats user_id as one.
LOCAL_USER_ID = os.getenv("LOCAL_USER_ID") or "00000000-0000-4000-8000-000000000001"
LOCAL_USER_EMAIL = os.getenv("LOCAL_USER_EMAIL") or "local@synergy.dev"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                relative_path TEXT,
                storage_key TEXT NOT NULL,
                storage_provider TEXT NOT NULL DEFAULT 'local',
                file_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_length INTEGER NOT NULL,
                embedding TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(document_id, chunk_index);
            """
        )


# =============================================
# Result / query objects
# =============================================

class LocalResponse:
    """Mirrors the shape app.py reads off a supabase execute(): .data / .count"""

    def __init__(self, data: Any, count: Optional[int] = None):
        self.data = data
        self.count = count


class _Filter:
    __slots__ = ("op", "column", "value")

    def __init__(self, op: str, column: str, value: Any):
        self.op = op
        self.column = column
        self.value = value


class TableQuery:
    """Chainable query builder over one table."""

    # Columns that hold JSON-encoded vectors and must be decoded on read.
    _VECTOR_COLUMNS = {"embedding"}

    def __init__(self, table: str):
        if table not in ("documents", "chunks"):
            raise ValueError(f"local_store: unknown table {table!r}")
        self.table = table
        self.action = "select"
        self.columns = "*"
        self.want_count = False
        self.filters: List[_Filter] = []
        self.order_by: Optional[tuple] = None
        self.limit_n: Optional[int] = None
        self.payload: Any = None

    # --- verbs ---

    def select(self, columns: str = "*", count: Optional[str] = None) -> "TableQuery":
        self.action = "select"
        self.columns = columns
        self.want_count = count is not None
        return self

    def insert(self, payload: Any) -> "TableQuery":
        self.action = "insert"
        self.payload = payload
        return self

    def update(self, payload: Dict[str, Any]) -> "TableQuery":
        self.action = "update"
        self.payload = payload
        return self

    def delete(self) -> "TableQuery":
        self.action = "delete"
        return self

    # --- filters / modifiers ---

    def eq(self, column: str, value: Any) -> "TableQuery":
        self.filters.append(_Filter("eq", column, value))
        return self

    def in_(self, column: str, values: List[Any]) -> "TableQuery":
        self.filters.append(_Filter("in", column, list(values)))
        return self

    def ilike(self, column: str, pattern: str) -> "TableQuery":
        self.filters.append(_Filter("ilike", column, pattern))
        return self

    def is_(self, column: str, value: Any) -> "TableQuery":
        self.filters.append(_Filter("is", column, value))
        return self

    def order(self, column: str, desc: bool = False) -> "TableQuery":
        self.order_by = (column, desc)
        return self

    def limit(self, n: int) -> "TableQuery":
        self.limit_n = n
        return self

    # --- SQL assembly ---

    def _where(self) -> tuple:
        clauses: List[str] = []
        params: List[Any] = []
        for f in self.filters:
            if f.op == "eq":
                clauses.append(f"{f.column} = ?")
                params.append(f.value)
            elif f.op == "in":
                if not f.value:
                    # An empty IN () matches nothing; keep that semantics.
                    clauses.append("1 = 0")
                    continue
                clauses.append(f"{f.column} IN ({','.join('?' * len(f.value))})")
                params.extend(f.value)
            elif f.op == "ilike":
                # Postgres ilike -> SQLite LIKE (already case-insensitive for ASCII).
                clauses.append(f"{f.column} LIKE ?")
                params.append(f.value)
            elif f.op == "is":
                if str(f.value).lower() in ("null", "none") or f.value is None:
                    clauses.append(f"{f.column} IS NULL")
                else:
                    clauses.append(f"{f.column} = ?")
                    params.append(f.value)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _select_columns(self) -> List[str]:
        if self.columns.strip() == "*":
            return ["*"]
        return [c.strip() for c in self.columns.split(",") if c.strip()]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        out = dict(row)
        for column in TableQuery._VECTOR_COLUMNS:
            if column in out and isinstance(out[column], str):
                try:
                    out[column] = json.loads(out[column])
                except (ValueError, TypeError):
                    out[column] = None
        return out

    def execute(self) -> LocalResponse:
        with _lock, _connect() as conn:
            if self.action == "select":
                return self._do_select(conn)
            if self.action == "insert":
                return self._do_insert(conn)
            if self.action == "update":
                return self._do_update(conn)
            if self.action == "delete":
                return self._do_delete(conn)
        raise ValueError(f"local_store: unsupported action {self.action}")

    def _do_select(self, conn: sqlite3.Connection) -> LocalResponse:
        where, params = self._where()
        columns = self._select_columns()
        projection = "*" if columns == ["*"] else ", ".join(columns)

        sql = f"SELECT {projection} FROM {self.table}{where}"
        if self.order_by:
            column, desc = self.order_by
            sql += f" ORDER BY {column} {'DESC' if desc else 'ASC'}"
        if self.limit_n:
            sql += f" LIMIT {int(self.limit_n)}"

        rows = conn.execute(sql, params).fetchall()
        data = [self._row_to_dict(r) for r in rows]

        count = None
        if self.want_count:
            count_sql = f"SELECT COUNT(*) AS n FROM {self.table}{where}"
            count = conn.execute(count_sql, params).fetchone()["n"]
        return LocalResponse(data, count)

    def _do_insert(self, conn: sqlite3.Connection) -> LocalResponse:
        rows = self.payload if isinstance(self.payload, list) else [self.payload]
        inserted: List[Dict[str, Any]] = []

        for row in rows:
            record = dict(row)
            record.setdefault("id", str(uuid.uuid4()))
            record.setdefault("created_at", _now())
            if self.table == "documents":
                record.setdefault("storage_provider", "local")

            # Vectors arrive as python lists; store them as JSON text.
            for column in self._VECTOR_COLUMNS:
                if column in record and isinstance(record[column], (list, tuple, np.ndarray)):
                    record[column] = json.dumps([float(x) for x in record[column]])

            columns = list(record.keys())
            placeholders = ",".join("?" * len(columns))
            conn.execute(
                f"INSERT INTO {self.table} ({','.join(columns)}) VALUES ({placeholders})",
                [record[c] for c in columns],
            )
            inserted.append(dict(row, id=record["id"], created_at=record["created_at"]))

        return LocalResponse(inserted)

    def _do_update(self, conn: sqlite3.Connection) -> LocalResponse:
        patch = dict(self.payload or {})
        for column in self._VECTOR_COLUMNS:
            if column in patch and isinstance(patch[column], (list, tuple, np.ndarray)):
                patch[column] = json.dumps([float(x) for x in patch[column]])

        if not patch:
            return LocalResponse([])

        where, params = self._where()
        assignments = ", ".join(f"{c} = ?" for c in patch)
        conn.execute(
            f"UPDATE {self.table} SET {assignments}{where}",
            list(patch.values()) + params,
        )
        rows = conn.execute(f"SELECT * FROM {self.table}{where}", params).fetchall()
        return LocalResponse([self._row_to_dict(r) for r in rows])

    def _do_delete(self, conn: sqlite3.Connection) -> LocalResponse:
        where, params = self._where()
        rows = conn.execute(f"SELECT * FROM {self.table}{where}", params).fetchall()
        deleted = [self._row_to_dict(r) for r in rows]
        conn.execute(f"DELETE FROM {self.table}{where}", params)
        return LocalResponse(deleted)


# =============================================
# RPC: match_chunks (pgvector equivalent)
# =============================================

class RPCQuery:
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params or {}

    def execute(self) -> LocalResponse:
        if self.name != "match_chunks":
            raise ValueError(f"local_store: unsupported rpc {self.name!r}")
        return LocalResponse(self._match_chunks())

    def _match_chunks(self) -> List[Dict[str, Any]]:
        query_embedding = self.params.get("query_embedding")
        match_count = int(self.params.get("match_count") or 5)
        match_user_id = self.params.get("match_user_id")

        if not query_embedding:
            return []

        sql = (
            "SELECT c.id, c.document_id, c.chunk_index, c.content, "
            "       c.token_length, c.embedding "
            "FROM chunks c JOIN documents d ON c.document_id = d.id "
            "WHERE c.embedding IS NOT NULL"
        )
        params: List[Any] = []
        if match_user_id:
            sql += " AND d.user_id = ?"
            params.append(match_user_id)

        with _lock, _connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        query_vector = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            return []

        candidates: List[Dict[str, Any]] = []
        matrix: List[np.ndarray] = []
        for row in rows:
            try:
                vector = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
            except (ValueError, TypeError):
                continue
            # A corpus embedded under a different provider will have a different
            # width; skip rather than crash on the dot product.
            if vector.shape != query_vector.shape:
                continue
            matrix.append(vector)
            candidates.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "token_length": row["token_length"],
                }
            )

        if not candidates:
            return []

        stacked = np.vstack(matrix)
        norms = np.linalg.norm(stacked, axis=1)
        similarities = (stacked @ query_vector) / (norms * query_norm + 1e-9)

        order = np.argsort(-similarities)[:match_count]
        results = []
        for i in order:
            record = dict(candidates[i])
            record["similarity"] = float(similarities[i])
            results.append(record)
        return results


# =============================================
# Storage
# =============================================

class LocalBucket:
    def __init__(self, bucket: str):
        self.root = STORAGE_DIR / bucket

    def upload(self, path: str, file: bytes, file_options: Optional[dict] = None):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file)
        return {"path": path}

    def remove(self, paths: List[str]):
        removed = []
        for path in paths:
            target = self.root / path
            if target.exists():
                target.unlink()
                removed.append(path)
        return removed

    def download(self, path: str) -> bytes:
        return (self.root / path).read_bytes()


class LocalStorage:
    def from_(self, bucket: str) -> LocalBucket:
        return LocalBucket(bucket)


# =============================================
# Auth
# =============================================

class _LocalUser:
    def __init__(self):
        self.id = LOCAL_USER_ID
        self.email = LOCAL_USER_EMAIL


class _LocalUserResponse:
    def __init__(self):
        self.user = _LocalUser()


class LocalAuth:
    """
    Single-user local auth. Any token resolves to the same identity — there is
    no Supabase Auth to validate against, and the server is bound to localhost.
    """

    def get_user(self, token: str):
        return _LocalUserResponse()


# =============================================
# Client
# =============================================

class LocalStoreClient:
    """Drop-in for the subset of supabase.Client that app.py touches."""

    def __init__(self):
        init_db()
        self.storage = LocalStorage()
        self.auth = LocalAuth()

    def table(self, name: str) -> TableQuery:
        return TableQuery(name)

    def rpc(self, name: str, params: Dict[str, Any]) -> RPCQuery:
        return RPCQuery(name, params)


_client: Optional[LocalStoreClient] = None


def get_local_client() -> LocalStoreClient:
    global _client
    if _client is None:
        _client = LocalStoreClient()
        print(f"[Store] Local SQLite store at {DB_PATH}")
    return _client


def stats() -> dict:
    init_db()
    with _lock, _connect() as conn:
        documents = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        embedded = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()["n"]
    return {
        "backend": "local-sqlite",
        "path": str(DB_PATH),
        "documents": documents,
        "chunks": chunks,
        "chunks_embedded": embedded,
    }
