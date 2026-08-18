"""
Cut over from the local SQLite store to Supabase.

Run this AFTER pasting supabase/setup.sql into the Supabase SQL editor. DDL cannot
go through the REST API, so creating the schema is a manual step; everything after
it is automated here.

What it does:
  1. Verifies every table and the match_chunks RPC exist, and that the embedding
     column width matches the active embedding provider.
  2. Copies documents + chunks (with embeddings) out of server/local_data/synergy.db
     into Supabase, attributing them to a real auth user.
  3. Flips SUPABASE_URL on in server/config.env and .env.local.

Usage:
    .venv/Scripts/python.exe switch_to_supabase.py --check
    .venv/Scripts/python.exe switch_to_supabase.py --owner you@example.com

--check only inspects; nothing is written. --owner performs the migration. The
owner must already exist as a Supabase auth user (sign up through the app once
with Supabase enabled, or create one in the dashboard), because documents.user_id
is a foreign key onto auth.users.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "config.env"
FRONTEND_ENV = HERE.parent / ".env.local"
DB_PATH = HERE / "local_data" / "synergy.db"

load_dotenv(CONFIG_PATH)

import providers  # noqa: E402  (after load_dotenv so it sees the env)

TABLES = ["documents", "chunks", "folders", "threads", "messages", "message_attachments"]


def read_config_value(name: str) -> str:
    """Read a key from config.env even when it is commented out."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(rf"^#?\s*{name}=(.*)$", text, re.MULTILINE)
    return (match.group(1).strip() if match else "")


def resolve_target():
    url = read_config_value("SUPABASE_URL")
    service_key = read_config_value("SUPABASE_SERVICE_KEY")
    if not url:
        sys.exit(
            "SUPABASE_URL is not set in config.env (not even commented).\n"
            "Add it, then re-run."
        )
    if not service_key:
        sys.exit(
            "SUPABASE_SERVICE_KEY is missing. The anon key will not work: the backend\n"
            "writes on the user's behalf and RLS blocks anon (auth.uid() is null)."
        )
    return url.rstrip("/"), service_key


def client(url: str, key: str) -> httpx.Client:
    return httpx.Client(
        base_url=url,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    )


def check_schema(http: httpx.Client) -> bool:
    print("Checking schema...")
    ok = True

    for table in TABLES:
        response = http.get(f"/rest/v1/{table}", params={"select": "*", "limit": 1})
        state = "ok" if response.status_code == 200 else f"MISSING ({response.status_code})"
        if response.status_code != 200:
            ok = False
        print(f"  table {table:<22} {state}")

    dim = providers.embedding_dim()
    probe = http.post(
        "/rest/v1/rpc/match_chunks",
        content=json.dumps({"query_embedding": [0.0] * dim, "match_count": 1}),
    )
    if probe.status_code == 200:
        print(f"  rpc   match_chunks({dim})     ok")
    else:
        ok = False
        body = probe.text[:200]
        print(f"  rpc   match_chunks({dim})     FAILED ({probe.status_code}) {body}")
        if "expected" in body or "does not exist" in body:
            print(
                f"\n  The RPC signature does not accept a {dim}-dim vector. setup.sql "
                f"declares vector({dim});\n  migration 002's vector(768) is Gemini-only."
            )

    if not ok:
        print("\nSchema incomplete. Paste supabase/setup.sql into the SQL editor first.")
    return ok


def find_user(http: httpx.Client, email: str) -> str:
    """Look up an auth user by email via the admin API."""
    response = http.get("/auth/v1/admin/users", params={"page": 1, "per_page": 200})
    if response.status_code != 200:
        sys.exit(f"Could not list auth users ({response.status_code}): {response.text[:200]}")

    users = response.json().get("users", response.json() if isinstance(response.json(), list) else [])
    for user in users:
        if (user.get("email") or "").lower() == email.lower():
            return user["id"]

    known = ", ".join(u.get("email", "?") for u in users) or "none"
    sys.exit(
        f"No Supabase auth user with email {email}.\n"
        f"Existing users: {known}\n"
        "Create one in the dashboard (Authentication -> Users), or sign up through "
        "the app once Supabase is enabled."
    )


def migrate(http: httpx.Client, owner_id: str) -> None:
    if not DB_PATH.exists():
        print(f"No local database at {DB_PATH}; nothing to migrate.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    documents = conn.execute("SELECT * FROM documents").fetchall()
    if not documents:
        print("Local store has no documents; nothing to migrate.")
        conn.close()
        return

    dim = providers.embedding_dim()
    print(f"\nMigrating {len(documents)} document(s) to Supabase as owner {owner_id}...")

    migrated_docs = 0
    migrated_chunks = 0
    skipped_chunks = 0

    for document in documents:
        payload = {
            "filename": document["filename"],
            "relative_path": document["relative_path"],
            "storage_key": document["storage_key"],
            "storage_provider": "supabase",
            "file_type": document["file_type"],
            "user_id": owner_id,
        }
        response = http.post(
            "/rest/v1/documents",
            content=json.dumps(payload),
            headers={"Prefer": "return=representation"},
        )
        if response.status_code not in (200, 201):
            print(f"  ! {document['filename']}: {response.status_code} {response.text[:160]}")
            continue

        new_id = response.json()[0]["id"]
        migrated_docs += 1

        chunk_rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document["id"],),
        ).fetchall()

        batch = []
        for chunk in chunk_rows:
            embedding = None
            if chunk["embedding"]:
                try:
                    embedding = json.loads(chunk["embedding"])
                except (ValueError, TypeError):
                    embedding = None
            # A vector of the wrong width would be rejected by Postgres anyway;
            # drop it and let /api/documents/reembed refill it later.
            if embedding is not None and len(embedding) != dim:
                embedding = None
                skipped_chunks += 1

            batch.append(
                {
                    "document_id": new_id,
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "token_length": chunk["token_length"],
                    "embedding": embedding,
                }
            )

        # Insert in slices; a single 65-chunk payload with 384-dim vectors is large.
        for start in range(0, len(batch), 20):
            slice_ = batch[start : start + 20]
            chunk_response = http.post("/rest/v1/chunks", content=json.dumps(slice_))
            if chunk_response.status_code not in (200, 201):
                print(f"  ! chunks {start}-{start + len(slice_)}: "
                      f"{chunk_response.status_code} {chunk_response.text[:160]}")
            else:
                migrated_chunks += len(slice_)

        print(f"  + {document['filename']}: {len(chunk_rows)} chunks")

    conn.close()
    print(f"\nMigrated {migrated_docs} document(s), {migrated_chunks} chunk(s).")
    if skipped_chunks:
        print(f"{skipped_chunks} embedding(s) had the wrong width and were left null; "
              "call POST /api/documents/reembed to regenerate them.")


def enable_env() -> None:
    """Uncomment SUPABASE_URL in both env files."""
    url = read_config_value("SUPABASE_URL")

    config = CONFIG_PATH.read_text(encoding="utf-8")
    config = re.sub(r"^SUPABASE_URL=\s*$", f"SUPABASE_URL={url}", config, flags=re.MULTILINE)
    CONFIG_PATH.write_text(config, encoding="utf-8")
    print(f"config.env      -> SUPABASE_URL={url}")

    if FRONTEND_ENV.exists():
        frontend = FRONTEND_ENV.read_text(encoding="utf-8")
        anon = ""
        match = re.search(r"^#\s*VITE_SUPABASE_ANON_KEY=(.*)$", frontend, re.MULTILINE)
        if match:
            anon = match.group(1).strip()
        frontend = re.sub(
            r"^VITE_SUPABASE_URL=\s*$", f"VITE_SUPABASE_URL={url}", frontend, flags=re.MULTILINE
        )
        if anon:
            frontend = re.sub(
                r"^VITE_SUPABASE_ANON_KEY=\s*$",
                f"VITE_SUPABASE_ANON_KEY={anon}",
                frontend,
                flags=re.MULTILINE,
            )
        FRONTEND_ENV.write_text(frontend, encoding="utf-8")
        print(f".env.local      -> VITE_SUPABASE_URL={url}"
              + (" (+ anon key)" if anon else " (anon key not found, set it manually)"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify schema only, write nothing")
    parser.add_argument("--owner", help="email of the Supabase auth user to own migrated docs")
    parser.add_argument(
        "--skip-data", action="store_true", help="flip env without copying local data"
    )
    args = parser.parse_args()

    url, service_key = resolve_target()
    print(f"Target: {url}\n")

    with client(url, service_key) as http:
        if not check_schema(http):
            sys.exit(1)

        if args.check:
            print("\nSchema looks good. Re-run with --owner <email> to migrate and switch.")
            return

        if not args.skip_data:
            if not args.owner:
                sys.exit("\nPass --owner <email> (or --skip-data to flip env only).")
            owner_id = find_user(http, args.owner)
            migrate(http, owner_id)

    print()
    enable_env()
    print("\nDone. Restart uvicorn and Vite, then check http://127.0.0.1:8081/")


if __name__ == "__main__":
    main()
