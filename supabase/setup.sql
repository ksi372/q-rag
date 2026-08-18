-- =============================================
-- Synergy — complete schema, paste-and-run
--
-- Run this once in the Supabase SQL editor for a new project. It is the whole
-- current schema and is idempotent, so it supersedes running migrations
-- 001 / 002 / 003 individually (those are kept as history).
--
-- IMPORTANT — vector width is 384, not 768.
-- The app embeds locally with BAAI/bge-small-en-v1.5 (384-d) because the Yotta
-- gateway exposes no /embeddings route and the Gemini key has no credits.
-- Migration 002 hardcoded vector(768) for Gemini; using that width here would
-- make every insert fail. If you later switch EMBEDDING_PROVIDER=gemini, change
-- 384 -> 768 in the three places below and re-embed the corpus.
-- =============================================

CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================
-- RAG tables
-- =============================================

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    relative_path TEXT,
    storage_key TEXT NOT NULL,
    storage_provider TEXT NOT NULL DEFAULT 'supabase',
    file_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    token_length INT NOT NULL,
    embedding vector(384),   -- <-- 384: local bge-small. Gemini would be 768.
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(document_id, chunk_index);

-- IVFFlat needs lists <= row count to build sensibly. At this corpus size an
-- exact scan is fast anyway; create the index once there is real data.
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding
--   ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================
-- Similarity search RPC
-- =============================================

DROP FUNCTION IF EXISTS match_chunks(vector, int, uuid);
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(384),   -- <-- keep in step with chunks.embedding
  match_count int DEFAULT 5,
  match_user_id uuid DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  document_id uuid,
  chunk_index int,
  content text,
  token_length int,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    c.id,
    c.document_id,
    c.chunk_index,
    c.content,
    c.token_length,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM public.chunks c
  JOIN public.documents d ON c.document_id = d.id
  WHERE
    c.embedding IS NOT NULL
    AND (match_user_id IS NULL OR d.user_id = match_user_id)
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- =============================================
-- Chat tables
-- =============================================

CREATE TABLE IF NOT EXISTS folders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    folder_id UUID REFERENCES folders(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS message_attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_size BIGINT,
    file_type TEXT,
    file_url TEXT,
    file_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_folders_user_id ON folders(user_id);
CREATE INDEX IF NOT EXISTS idx_threads_user_id ON threads(user_id);
CREATE INDEX IF NOT EXISTS idx_threads_created_at ON threads(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_threads_folder_id ON threads(folder_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id, created_at);
CREATE INDEX IF NOT EXISTS idx_message_attachments_message_id ON message_attachments(message_id);

-- =============================================
-- Row Level Security
--
-- The browser talks to the chat tables directly with the anon key, so RLS is
-- the only boundary between users. The FastAPI backend uses the service_role
-- key, which bypasses RLS by design — that is why it needs the service key and
-- not the anon key.
-- =============================================

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE folders ENABLE ROW LEVEL SECURITY;
ALTER TABLE threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_attachments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users manage their own documents" ON documents;
CREATE POLICY "Users manage their own documents"
    ON documents FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users manage chunks of their own documents" ON chunks;
CREATE POLICY "Users manage chunks of their own documents"
    ON chunks FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = chunks.document_id
            AND documents.user_id = auth.uid()
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = chunks.document_id
            AND documents.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users manage their own folders" ON folders;
CREATE POLICY "Users manage their own folders"
    ON folders FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users manage their own threads" ON threads;
CREATE POLICY "Users manage their own threads"
    ON threads FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users manage their own messages" ON messages;
CREATE POLICY "Users manage their own messages"
    ON messages FOR ALL
    USING (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM threads
            WHERE threads.id = messages.thread_id
            AND threads.user_id = auth.uid()
        )
    )
    WITH CHECK (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM threads
            WHERE threads.id = messages.thread_id
            AND threads.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users manage their own attachments" ON message_attachments;
CREATE POLICY "Users manage their own attachments"
    ON message_attachments FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- =============================================
-- Storage buckets
--   documents -> RAG source files (written by the backend)
--   files     -> chat attachments (written by the browser)
-- Both private; access is per-user by first path segment.
-- =============================================

INSERT INTO storage.buckets (id, name, public)
VALUES ('documents', 'documents', false)
ON CONFLICT (id) DO NOTHING;

INSERT INTO storage.buckets (id, name, public)
VALUES ('files', 'files', false)
ON CONFLICT (id) DO NOTHING;

DROP POLICY IF EXISTS "Users manage their own document objects" ON storage.objects;
CREATE POLICY "Users manage their own document objects"
    ON storage.objects FOR ALL
    USING (
        bucket_id = 'documents'
        AND auth.uid()::text = (storage.foldername(name))[1]
    )
    WITH CHECK (
        bucket_id = 'documents'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

DROP POLICY IF EXISTS "Users manage their own chat files" ON storage.objects;
CREATE POLICY "Users manage their own chat files"
    ON storage.objects FOR ALL
    USING (
        bucket_id = 'files'
        AND auth.uid()::text = (storage.foldername(name))[1]
    )
    WITH CHECK (
        bucket_id = 'files'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );
