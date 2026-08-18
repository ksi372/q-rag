-- =============================================
-- RAG Knowledge Base Tables
-- Run this in Supabase SQL Editor
-- =============================================

-- 1) Documents table: stores uploaded document metadata
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    relative_path TEXT,  -- preserve folder upload paths (nullable)
    storage_key TEXT NOT NULL,  -- storage path only, NOT public URL
    storage_provider TEXT NOT NULL DEFAULT 'supabase',  -- future: 's3'
    file_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2) Chunks table: stores document chunks for RAG retrieval
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,  -- order within document
    content TEXT NOT NULL,
    token_length INT NOT NULL,  -- approx tokens
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(document_id, chunk_index);

-- =============================================
-- Row Level Security (RLS)
-- =============================================

-- Enable RLS on documents table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Documents policies: users can only access their own documents
CREATE POLICY "Users can view their own documents"
    ON documents FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own documents"
    ON documents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own documents"
    ON documents FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own documents"
    ON documents FOR DELETE
    USING (auth.uid() = user_id);

-- Enable RLS on chunks table
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

-- Chunks policies: enforce access via parent document ownership
CREATE POLICY "Users can view chunks of their own documents"
    ON chunks FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = chunks.document_id
            AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert chunks for their own documents"
    ON chunks FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = chunks.document_id
            AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update chunks of their own documents"
    ON chunks FOR UPDATE
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

CREATE POLICY "Users can delete chunks of their own documents"
    ON chunks FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = chunks.document_id
            AND documents.user_id = auth.uid()
        )
    );

-- =============================================
-- Storage bucket for RAG documents
-- Run this separately in Storage settings or via SQL
-- =============================================
-- Note: Create a NEW bucket called 'documents' in Supabase Storage
-- with the following RLS policies:
--
-- SELECT: (bucket_id = 'documents') AND (auth.uid()::text = (storage.foldername(name))[1])
-- INSERT: (bucket_id = 'documents') AND (auth.uid()::text = (storage.foldername(name))[1])
-- UPDATE: (bucket_id = 'documents') AND (auth.uid()::text = (storage.foldername(name))[1])
-- DELETE: (bucket_id = 'documents') AND (auth.uid()::text = (storage.foldername(name))[1])

