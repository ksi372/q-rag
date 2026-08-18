-- =============================================
-- Chat tables: folders / threads / messages / message_attachments
--
-- These were created by hand in the original (now deleted) project and were
-- never captured in a migration, so the schema was undocumented. Reconstructed
-- from the columns src/pages/Workspace.jsx actually selects, inserts, and
-- filters on. Run this after 001 and 002 on a new project.
-- =============================================

-- 1) Folders: optional grouping for threads
CREATE TABLE IF NOT EXISTS folders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2) Threads: one conversation. folder_id is nullable ("All" / ungrouped),
--    and ON DELETE SET NULL so deleting a folder keeps its threads.
CREATE TABLE IF NOT EXISTS threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    folder_id UUID REFERENCES folders(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3) Messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4) Message attachments: files pinned to a message, stored in the
--    'files' storage bucket (distinct from the RAG 'documents' bucket)
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

-- Indexes matching the app's query patterns
CREATE INDEX IF NOT EXISTS idx_folders_user_id ON folders(user_id);
CREATE INDEX IF NOT EXISTS idx_threads_user_id ON threads(user_id);
CREATE INDEX IF NOT EXISTS idx_threads_created_at ON threads(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_threads_folder_id ON threads(folder_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id, created_at);
CREATE INDEX IF NOT EXISTS idx_message_attachments_message_id ON message_attachments(message_id);

-- =============================================
-- Row Level Security
-- The client talks to these tables directly with the anon key, so RLS is the
-- only thing separating users. Every policy keys off auth.uid().
-- =============================================

ALTER TABLE folders ENABLE ROW LEVEL SECURITY;
ALTER TABLE threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_attachments ENABLE ROW LEVEL SECURITY;

-- Folders
DROP POLICY IF EXISTS "Users manage their own folders" ON folders;
CREATE POLICY "Users manage their own folders"
    ON folders FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Threads
DROP POLICY IF EXISTS "Users manage their own threads" ON threads;
CREATE POLICY "Users manage their own threads"
    ON threads FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Messages: ownership is on the row, and the parent thread must also be theirs
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

-- Attachments
DROP POLICY IF EXISTS "Users manage their own attachments" ON message_attachments;
CREATE POLICY "Users manage their own attachments"
    ON message_attachments FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- =============================================
-- Storage bucket for chat attachments
-- =============================================
-- Create a bucket called 'files' (separate from the RAG 'documents' bucket),
-- with the same per-user folder policies:
--
-- SELECT/INSERT/UPDATE/DELETE:
--   (bucket_id = 'files') AND (auth.uid()::text = (storage.foldername(name))[1])
