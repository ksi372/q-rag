-- =============================================
-- Enable pgvector extension for embeddings
-- =============================================
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================
-- Add embedding column to chunks table
-- Using 768 dimensions (Gemini embedding-001 output size)
-- =============================================
ALTER TABLE public.chunks 
ADD COLUMN IF NOT EXISTS embedding vector(768);

-- =============================================
-- Create index for fast similarity search
-- Using IVFFlat for good balance of speed/accuracy
-- =============================================
CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
ON public.chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- =============================================
-- Create function for similarity search
-- =============================================
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(768),
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
    1 - (c.embedding <=> query_embedding) as similarity
  FROM public.chunks c
  JOIN public.documents d ON c.document_id = d.id
  WHERE 
    c.embedding IS NOT NULL
    AND (match_user_id IS NULL OR d.user_id = match_user_id)
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

