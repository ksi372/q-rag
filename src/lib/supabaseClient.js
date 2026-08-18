import { createClient } from '@supabase/supabase-js';
import { createLocalClient } from './localBackend.js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

const hasSupabaseConfig = Boolean(supabaseUrl && supabaseAnonKey);

/**
 * Real Supabase when configured, otherwise a localStorage-backed stand-in.
 *
 * The project this app shipped against is gone, so without the fallback every
 * route redirects to login and nothing is reachable. See localBackend.js for
 * the supported API surface, and supabase/migrations/003 for the chat schema
 * to run when a new project is created.
 */
export const supabase = hasSupabaseConfig
  ? createClient(supabaseUrl, supabaseAnonKey)
  : createLocalClient();

export const isLocalBackend = !hasSupabaseConfig;
