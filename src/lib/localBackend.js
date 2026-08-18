/**
 * Browser-side stand-in for the Supabase client.
 *
 * The original Supabase project no longer exists, so auth and the chat tables
 * (threads / messages / folders / message_attachments) had nowhere to live and
 * every page bounced straight back to the login screen. This implements the
 * subset of the supabase-js API the pages actually call, backed by
 * localStorage, so the app is usable standalone.
 *
 * Supported surface:
 *   auth.signUp / signInWithPassword / signOut / getSession /
 *   onAuthStateChange / updateUser / resetPasswordForEmail
 *   from(table).select(cols).eq().in().order().single()
 *   from(table).insert(rows).select().single()
 *   from(table).update(patch).eq().select().single()
 *   from(table).delete().eq()
 *   storage.from(bucket).upload / getPublicUrl / remove
 *
 * Not supported: signInWithOAuth (no provider to redirect to) — it returns a
 * clear error instead of failing silently.
 *
 * RAG documents are NOT stored here. Those go through the FastAPI backend,
 * which owns its own SQLite store; this only covers auth and chat history.
 *
 * Set VITE_SUPABASE_URL + VITE_SUPABASE_ANON_KEY to use real Supabase instead
 * (see supabase/migrations for the schema, including 003 for these tables).
 */

const PREFIX = 'synergy.local';
const USERS_KEY = `${PREFIX}.users`;
const SESSION_KEY = `${PREFIX}.session`;
const TABLES = ['threads', 'messages', 'folders', 'message_attachments'];

// Blob URLs can't survive a reload, so attachment bodies are per-session.
// Metadata rows persist; a stale url just won't preview.
const blobUrls = new Map();

const nowIso = () => new Date().toISOString();

const uuid = () =>
  crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

function readJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

const tableKey = (table) => `${PREFIX}.table.${table}`;
const readTable = (table) => readJson(tableKey(table), []);
const writeTable = (table, rows) => writeJson(tableKey(table), rows);

/** Passwords are never kept in the clear, even in a local stand-in. */
async function hashPassword(password) {
  const bytes = new TextEncoder().encode(`${PREFIX}:${password}`);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

const ok = (data) => ({ data, error: null });
const fail = (message) => ({ data: null, error: { message } });

/** Narrow a row to the columns a .select('a, b') asked for. */
function project(row, columns) {
  if (!columns || columns.trim() === '*') return { ...row };
  const wanted = columns.split(',').map((c) => c.trim()).filter(Boolean);
  const out = {};
  wanted.forEach((c) => {
    out[c] = row[c];
  });
  return out;
}

// =============================================
// Query builder
// =============================================

/**
 * Thenable builder: awaiting it runs the query, matching supabase-js where
 * `await supabase.from(t).select()` resolves without an explicit .execute().
 */
class LocalQuery {
  constructor(table) {
    this.table = table;
    this.action = 'select';
    this.columns = '*';
    this.filters = [];
    this.orderBy = null;
    this.payload = null;
    this.wantSingle = false;
    this.selectAfterWrite = false;
  }

  select(columns = '*') {
    if (this.action === 'select') {
      this.columns = columns;
    } else {
      // .insert(...).select(cols) — return the written rows.
      this.selectAfterWrite = true;
      this.columns = columns;
    }
    return this;
  }

  insert(payload) {
    this.action = 'insert';
    this.payload = payload;
    return this;
  }

  update(payload) {
    this.action = 'update';
    this.payload = payload;
    return this;
  }

  delete() {
    this.action = 'delete';
    return this;
  }

  eq(column, value) {
    this.filters.push({ op: 'eq', column, value });
    return this;
  }

  in(column, values) {
    this.filters.push({ op: 'in', column, value: values ?? [] });
    return this;
  }

  order(column, options = {}) {
    this.orderBy = { column, ascending: options.ascending !== false };
    return this;
  }

  single() {
    this.wantSingle = true;
    return this;
  }

  maybeSingle() {
    this.wantSingle = true;
    return this;
  }

  matches(row) {
    return this.filters.every(({ op, column, value }) => {
      if (op === 'eq') return row[column] === value;
      if (op === 'in') return value.includes(row[column]);
      return true;
    });
  }

  run() {
    const rows = readTable(this.table);

    if (this.action === 'select') {
      let result = rows.filter((r) => this.matches(r));
      if (this.orderBy) {
        const { column, ascending } = this.orderBy;
        result = [...result].sort((a, b) => {
          const x = a[column];
          const y = b[column];
          if (x === y) return 0;
          const cmp = x > y ? 1 : -1;
          return ascending ? cmp : -cmp;
        });
      }
      const projected = result.map((r) => project(r, this.columns));
      if (this.wantSingle) {
        return projected.length ? ok(projected[0]) : fail('No rows found');
      }
      return ok(projected);
    }

    if (this.action === 'insert') {
      const incoming = Array.isArray(this.payload) ? this.payload : [this.payload];
      const created = incoming.map((row) => ({
        id: row.id ?? uuid(),
        created_at: row.created_at ?? nowIso(),
        ...row,
      }));
      writeTable(this.table, [...rows, ...created]);
      const projected = created.map((r) => project(r, this.columns));
      if (this.wantSingle) return ok(projected[0]);
      return ok(projected);
    }

    if (this.action === 'update') {
      if (!this.filters.length) return fail('Refusing to update without a filter');
      const updated = [];
      const next = rows.map((row) => {
        if (!this.matches(row)) return row;
        const merged = { ...row, ...this.payload };
        updated.push(merged);
        return merged;
      });
      writeTable(this.table, next);
      const projected = updated.map((r) => project(r, this.columns));
      if (this.wantSingle) {
        return projected.length ? ok(projected[0]) : fail('No rows updated');
      }
      return ok(projected);
    }

    if (this.action === 'delete') {
      if (!this.filters.length) return fail('Refusing to delete without a filter');
      const removed = rows.filter((r) => this.matches(r));
      writeTable(this.table, rows.filter((r) => !this.matches(r)));

      // No ON DELETE CASCADE here, so mirror the schema's cascades by hand.
      if (this.table === 'threads') {
        const threadIds = new Set(removed.map((t) => t.id));
        const messages = readTable('messages');
        const orphaned = new Set(
          messages.filter((m) => threadIds.has(m.thread_id)).map((m) => m.id),
        );
        writeTable('messages', messages.filter((m) => !threadIds.has(m.thread_id)));
        writeTable(
          'message_attachments',
          readTable('message_attachments').filter((a) => !orphaned.has(a.message_id)),
        );
      }
      if (this.table === 'messages') {
        const messageIds = new Set(removed.map((m) => m.id));
        writeTable(
          'message_attachments',
          readTable('message_attachments').filter((a) => !messageIds.has(a.message_id)),
        );
      }

      return ok(removed.map((r) => project(r, this.columns)));
    }

    return fail(`Unsupported action ${this.action}`);
  }

  then(resolve, reject) {
    return Promise.resolve()
      .then(() => this.run())
      .then(resolve, reject);
  }
}

// =============================================
// Auth
// =============================================

function makeSession(user) {
  return {
    // The backend accepts any bearer token in local mode, but send something
    // token-shaped so the Authorization header is well formed.
    access_token: `local.${user.id}`,
    token_type: 'bearer',
    expires_at: Math.floor(Date.now() / 1000) + 60 * 60 * 24 * 30,
    user: { id: user.id, email: user.email },
  };
}

class LocalAuth {
  constructor() {
    this.listeners = new Set();
  }

  emit(event, session) {
    this.listeners.forEach((cb) => {
      try {
        cb(event, session);
      } catch (e) {
        console.error('[localBackend] auth listener failed', e);
      }
    });
  }

  async getSession() {
    return ok({ session: readJson(SESSION_KEY, null) });
  }

  async getUser() {
    const session = readJson(SESSION_KEY, null);
    return ok({ user: session?.user ?? null });
  }

  onAuthStateChange(callback) {
    this.listeners.add(callback);
    // Match supabase-js: fire once with the current state.
    Promise.resolve().then(() =>
      callback('INITIAL_SESSION', readJson(SESSION_KEY, null)),
    );
    return {
      data: {
        subscription: {
          unsubscribe: () => this.listeners.delete(callback),
        },
      },
    };
  }

  async signUp({ email, password }) {
    const normalized = (email || '').trim().toLowerCase();
    if (!normalized || !password) return fail('Email and password are required');

    const users = readJson(USERS_KEY, []);
    if (users.some((u) => u.email === normalized)) {
      return fail('An account with this email already exists');
    }

    const user = {
      id: uuid(),
      email: normalized,
      password_hash: await hashPassword(password),
      created_at: nowIso(),
    };
    writeJson(USERS_KEY, [...users, user]);

    const session = makeSession(user);
    writeJson(SESSION_KEY, session);
    this.emit('SIGNED_IN', session);
    return ok({ user: session.user, session });
  }

  async signInWithPassword({ email, password }) {
    const normalized = (email || '').trim().toLowerCase();
    const users = readJson(USERS_KEY, []);
    const user = users.find((u) => u.email === normalized);
    if (!user) return fail('No account found for that email');

    if (user.password_hash !== (await hashPassword(password))) {
      return fail('Incorrect password');
    }

    const session = makeSession(user);
    writeJson(SESSION_KEY, session);
    this.emit('SIGNED_IN', session);
    return ok({ user: session.user, session });
  }

  async signInWithOAuth() {
    return fail(
      'Google sign-in needs a Supabase project. Use email and password in local mode.',
    );
  }

  async signOut() {
    localStorage.removeItem(SESSION_KEY);
    this.emit('SIGNED_OUT', null);
    return { error: null };
  }

  async updateUser({ password }) {
    const session = readJson(SESSION_KEY, null);
    if (!session) return fail('Not signed in');
    if (!password) return fail('A new password is required');

    const users = readJson(USERS_KEY, []);
    const hash = await hashPassword(password);
    writeJson(
      USERS_KEY,
      users.map((u) => (u.id === session.user.id ? { ...u, password_hash: hash } : u)),
    );
    return ok({ user: session.user });
  }

  async resetPasswordForEmail(email) {
    const users = readJson(USERS_KEY, []);
    if (!users.some((u) => u.email === (email || '').trim().toLowerCase())) {
      return fail('No account found for that email');
    }
    // No mail transport locally; the reset page works off the live session.
    return fail(
      'Password reset email needs a Supabase project. In local mode, sign in and change your password from the dashboard.',
    );
  }
}

// =============================================
// Storage
// =============================================

class LocalBucket {
  constructor(bucket) {
    this.bucket = bucket;
  }

  async upload(path, file) {
    blobUrls.set(`${this.bucket}/${path}`, URL.createObjectURL(file));
    return ok({ path });
  }

  getPublicUrl(path) {
    return { data: { publicUrl: blobUrls.get(`${this.bucket}/${path}`) ?? '' } };
  }

  /** Blob URLs need no signing; return the same handle so callers work unchanged. */
  async createSignedUrl(path) {
    const url = blobUrls.get(`${this.bucket}/${path}`);
    if (!url) return fail('File is no longer available in this session');
    return ok({ signedUrl: url });
  }

  async remove(paths) {
    (paths || []).forEach((path) => {
      const key = `${this.bucket}/${path}`;
      const url = blobUrls.get(key);
      if (url) URL.revokeObjectURL(url);
      blobUrls.delete(key);
    });
    return ok(paths ?? []);
  }
}

class LocalStorage {
  from(bucket) {
    return new LocalBucket(bucket);
  }
}

// =============================================
// Client
// =============================================

export function createLocalClient() {
  TABLES.forEach((table) => {
    if (localStorage.getItem(tableKey(table)) === null) writeTable(table, []);
  });

  console.info(
    '[localBackend] No Supabase config found — running on the local browser store. ' +
      'Auth and chat history live in localStorage; RAG documents live in the backend.',
  );

  return {
    isLocal: true,
    auth: new LocalAuth(),
    storage: new LocalStorage(),
    from: (table) => new LocalQuery(table),
  };
}
