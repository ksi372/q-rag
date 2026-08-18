import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient.js';
import { marked } from 'marked';
import { extractFileText } from '../utils/fileExtractor.js';
import './Workspace.css';
// Imported after Workspace.css so the HUD theme wins on shared selectors.
import './comparison-hud.css';

const Workspace = () => {
  const navigate = useNavigate();
  const [stage, setStage] = useState('intro'); // intro -> chat
  const [sessionChecked, setSessionChecked] = useState(false);
  const [session, setSession] = useState(null);
  const [threads, setThreads] = useState([]);
  const [threadsLoading, setThreadsLoading] = useState(false);
  const [selectedThreadId, setSelectedThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [composerValue, setComposerValue] = useState('');
  const [creatingThread, setCreatingThread] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [threadMenuOpenId, setThreadMenuOpenId] = useState(null);
  const [editingThreadId, setEditingThreadId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [folders, setFolders] = useState([]);
  const [foldersLoading, setFoldersLoading] = useState(false);
  const [selectedFolderId, setSelectedFolderId] = useState('all');
  const [folderMenuOpenId, setFolderMenuOpenId] = useState(null);
  const [editingFolderId, setEditingFolderId] = useState(null);
  const [editingFolderName, setEditingFolderName] = useState('');
  const [draggedThreadId, setDraggedThreadId] = useState(null);
  const [dragOverFolderId, setDragOverFolderId] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [searchActive, setSearchActive] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchResults, setSearchResults] = useState({ folders: [], threads: [] });
  const feedRef = useRef(null);
  const bottomAnchorRef = useRef(null);
  const messagesCacheRef = useRef(new Map());
  const searchInputRef = useRef(null);
  const composerTextareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [messageAttachments, setMessageAttachments] = useState(new Map());
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const dropZoneRef = useRef(null);
  const [deleteConfirmThread, setDeleteConfirmThread] = useState(null); // thread object to confirm delete
  
  // RAG Documents state
  const [ragDocuments, setRagDocuments] = useState([]);
  const [ragDocsLoading, setRagDocsLoading] = useState(false);
  const [ragDocsExpanded, setRagDocsExpanded] = useState(false);
  const [uploadingToRag, setUploadingToRag] = useState(false);
  const [deleteConfirmDoc, setDeleteConfirmDoc] = useState(null);
  const ragFileInputRef = useRef(null);
  
  // RAG Comparison state
  const [ragComparison, setRagComparison] = useState(null);
  const [showRagComparison, setShowRagComparison] = useState(false);
  
  // Was hardcoded to :8001 while the backend actually serves :8081, so every
  // RAG call silently failed. Driven by env now, with the real default.
  const RAG_BACKEND_URL = import.meta.env.VITE_QUBO_SOLVER_URL || 'http://127.0.0.1:8081';

  const delay = useCallback((ms) => new Promise((resolve) => setTimeout(resolve, ms)), []);

  /**
   * Generation goes through our backend rather than straight to a model API.
   * The old path shipped the LLM key to the browser and hardcoded one vendor.
   */
  const generateWithRetries = useCallback(
    async (payload, attempt = 1) => {
      const response = await fetch(`${RAG_BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${session?.access_token ?? 'local'}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      let body = null;
      const contentType = response.headers.get('content-type') ?? '';
      if (contentType.includes('application/json')) {
        body = await response.json();
      } else {
        body = { detail: await response.text() };
      }

      if (response.ok) return body;

      const message = body?.detail ?? `Chat request failed (${response.status} ${response.statusText})`;

      const transientStatuses = [408, 429, 500, 502, 503, 504];
      if (attempt < 3 && transientStatuses.includes(response.status)) {
        await delay(2000 * attempt);
        return generateWithRetries(payload, attempt + 1);
      }

      throw new Error(message);
    },
    [RAG_BACKEND_URL, session, delay],
  );

  const renderMarkdown = useCallback((value) => {
    return marked.parse(value ?? '', { breaks: true });
  }, []);

  const bumpThreadToTop = useCallback((threadId, overrideThread) => {
    setThreads((prev) => {
      const existing = overrideThread ?? prev.find((thread) => thread.id === threadId);
      if (!existing) return prev;
      const rest = prev.filter((thread) => thread.id !== threadId);
      return [{ ...existing }, ...rest];
    });
  }, []);

  const adjustComposerHeight = useCallback(() => {
    const textarea = composerTextareaRef.current;
    if (!textarea) return;
    textarea.style.height = '0px';
    const next = Math.min(Math.max(textarea.scrollHeight, 54), 240);
    textarea.style.height = `${next}px`;
  }, []);

  const handleComposerInput = useCallback(
    (event) => {
      const { value } = event.target;
      setComposerValue(value);
      requestAnimationFrame(adjustComposerHeight);
    },
    [adjustComposerHeight],
  );

  // Session management
  useEffect(() => {
    let isMounted = true;

    const loadSession = async () => {
      const { data } = await supabase.auth.getSession();
      if (!isMounted) return;

      if (!data.session) {
        navigate('/login', { replace: true });
        return;
      }

      setSession(data.session);
      setSessionChecked(true);
    };

    loadSession();

    const { data: listener } = supabase.auth.onAuthStateChange((_event, currentSession) => {
      if (!currentSession) {
        navigate('/login', { replace: true });
      } else {
        setSession(currentSession);
      }
    });

    return () => {
      isMounted = false;
      listener.subscription.unsubscribe();
    };
  }, [navigate]);

  // Intro animation timer
  useEffect(() => {
    if (!sessionChecked) return;
    const timer = setTimeout(() => setStage('chat'), 5400);
    return () => clearTimeout(timer);
  }, [sessionChecked]);

  // Search focus
  useEffect(() => {
    if (searchActive && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [searchActive]);

  useEffect(() => {
    if (!searchActive && !searchQuery.trim()) {
      setSearchResults({ folders: [], threads: [] });
    }
  }, [searchActive, searchQuery]);

  // Fetch folders
  useEffect(() => {
    if (!session?.user?.id) return;

    const fetchFolders = async () => {
      setFoldersLoading(true);
      const { data, error } = await supabase
        .from('folders')
        .select('id, name, created_at')
        .eq('user_id', session.user.id)
        .order('created_at', { ascending: true });

      if (!error && data) setFolders(data);
      setFoldersLoading(false);
    };

    fetchFolders();
  }, [session]);

  // Fetch RAG documents
  const fetchRagDocuments = useCallback(async () => {
    if (!session?.access_token) {
      console.log('fetchRagDocuments: No access token');
      return;
    }
    
    setRagDocsLoading(true);
    try {
      console.log('Fetching RAG documents...');
      const response = await fetch(`${RAG_BACKEND_URL}/api/documents`, {
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      });
      
      const data = await response.json();
      console.log('RAG documents response:', response.status, data);
      
      if (response.ok) {
        setRagDocuments(data.documents || []);
      } else {
        console.error('Failed to fetch documents:', data);
      }
    } catch (error) {
      console.error('Error fetching RAG documents:', error);
    } finally {
      setRagDocsLoading(false);
    }
  }, [session?.access_token]);

  useEffect(() => {
    if (session?.access_token) {
      fetchRagDocuments();
    }
  }, [session?.access_token, fetchRagDocuments]);

  // Fetch threads
  useEffect(() => {
    if (!session?.user?.id) return;

    const fetchThreads = async () => {
      setThreadsLoading(true);
      const { data, error } = await supabase
        .from('threads')
        .select('id, title, created_at, folder_id')
        .eq('user_id', session.user.id)
        .order('created_at', { ascending: false });

      if (!error && data) setThreads(data);
      setThreadsLoading(false);
    };

    fetchThreads();
  }, [session]);

  // Fetch messages for selected thread
  useEffect(() => {
    if (!selectedThreadId) {
      setMessages([]);
      setMessagesLoading(false);
      return;
    }

    const cached = messagesCacheRef.current.get(selectedThreadId);
    if (cached) {
      setMessages(cached);
      setMessagesLoading(false);
    } else {
      setMessages([]);
      setMessagesLoading(true);
    }

    let isCancelled = false;

    const fetchMessages = async () => {
      const { data, error } = await supabase
        .from('messages')
        .select('id, role, content, created_at')
        .eq('thread_id', selectedThreadId)
        .order('created_at', { ascending: true });

      if (!isCancelled) {
        if (!error && data) {
          const cleanedData = data.map((msg) => {
            const cleanedContent = msg.content.replace(/\n\n\[Attachments:.*?\]/g, '').trim();
            return { ...msg, content: cleanedContent };
          });

          messagesCacheRef.current.set(selectedThreadId, cleanedData);
          setMessages(cleanedData);

          // Fetch attachments
          const messageIds = cleanedData.map((m) => m.id);
          if (messageIds.length > 0) {
            const { data: attachmentsData } = await supabase
              .from('message_attachments')
              .select('message_id, file_name, file_type, file_url, file_size')
              .in('message_id', messageIds);

            if (!isCancelled && attachmentsData) {
              const attachmentsMap = new Map();
              attachmentsData.forEach((att) => {
                if (!attachmentsMap.has(att.message_id)) {
                  attachmentsMap.set(att.message_id, []);
                }
                attachmentsMap.get(att.message_id).push({
                  name: att.file_name,
                  type: att.file_type,
                  url: att.file_url,
                  size: att.file_size,
                });
              });
              setMessageAttachments(attachmentsMap);
            }
          }
        }
        setMessagesLoading(false);
      }
    };

    fetchMessages();
    return () => { isCancelled = true; };
  }, [selectedThreadId]);

  // Search functionality
  useEffect(() => {
    if (!session?.user?.id) return;
    const q = searchQuery.trim();
    if (q.length < 2) {
      setSearchLoading(false);
      setSearchResults({ folders: [], threads: [] });
      return;
    }

    let cancelled = false;
    setSearchLoading(true);

    const normalized = q.replace(/[%_]/g, '\\$&');
    const pattern = `%${normalized}%`;

    const runSearch = async () => {
      try {
        const [foldersRes, threadsRes] = await Promise.all([
          supabase
            .from('folders')
            .select('id, name, created_at')
            .eq('user_id', session.user.id)
            .ilike('name', pattern)
            .order('name', { ascending: true })
            .limit(12),
          supabase
            .from('threads')
            .select('id, title, folder_id, created_at')
            .eq('user_id', session.user.id)
            .ilike('title', pattern)
            .order('created_at', { ascending: false })
            .limit(20),
        ]);

        if (cancelled) return;

        const foldersData = foldersRes.error ? [] : foldersRes.data ?? [];
        const threadsData = threadsRes.error ? [] : threadsRes.data ?? [];

        setSearchResults({ folders: foldersData, threads: threadsData });
      } finally {
        if (!cancelled) setSearchLoading(false);
      }
    };

    runSearch();
    return () => { cancelled = true; };
  }, [searchQuery, session?.user?.id]);

  const profileName = useMemo(() => {
    if (!session) return 'User';
    const name = session.user.user_metadata?.full_name;
    if (name && name.trim().length > 0) return name.trim();
    return session.user.email?.split('@')[0] ?? 'User';
  }, [session]);

  const DEFAULT_TITLE = 'Untitled chat';
  const DEFAULT_FOLDER_NAME = 'Untitled project';

  const filteredThreads = useMemo(() => {
    if (selectedFolderId === 'all') return threads;
    return threads.filter((thread) => thread.folder_id === selectedFolderId);
  }, [threads, selectedFolderId]);

  useEffect(() => {
    if (filteredThreads.length === 0) {
      setSelectedThreadId(null);
      return;
    }
    const exists = filteredThreads.some((thread) => thread.id === selectedThreadId);
    if (!exists) {
      setSelectedThreadId(filteredThreads[0].id);
    }
  }, [filteredThreads, selectedThreadId]);

  const handleCreateThread = async (presetTitle) => {
    if (!session?.user?.id) return null;
    const title = (presetTitle && presetTitle.trim().length > 0) ? presetTitle : DEFAULT_TITLE;
    const folderId = selectedFolderId === 'all' ? null : selectedFolderId;

    setCreatingThread(true);
    const { data, error } = await supabase
      .from('threads')
      .insert({ user_id: session.user.id, title, folder_id: folderId })
      .select('id, title, created_at, folder_id')
      .single();

    setCreatingThread(false);

    if (!error && data) {
      setThreads((prev) => [data, ...prev]);
      setSelectedThreadId(data.id);
      setMessages([]);
      messagesCacheRef.current.set(data.id, []);
      return data.id;
    }
    return null;
  };

  const handleCreateFolder = async () => {
    if (!session?.user?.id) return;
    const { data, error } = await supabase
      .from('folders')
      .insert({ user_id: session.user.id, name: DEFAULT_FOLDER_NAME })
      .select('id, name, created_at')
      .single();

    if (!error && data) {
      setFolders((prev) => [...prev, data]);
      setSelectedFolderId(data.id);
      setEditingFolderId(data.id);
      setEditingFolderName(data.name);
    }
  };

  const handleSelectFolder = (folderId) => {
    setSelectedFolderId(folderId);
    setFolderMenuOpenId(null);
    setEditingFolderId(null);
  };

  const handleSelectThread = (threadId) => {
    setSelectedThreadId(threadId);
    setThreadMenuOpenId(null);
  };

  // File handling
  const MAX_FILE_SIZE = 50 * 1024 * 1024;

  const handleFileSelect = async (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0 || !session?.user?.id) return;

    const oversizedFiles = files.filter((file) => file.size > MAX_FILE_SIZE);
    if (oversizedFiles.length > 0) {
      alert(`File size limit exceeded (50MB max).`);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    setUploadingFiles(true);

    try {
      const uploadPromises = files.map(async (file) => {
        const fileExt = file.name.split('.').pop();
        const fileName = `${Date.now()}_${Math.random().toString(36).substring(7)}.${fileExt}`;
        const filePath = `${session.user.id}/${fileName}`;

        const { data: uploadData, error: uploadError } = await supabase.storage
          .from('files')
          .upload(filePath, file, { cacheControl: '3600', upsert: false });

        if (uploadError) {
          console.error('Upload error:', uploadError);
          return null;
        }

        const { data: urlData } = supabase.storage.from('files').getPublicUrl(filePath);

        return {
          id: uploadData.path,
          name: file.name,
          size: file.size,
          type: file.type,
          url: urlData.publicUrl,
          path: filePath,
        };
      });

      const uploadedFiles = (await Promise.all(uploadPromises)).filter(Boolean);
      setAttachedFiles((prev) => [...prev, ...uploadedFiles]);
    } catch (error) {
      console.error('Error uploading files:', error);
    } finally {
      setUploadingFiles(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleRemoveFile = async (fileId) => {
    const fileToRemove = attachedFiles.find((file) => file.id === fileId);
    setAttachedFiles((prev) => prev.filter((file) => file.id !== fileId));

    if (fileToRemove?.path) {
      try {
        await supabase.storage.from('files').remove([fileToRemove.path]);
      } catch (error) {
        console.error('Error cleaning up file:', error);
      }
    }
  };

  const getFileIcon = useCallback((fileName, fileType) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    if (ext === 'pdf' || fileType?.includes('pdf')) {
      return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="file-type-icon">
          <rect x="4" y="4" width="16" height="16" rx="2" fill="#DC2626" />
          <path d="M8 8h8M8 12h8M8 16h4" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      );
    }
    if (ext === 'docx' || ext === 'doc' || fileType?.includes('word')) {
      return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="file-type-icon">
          <rect x="4" y="4" width="16" height="16" rx="2" fill="#2563EB" />
          <path d="M8 8h3M8 12h6M8 16h4" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      );
    }
    if (ext === 'xlsx' || ext === 'xls' || ext === 'csv' || fileType?.includes('sheet') || fileType?.includes('csv')) {
      return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="file-type-icon">
          <rect x="4" y="4" width="16" height="16" rx="2" fill="#16A34A" />
          <path d="M8 8h8v8H8z" stroke="white" strokeWidth="1.5" />
          <path d="M8 12h8M12 8v8" stroke="white" strokeWidth="1.5" />
        </svg>
      );
    }
    return (
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="file-type-icon">
        <rect x="5" y="4" width="14" height="16" rx="1.5" fill="rgba(156, 163, 175, 0.8)" />
        <path d="M9 8h6M9 12h6M9 16h4" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    );
  }, []);

  const handleAttachClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  // Drag and drop handlers for file drop zone
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    // Only set to false if we're leaving the drop zone entirely
    if (dropZoneRef.current && !dropZoneRef.current.contains(e.relatedTarget)) {
      setIsDraggingOver(false);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  // RAG Document upload handler (defined before drop handler)
  const handleRagFileUpload = useCallback(async (filesToUpload) => {
    if (!session?.access_token || filesToUpload.length === 0) return;
    
    setUploadingToRag(true);
    
    try {
      const formData = new FormData();
      // Backend expects 'files' (plural) as the field name
      for (const file of filesToUpload) {
        formData.append('files', file);
      }
      
      console.log('Uploading files to RAG backend...', filesToUpload.map(f => f.name));
      
      const response = await fetch(`${RAG_BACKEND_URL}/api/documents/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: formData,
      });
      
      const result = await response.json();
      
      if (!response.ok) {
        console.error('Upload failed:', result);
        alert(`Upload failed: ${result.detail || 'Unknown error'}`);
      } else {
        console.log('Upload successful:', result);
        if (result.errors && result.errors.length > 0) {
          console.warn('Upload warnings:', result.errors);
        }
      }
      
      // Refresh document list
      await fetchRagDocuments();
    } catch (error) {
      console.error('Error uploading to RAG:', error);
      alert(`Network error: ${error.message}`);
    } finally {
      setUploadingToRag(false);
    }
  }, [session?.access_token, fetchRagDocuments]);

  // Drop handler for RAG knowledge base (empty state drop zone)
  const handleRagFileDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(false);

    if (!session?.access_token) return;

    const droppedFiles = Array.from(e.dataTransfer.files);
    if (droppedFiles.length === 0) return;

    // Filter by accepted types
    const acceptedExtensions = ['csv', 'xlsx', 'json', 'txt', 'pdf', 'doc', 'docx'];
    const validFiles = droppedFiles.filter((file) => {
      const ext = file.name.split('.').pop()?.toLowerCase();
      return acceptedExtensions.includes(ext);
    });

    if (validFiles.length === 0) {
      alert('Please drop supported files: PDF, DOC, DOCX, TXT, CSV, XLSX, JSON');
      return;
    }

    // Check file sizes
    const oversizedFiles = validFiles.filter((file) => file.size > MAX_FILE_SIZE);
    if (oversizedFiles.length > 0) {
      alert(`Some files exceed the 50MB limit and were skipped.`);
    }

    const filesToUpload = validFiles.filter((file) => file.size <= MAX_FILE_SIZE);
    if (filesToUpload.length === 0) return;

    // Upload to RAG backend
    await handleRagFileUpload(filesToUpload);
  }, [session?.access_token, MAX_FILE_SIZE, handleRagFileUpload]);

  // Main submit handler
  const handleComposerSubmit = async (event) => {
    event.preventDefault();
    const value = composerValue.trim();
    if (!value || !session?.user?.id) return;
    if (uploadingFiles) return;

    let activeThreadId = selectedThreadId;

    if (!activeThreadId) {
      activeThreadId = await handleCreateThread('Untitled chat');
      if (!activeThreadId) return;
    }

    const { data, error } = await supabase
      .from('messages')
      .insert({
        thread_id: activeThreadId,
        user_id: session.user.id,
        role: 'user',
        content: value,
      })
      .select('id, role, content, created_at')
      .single();

    if (!error && data) {
      // Store attachments
      if (attachedFiles.length > 0) {
        const attachmentsData = attachedFiles.map((file) => ({
          message_id: data.id,
          user_id: session.user.id,
          file_name: file.name,
          file_size: file.size,
          file_type: file.type,
          file_url: file.url,
          file_path: file.path,
        }));

        await supabase.from('message_attachments').insert(attachmentsData);

        const attachmentsMap = new Map(messageAttachments);
        attachmentsMap.set(data.id, attachedFiles.map((f) => ({
          name: f.name,
          type: f.type,
          url: f.url,
          size: f.size,
        })));
        setMessageAttachments(attachmentsMap);
      }

      setMessages((prev) => {
        const next = [...prev, data];
        messagesCacheRef.current.set(activeThreadId, next);
        return next;
      });
      setComposerValue('');
      setAttachedFiles([]);
      requestAnimationFrame(adjustComposerHeight);
      bumpThreadToTop(activeThreadId);

      // Auto-title from first message
      const currentThread = threads.find((t) => t.id === activeThreadId);
      if (currentThread?.title === DEFAULT_TITLE) {
        const suggestedTitle = value.length > 48 ? `${value.slice(0, 45)}…` : value;
        const { data: updatedThread } = await supabase
          .from('threads')
          .update({ title: suggestedTitle })
          .eq('id', activeThreadId)
          .select('id, title, created_at, folder_id')
          .single();

        if (updatedThread) bumpThreadToTop(activeThreadId, updatedThread);
      }

      await generateAssistantResponse(activeThreadId, [...messages, data]);
    }
  };

  // Generate the assistant reply with RAG context
  const generateAssistantResponse = async (threadId, updatedMessages) => {
    if (isGenerating) return;

    try {
      setIsGenerating(true);

      const lastMessage = updatedMessages[updatedMessages.length - 1];
      let enhancedMessages = [...updatedMessages];
      let ragContext = '';

      // Retrieve relevant RAG context if user has documents
      if (lastMessage && lastMessage.role === 'user' && ragDocuments.length > 0) {
        try {
          console.log('Querying RAG for context...');
          const ragResponse = await fetch(`${RAG_BACKEND_URL}/api/rag/query`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${session.access_token}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              query: lastMessage.content,
              top_k: 5,
              include_content: true,
              use_qubo: true,
              token_budget: 3000,
              candidate_pool: 20,
            }),
          });

          if (ragResponse.ok) {
            const ragData = await ragResponse.json();
            if (ragData.context && ragData.chunks?.length > 0) {
              ragContext = ragData.context;
              console.log(`RAG: Found ${ragData.chunks.length} relevant chunks (${ragData.total_tokens} tokens)`);
              console.log(`RAG: Selection method: ${ragData.selection_method}, RF model: ${ragData.rf_model_loaded}`);
              console.log(`RAG: Context preview (first 300 chars):`, ragContext.substring(0, 300));
              console.log(`RAG: Chunks from files:`, ragData.chunks.map(c => c.document_filename || 'unknown').join(', '));
              
              // Fetch comparison data for this query
              try {
                const compareResponse = await fetch(`${RAG_BACKEND_URL}/api/rag/compare`, {
                  method: 'POST',
                  headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({
                    query: lastMessage.content,
                    top_k: 5,
                    token_budget: 3000,
                    candidate_pool: 20,
                  }),
                });
                
                if (compareResponse.ok) {
                  const compareData = await compareResponse.json();
                  setRagComparison(compareData);
                  setShowRagComparison(true);
                  console.log('RAG Comparison:', compareData.comparison);
                }
              } catch (compareError) {
                console.error('RAG comparison failed:', compareError);
              }
            }
          }
        } catch (ragError) {
          console.error('RAG query failed:', ragError);
        }
      }

      // Process file attachments if present
      if (lastMessage && lastMessage.role === 'user') {
        try {
          const { data: attachmentData } = await supabase
            .from('message_attachments')
            .select('file_name, file_type, file_url, file_path')
            .eq('message_id', lastMessage.id);

          if (attachmentData && attachmentData.length > 0) {
            const fileContents = await Promise.all(
              attachmentData.map(async (attachment) => {
                try {
                  let fileUrl = attachment.file_url;

                  if (attachment.file_path) {
                    const { data: signedUrlData } = await supabase.storage
                      .from('files')
                      .createSignedUrl(attachment.file_path, 3600);

                    if (signedUrlData?.signedUrl) {
                      fileUrl = signedUrlData.signedUrl;
                    }
                  }

                  const text = await extractFileText(fileUrl, attachment.file_name, attachment.file_type);
                  return { fileName: attachment.file_name, content: text };
                } catch (error) {
                  return { fileName: attachment.file_name, content: `[Error reading file]` };
                }
              })
            );

            const fileContentText = fileContents
              .map((file) => `\n\n[File: ${file.fileName}]\n${file.content}`)
              .join('\n\n---\n');

            enhancedMessages = updatedMessages.map((msg, idx) => {
              if (idx === updatedMessages.length - 1 && msg.role === 'user') {
                return { ...msg, content: msg.content + fileContentText };
              }
              return msg;
            });
          }
        } catch (error) {
          console.error('Error processing attachments:', error);
        }
      }

      // The grounding instructions now live server-side in /api/chat; the client
      // just hands over the transcript and whatever context retrieval selected.
      const payload = {
        messages: enhancedMessages.slice(-12).map((message) => ({
          role: message.role === 'assistant' ? 'assistant' : 'user',
          content: message.content,
        })),
        context: ragContext || null,
      };

      const body = await generateWithRetries(payload);

      let text = (body?.content ?? '').trim();

      if (!text) text = "I'm having trouble responding right now.";

      const { data: assistantMsg, error: assistantError } = await supabase
        .from('messages')
        .insert({
          thread_id: threadId,
          user_id: session.user.id,
          role: 'assistant',
          content: text,
        })
        .select('id, role, content, created_at')
        .single();

      if (!assistantError && assistantMsg) {
        setMessages((prev) => {
          const next = [...prev, assistantMsg];
          messagesCacheRef.current.set(threadId, next);
          return next;
        });
        bumpThreadToTop(threadId);
      }
    } catch (error) {
      console.error('Assistant generation error:', error);
      const { data: fallback } = await supabase
        .from('messages')
        .insert({
          thread_id: threadId,
          user_id: session.user.id,
          role: 'assistant',
          content: error instanceof Error
            ? `Error: ${error.message}`
            : 'Something went wrong. Please try again.',
        })
        .select('id, role, content, created_at')
        .single();

      if (fallback) {
        setMessages((prev) => {
          const next = [...prev, fallback];
          messagesCacheRef.current.set(threadId, next);
          return next;
        });
      }
    } finally {
      setIsGenerating(false);
    }
  };

  // Thread/folder menu handlers
  const handleOpenThreadMenu = (event, threadId) => {
    event.stopPropagation();
    setFolderMenuOpenId(null);
    setThreadMenuOpenId((prev) => (prev === threadId ? null : threadId));
  };

  const handleOpenFolderMenu = (event, folderId) => {
    event.stopPropagation();
    setThreadMenuOpenId(null);
    setFolderMenuOpenId((prev) => (prev === folderId ? null : folderId));
  };

  const handleBeginRenameThread = (threadId) => {
    const current = threads.find((t) => t.id === threadId);
    if (!current) return;
    setEditingThreadId(threadId);
    setEditingTitle(current.title);
    setThreadMenuOpenId(null);
  };

  const handleBeginRenameFolder = (folderId) => {
    const current = folders.find((f) => f.id === folderId);
    if (!current) return;
    setEditingFolderId(folderId);
    setEditingFolderName(current.name);
    setFolderMenuOpenId(null);
  };

  const commitRenameThread = async () => {
    if (!editingThreadId) return;
    const current = threads.find((t) => t.id === editingThreadId);
    if (!current) {
      setEditingThreadId(null);
      return;
    }
    const trimmed = editingTitle.trim();
    if (!trimmed || trimmed === current.title) {
      setEditingThreadId(null);
      setEditingTitle('');
      return;
    }

    const { data } = await supabase
      .from('threads')
      .update({ title: trimmed })
      .eq('id', editingThreadId)
      .select('id, title, created_at, folder_id')
      .single();

    if (data) bumpThreadToTop(editingThreadId, data);
    setEditingThreadId(null);
    setEditingTitle('');
  };

  const commitRenameFolder = async () => {
    if (!editingFolderId) return;
    const current = folders.find((f) => f.id === editingFolderId);
    if (!current) {
      setEditingFolderId(null);
      return;
    }

    const trimmed = editingFolderName.trim();
    if (!trimmed || trimmed === current.name) {
      setEditingFolderId(null);
      setEditingFolderName('');
      return;
    }

    const { data } = await supabase
      .from('folders')
      .update({ name: trimmed })
      .eq('id', editingFolderId)
      .eq('user_id', session?.user?.id ?? '')
      .select('id, name, created_at')
      .single();

    if (data) {
      setFolders((prev) => prev.map((f) => (f.id === data.id ? data : f)));
    }
    setEditingFolderId(null);
    setEditingFolderName('');
  };

  const handleDeleteThreadRequest = (threadId) => {
    const thread = threads.find((t) => t.id === threadId);
    if (thread) {
      setDeleteConfirmThread(thread);
      setThreadMenuOpenId(null);
    }
  };

  const handleDeleteThreadConfirm = async () => {
    if (!session?.user?.id || !deleteConfirmThread) return;
    const threadId = deleteConfirmThread.id;

    const { data: threadMessages } = await supabase
      .from('messages')
      .select('id')
      .eq('thread_id', threadId)
      .eq('user_id', session.user.id);

    const messageIds = threadMessages?.map((msg) => msg.id) || [];

    // Cleanup files
    if (messageIds.length > 0) {
      const { data: attachments } = await supabase
        .from('message_attachments')
        .select('file_path')
        .in('message_id', messageIds);

      if (attachments?.length > 0) {
        const filePaths = attachments.map((att) => att.file_path);
        await supabase.storage.from('files').remove(filePaths);
      }
    }

    const { error } = await supabase.from('threads').delete().eq('id', threadId).eq('user_id', session.user.id);

    if (!error) {
      setThreads((prev) => prev.filter((t) => t.id !== threadId));
      if (selectedThreadId === threadId) {
        const remaining = threads.filter((t) => t.id !== threadId);
        setSelectedThreadId(remaining[0]?.id ?? null);
        setMessages([]);
      }
      messagesCacheRef.current.delete(threadId);
    }
    setDeleteConfirmThread(null);
  };

  const handleDeleteThreadCancel = () => {
    setDeleteConfirmThread(null);
  };

  const handleDeleteFolder = async (folderId) => {
    if (!session?.user?.id) return;
    await supabase
      .from('threads')
      .update({ folder_id: null })
      .eq('folder_id', folderId)
      .eq('user_id', session.user.id);

    const { error } = await supabase.from('folders').delete().eq('id', folderId).eq('user_id', session.user.id);
    if (!error) {
      setFolders((prev) => prev.filter((f) => f.id !== folderId));
      setThreads((prev) => prev.map((t) => (t.folder_id === folderId ? { ...t, folder_id: null } : t)));
      if (selectedFolderId === folderId) setSelectedFolderId('all');
    }
    setFolderMenuOpenId(null);
  };

  // RAG Document handlers
  const handleRagFileSelect = (event) => {
    console.log('File select triggered', event.target.files);
    const files = Array.from(event.target.files || []);
    if (files.length > 0) {
      console.log('Processing files:', files.map(f => f.name));
      handleRagFileUpload(files);
    }
    if (ragFileInputRef.current) ragFileInputRef.current.value = '';
  };

  const handleDeleteRagDocRequest = (doc) => {
    setDeleteConfirmDoc(doc);
  };

  const handleDeleteRagDocConfirm = async () => {
    if (!session?.access_token || !deleteConfirmDoc) return;
    
    try {
      const response = await fetch(`${RAG_BACKEND_URL}/api/documents/${deleteConfirmDoc.id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      });
      
      if (response.ok) {
        setRagDocuments((prev) => prev.filter((d) => d.id !== deleteConfirmDoc.id));
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    } finally {
      setDeleteConfirmDoc(null);
    }
  };

  const handleDeleteRagDocCancel = () => {
    setDeleteConfirmDoc(null);
  };

  // Drag & drop for threads into folders
  const handleThreadDragStart = (event, threadId) => {
    event.stopPropagation?.();
    setDraggedThreadId(threadId);
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', threadId);
  };

  const handleThreadDragEnd = () => {
    setDraggedThreadId(null);
    setDragOverFolderId(null);
  };

  const handleFolderDragOver = (event, folderId) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    setDragOverFolderId(folderId);
  };

  const handleFolderDragLeave = () => setDragOverFolderId(null);

  const handleFolderDrop = async (event, folderId) => {
    event.preventDefault();
    const threadId = event.dataTransfer.getData('text/plain') || draggedThreadId;
    if (!threadId || !session?.user?.id) return;

    const targetFolder = folderId === 'all' ? null : folderId;

    const { data } = await supabase
      .from('threads')
      .update({ folder_id: targetFolder })
      .eq('id', threadId)
      .eq('user_id', session.user.id)
      .select('id, title, created_at, folder_id')
      .single();

    if (data) {
      setThreads((prev) => prev.map((t) => (t.id === data.id ? data : t)));
    }
    setDragOverFolderId(null);
    setDraggedThreadId(null);
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    messagesCacheRef.current.clear();
    navigate('/login', { replace: true });
  };

  // Close menus on click outside
  useEffect(() => {
    if (!threadMenuOpenId) return;
    const handleClick = () => setThreadMenuOpenId(null);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [threadMenuOpenId]);

  useEffect(() => {
    if (!folderMenuOpenId) return;
    const handleClick = () => setFolderMenuOpenId(null);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [folderMenuOpenId]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (stage !== 'chat') return;
    const feedEl = feedRef.current;
    const anchorEl = bottomAnchorRef.current;
    if (!feedEl || !anchorEl) return;

    const scrollBehavior = messagesLoading ? 'auto' : 'smooth';
    const scrollToBottom = () => {
      anchorEl.scrollIntoView({ behavior: scrollBehavior, block: 'end' });
    };

    const handle = requestAnimationFrame(scrollToBottom);
    return () => cancelAnimationFrame(handle);
  }, [messages, messagesLoading, stage]);

  useEffect(() => {
    adjustComposerHeight();
  }, [adjustComposerHeight, composerValue, stage]);

  // Loading state
  if (!sessionChecked) {
    return (
      <div className="workspace-page">
        <div className="workspace-loader">
          <div className="ring" aria-hidden="true" />
          <span>Loading…</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`workspace-page ${stage === 'intro' ? 'stage-intro' : 'stage-chat'}`}>
      {stage === 'intro' ? (
        <div className="intro-sequence">
          <div className="particle-field" aria-hidden="true">
            {Array.from({ length: 28 }).map((_, index) => (
              <span key={index} className={`particle particle-${index + 1}`} />
            ))}
          </div>
          <div className="atom-container">
            <div className="atom-core">
              <span className="core-glow" />
              <span className="core-glyph">S</span>
            </div>
            <div className="orbit orbit-one">
              <span className="electron" />
            </div>
            <div className="orbit orbit-two">
              <span className="electron" />
            </div>
            <div className="orbit orbit-three">
              <span className="electron" />
            </div>
          </div>
          <div className="signature-wrapper">
            <svg className="signature-svg" viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="signature-grad" x1="40" y1="110" x2="600" y2="110" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stopColor="#9DEAFB" />
                  <stop offset="50%" stopColor="#67C5FF" />
                  <stop offset="100%" stopColor="#3F7CFF" />
                </linearGradient>
              </defs>
              <text x="50%" y="58%" textAnchor="middle" className="signature-text">
                Synergy
              </text>
            </svg>
          </div>
        </div>
      ) : (
        <div className={`workspace-frame ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
          <aside className="gpt-sidebar">
            <header className="gpt-sidebar__brand">
              {!sidebarCollapsed && (
                <>
                  <div className="logo">
                    <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                      <defs>
                        <linearGradient id="workspace-gradient" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0" stopColor="#4f46ef" />
                          <stop offset="0.45" stopColor="#2563eb" />
                          <stop offset="1" stopColor="#0ea5e9" />
                        </linearGradient>
                      </defs>
                      <rect x="6" y="6" width="28" height="28" rx="10" fill="url(#workspace-gradient)" />
                      <circle cx="20" cy="20" r="6.4" fill="none" stroke="rgba(255,255,255,0.68)" strokeWidth="1.6" />
                      <circle cx="20" cy="20" r="3" fill="#f8fbff" />
                    </svg>
                  </div>
                  <div>
                    <strong>Synergy</strong>
                    <span>AI Assistant</span>
                  </div>
                </>
              )}
              <button
                type="button"
                className={`sidebar-toggle ${sidebarCollapsed ? 'collapsed' : ''}`}
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                {sidebarCollapsed ? (
                  <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" className="synergy-logo-icon">
                    <defs>
                      <linearGradient id="workspace-gradient-toggle" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0" stopColor="#4f46ef" />
                        <stop offset="0.45" stopColor="#2563eb" />
                        <stop offset="1" stopColor="#0ea5e9" />
                      </linearGradient>
                    </defs>
                    <rect x="6" y="6" width="28" height="28" rx="10" fill="url(#workspace-gradient-toggle)" />
                    <circle cx="20" cy="20" r="6.4" fill="none" stroke="rgba(255,255,255,0.68)" strokeWidth="1.6" />
                    <circle cx="20" cy="20" r="3" fill="#f8fbff" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="three-dots-icon">
                    <circle cx="6" cy="12" r="1.5" fill="currentColor" />
                    <circle cx="12" cy="12" r="1.5" fill="currentColor" />
                    <circle cx="18" cy="12" r="1.5" fill="currentColor" />
                  </svg>
                )}
              </button>
            </header>

            <div className="gpt-sidebar__scroll">
              <button type="button" className="gpt-sidebar__new" disabled={creatingThread} onClick={() => handleCreateThread()}>
                <span>{creatingThread ? 'Creating…' : 'New chat'}</span>
                {!creatingThread && (
                  <>
                    <kbd>Ctrl</kbd>
                    <kbd>Shift</kbd>
                    <kbd>O</kbd>
                  </>
                )}
              </button>

              <div className={`sidebar-search ${searchActive ? 'active' : ''}`}>
                <button type="button" className="sidebar-search__trigger" onClick={() => setSearchActive(true)}>
                  <span className="sidebar-search__icon">
                    <svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                      <path
                        d="M9 2.5a6.5 6.5 0 1 1 0 13 6.5 6.5 0 0 1 0-13Zm0 2a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9Zm4.9 7.8 3.1 3.1a1 1 0 0 1-1.41 1.4l-3.1-3.08a7 7 0 0 0 1.41-1.42Z"
                        fill="currentColor"
                      />
                    </svg>
                  </span>
                  <span>Search chats</span>
                </button>

                {searchActive && (
                  <div className="sidebar-search__overlay" onClick={() => { setSearchActive(false); setSearchQuery(''); }}>
                    <div className="sidebar-search__panel" onClick={(e) => e.stopPropagation()}>
                      <div className="sidebar-search__header">
                        <input
                          ref={searchInputRef}
                          type="text"
                          placeholder="Search chats..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                        />
                        <button type="button" onClick={() => { setSearchActive(false); setSearchQuery(''); }}>✕</button>
                      </div>
                      <div className="sidebar-search__content">
                        {searchLoading ? (
                          <div className="sidebar-search__status">Searching…</div>
                        ) : searchQuery.trim().length < 2 ? (
                          <div className="sidebar-search__status">Type at least 2 characters to search</div>
                        ) : searchResults.folders.length === 0 && searchResults.threads.length === 0 ? (
                          <div className="sidebar-search__status">No results found</div>
                        ) : (
                          <>
                            {searchResults.folders.length > 0 && (
                              <div className="sidebar-search__group">
                                <span className="sidebar-search__group-label">Projects</span>
                                <ul>
                                  {searchResults.folders.map((folder) => (
                                    <li key={folder.id}>
                                      <button
                                        type="button"
                                        onClick={() => {
                                          handleSelectFolder(folder.id);
                                          setSearchActive(false);
                                          setSearchQuery('');
                                        }}
                                      >
                                        <span className="search-result-icon">📁</span>
                                        <span className="search-result-text">
                                          <strong>{folder.name}</strong>
                                        </span>
                                      </button>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                            {searchResults.threads.length > 0 && (
                              <div className="sidebar-search__group">
                                <span className="sidebar-search__group-label">Chats</span>
                                <ul>
                                  {searchResults.threads.map((thread) => (
                                    <li key={thread.id}>
                                      <button
                                        type="button"
                                        onClick={() => {
                                          handleSelectThread(thread.id);
                                          setSearchActive(false);
                                          setSearchQuery('');
                                        }}
                                      >
                                        <span className="search-result-icon search-result-icon--thread">💬</span>
                                        <span className="search-result-text">
                                          <strong>{thread.title}</strong>
                                        </span>
                                      </button>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="gpt-sidebar__folders">
                <div className="folders-header">
                  <h3>Projects</h3>
                  <button type="button" onClick={handleCreateFolder} aria-label="Create project">＋</button>
                </div>
                <ul>
                  <li
                    className={`folder-item ${selectedFolderId === 'all' ? 'active' : ''} ${dragOverFolderId === 'all' ? 'drag-over' : ''}`}
                    onClick={() => handleSelectFolder('all')}
                    onDragOver={(e) => handleFolderDragOver(e, 'all')}
                    onDragLeave={handleFolderDragLeave}
                    onDrop={(e) => handleFolderDrop(e, 'all')}
                  >
                    <span className="folder-icon">
                      <svg viewBox="0 0 28 24" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                          <linearGradient id="folder-gradient-all" x1="4" y1="3" x2="22" y2="21" gradientUnits="userSpaceOnUse">
                            <stop offset="0%" stopColor="#4f46ef" />
                            <stop offset="50%" stopColor="#2dd4ff" />
                            <stop offset="100%" stopColor="#0ea5e9" />
                          </linearGradient>
                        </defs>
                        <path
                          d="M4 5.5A2.5 2.5 0 0 1 6.5 3h4.054a2.5 2.5 0 0 1 1.768.732l1.292 1.292a2 2 0 0 0 1.42.59H21a2 2 0 0 1 2 2v8.67A2.33 2.33 0 0 1 20.67 19H5.33A2.33 2.33 0 0 1 3 16.67V6Z"
                          fill="url(#folder-gradient-all)"
                        />
                        <path
                          d="M3 10h16.2a2.8 2.8 0 0 1 2.8 2.8V16.5A2.5 2.5 0 0 1 19.5 19H4.5A2.5 2.5 0 0 1 2 16.5V11a1 1 0 0 1 1-1Z"
                          fill="rgba(6, 12, 32, 0.6)"
                        />
                      </svg>
                    </span>
                    <span className="folder-name">All chats</span>
                  </li>
                  {foldersLoading ? (
                    <li className="loading">Loading…</li>
                  ) : folders.length === 0 ? (
                    <li className="empty">No projects yet</li>
                  ) : (
                    folders.map((folder) => (
                      <li
                        key={folder.id}
                        className={`folder-item ${selectedFolderId === folder.id ? 'active' : ''} ${dragOverFolderId === folder.id ? 'drag-over' : ''}`}
                        onClick={() => handleSelectFolder(folder.id)}
                        onDragOver={(e) => handleFolderDragOver(e, folder.id)}
                        onDragLeave={handleFolderDragLeave}
                        onDrop={(e) => handleFolderDrop(e, folder.id)}
                      >
                        <button type="button" className="folder-select">
                          <span className="folder-icon">
                            <svg viewBox="0 0 28 24" xmlns="http://www.w3.org/2000/svg">
                              <defs>
                                <linearGradient id={`folder-gradient-${folder.id}`} x1="4" y1="3" x2="22" y2="21" gradientUnits="userSpaceOnUse">
                                  <stop offset="0%" stopColor="#5b5afc" />
                                  <stop offset="50%" stopColor="#38b6ff" />
                                  <stop offset="100%" stopColor="#22d3ee" />
                                </linearGradient>
                              </defs>
                              <path
                                d="M4 5.5A2.5 2.5 0 0 1 6.5 3h4.054a2.5 2.5 0 0 1 1.768.732l1.292 1.292a2.5 2.5 0 0 0 1.768.732H21.5A2.5 2.5 0 0 1 24 8.256V18.5A2.5 2.5 0 0 1 21.5 21H6.5A2.5 2.5 0 0 1 4 18.5V5.5Z"
                                fill={`url(#folder-gradient-${folder.id})`}
                              />
                              <path
                                d="M4 8.25h16.5a2.5 2.5 0 0 1 2.5 2.5V18.5A2.5 2.5 0 0 1 20.5 21H5.5A2.5 2.5 0 0 1 3 18.5V9.25A1 1 0 0 1 4 8.25Z"
                                fill="rgba(5, 12, 32, 0.45)"
                              />
                            </svg>
                          </span>
                          {editingFolderId === folder.id ? (
                            <input
                              className="folder-edit"
                              value={editingFolderName}
                              onChange={(e) => setEditingFolderName(e.target.value)}
                              onBlur={commitRenameFolder}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') { e.preventDefault(); commitRenameFolder(); }
                                if (e.key === 'Escape') { setEditingFolderId(null); setEditingFolderName(''); }
                              }}
                              autoFocus
                            />
                          ) : (
                            <span className="folder-name">{folder.name}</span>
                          )}
                        </button>
                        <div className="folder-actions" onClick={(e) => e.stopPropagation()}>
                          <button type="button" className="folder-more" onClick={(e) => handleOpenFolderMenu(e, folder.id)}>⋯</button>
                          {folderMenuOpenId === folder.id && (
                            <div className="thread-menu" onClick={(e) => e.stopPropagation()}>
                              <button type="button" onClick={() => handleBeginRenameFolder(folder.id)}>Rename</button>
                              <button type="button" className="danger" onClick={() => handleDeleteFolder(folder.id)}>Delete</button>
                            </div>
                          )}
                        </div>
                      </li>
                    ))
                  )}
                </ul>
              </div>

              {/* RAG Documents Section */}
              <div className="gpt-sidebar__documents">
                <div className="documents-header" onClick={() => setRagDocsExpanded(!ragDocsExpanded)}>
                  <div className="documents-header-left">
                    <span className={`documents-chevron ${ragDocsExpanded ? 'expanded' : ''}`}>
                      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="9 18 15 12 9 6" />
                      </svg>
                    </span>
                    <h3>Documents</h3>
                    <span className="documents-count">{ragDocuments.length}</span>
                  </div>
                  <button 
                    type="button" 
                    className="documents-add-btn"
                    onClick={(e) => { e.stopPropagation(); ragFileInputRef.current?.click(); }}
                    aria-label="Add document"
                  >
                    ＋
                  </button>
                  <input
                    ref={ragFileInputRef}
                    type="file"
                    multiple
                    accept=".pdf,.doc,.docx,.txt,.csv,.xlsx,.json"
                    style={{ display: 'none' }}
                    onChange={handleRagFileSelect}
                  />
                </div>
                
                {ragDocsExpanded && (
                  <ul className="documents-list">
                    {ragDocsLoading ? (
                      <li className="loading">Loading…</li>
                    ) : ragDocuments.length === 0 ? (
                      <li className="empty">No documents uploaded</li>
                    ) : (
                      ragDocuments.map((doc) => (
                        <li key={doc.id} className="document-item">
                          <span className="document-icon">
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                              <polyline points="14 2 14 8 20 8" />
                            </svg>
                          </span>
                          <span className="document-name">{doc.filename}</span>
                          <button
                            type="button"
                            className="document-delete"
                            onClick={() => handleDeleteRagDocRequest(doc)}
                            aria-label="Delete document"
                          >
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                              <polyline points="3 6 5 6 21 6" />
                              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                            </svg>
                          </button>
                        </li>
                      ))
                    )}
                    {uploadingToRag && (
                      <li className="uploading">Uploading…</li>
                    )}
                  </ul>
                )}
              </div>

              <div className="gpt-sidebar__list">
                <h3>Your chats</h3>
                <ul>
                  {threadsLoading ? (
                    <li className="loading">Loading…</li>
                  ) : filteredThreads.length === 0 ? (
                    <li className="empty">{selectedFolderId === 'all' ? 'No conversations yet' : 'No chats in this project'}</li>
                  ) : (
                    filteredThreads.map((thread) => {
                      const isActive = thread.id === selectedThreadId;
                      const isMenuOpen = threadMenuOpenId === thread.id;
                      const isDragging = draggedThreadId === thread.id;
                      const className = ['thread-item'];
                      if (isActive) className.push('active');
                      if (isMenuOpen) className.push('menu-open');
                      if (isDragging) className.push('dragging');

                      return (
                        <li
                          key={thread.id}
                          className={className.join(' ')}
                          draggable
                          onDragStart={(e) => handleThreadDragStart(e, thread.id)}
                          onDragEnd={handleThreadDragEnd}
                        >
                          <button type="button" className="thread-select" onClick={() => handleSelectThread(thread.id)}>
                            {editingThreadId === thread.id ? (
                              <input
                                className="thread-edit"
                                value={editingTitle}
                                onChange={(e) => setEditingTitle(e.target.value)}
                                onBlur={commitRenameThread}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') { e.preventDefault(); commitRenameThread(); }
                                  if (e.key === 'Escape') { setEditingThreadId(null); setEditingTitle(''); }
                                }}
                                autoFocus
                                onClick={(e) => e.stopPropagation()}
                              />
                            ) : (
                              <span className="thread-title">{thread.title}</span>
                            )}
                          </button>
                          <div className="thread-actions" onClick={(e) => e.stopPropagation()}>
                            <button type="button" className="thread-more" onClick={(e) => handleOpenThreadMenu(e, thread.id)}>⋯</button>
                            {threadMenuOpenId === thread.id && (
                              <div className="thread-menu" onClick={(e) => e.stopPropagation()}>
                                <button type="button" onClick={() => handleBeginRenameThread(thread.id)}>Rename</button>
                                <button type="button" className="danger" onClick={() => handleDeleteThreadRequest(thread.id)}>Delete</button>
                              </div>
                            )}
                          </div>
                        </li>
                      );
                    })
                  )}
                </ul>
              </div>

              <div className="sidebar-footer">
                <button type="button" className="sidebar-logout" onClick={handleLogout}>
                  <span className="sidebar-logout__icon" aria-hidden="true">⏻</span>
                  <span>Logout</span>
                </button>
              </div>
            </div>
          </aside>

          <main className="gpt-main">
            {messages.length === 0 && !messagesLoading ? (
              /* Empty state with drop zone */
              <div className="empty-state">
                <div className="gpt-hero">
                  <div className="gpt-logo" aria-hidden="true">
                    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
                      <defs>
                        <linearGradient id="synergy-glyph-grad" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0" stopColor="#4f46ef" />
                          <stop offset="0.45" stopColor="#2563eb" />
                          <stop offset="1" stopColor="#0ea5e9" />
                        </linearGradient>
                      </defs>
                      <path
                        d="M24 4c4.4 0 8 3.6 8 8 0 2.7-1.3 5.1-3.4 6.6l0.9 2.8H36c4.4 0 8 3.6 8 8s-3.6 8-8 8h-6.1l-1.3 4.1c-0.7 2.3-2.9 3.9-5.3 3.9s-4.6-1.6-5.3-3.9l-1.3-4.1H12c-4.4 0-8-3.6-8-8s3.6-8 8-8h4.5l0.9-2.8C15.3 17.1 14 14.7 14 12c0-4.4 3.6-8 8-8z"
                        fill="url(#synergy-glyph-grad)"
                      />
                      <circle cx="24" cy="24" r="5.2" fill="#050a1a" opacity="0.5" />
                      <circle cx="24" cy="24" r="3" fill="#f8fbff" />
                    </svg>
                  </div>
                  <h1>Hi, {profileName}</h1>
                  <p>Upload documents and ask questions</p>
                </div>

                {/* Central Drop Zone - RAG Knowledge Base */}
                <div
                  ref={dropZoneRef}
                  className={`file-drop-zone ${isDraggingOver ? 'dragging' : ''} ${ragDocuments.length > 0 ? 'has-files' : ''}`}
                  onDragEnter={handleDragEnter}
                  onDragLeave={handleDragLeave}
                  onDragOver={handleDragOver}
                  onDrop={handleRagFileDrop}
                  onClick={() => ragFileInputRef.current?.click()}
                >
                  {uploadingToRag ? (
                    <div className="drop-zone-uploading">
                      <div className="upload-spinner" />
                      <span>Uploading to knowledge base...</span>
                    </div>
                  ) : ragDocuments.length > 0 ? (
                    <div className="drop-zone-files">
                      <div className="drop-zone-files-grid">
                        {ragDocuments.slice(0, 6).map((doc) => (
                          <div key={doc.id} className="drop-zone-file-card">
                            <div className="drop-zone-file-icon">
                              <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="rgba(34, 211, 238, 0.8)" strokeWidth="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                <polyline points="14 2 14 8 20 8" />
                              </svg>
                            </div>
                            <span className="drop-zone-file-name" title={doc.filename}>{doc.filename}</span>
                            <button
                              type="button"
                              className="drop-zone-file-remove"
                              onClick={(e) => { e.stopPropagation(); handleDeleteRagDocRequest(doc); }}
                              aria-label="Remove document"
                            >
                              <svg viewBox="0 0 24 24" fill="none" width="16" height="16">
                                <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                              </svg>
                            </button>
                          </div>
                        ))}
                      </div>
                      {ragDocuments.length > 6 && (
                        <p className="drop-zone-hint">+{ragDocuments.length - 6} more documents in sidebar</p>
                      )}
                      <p className="drop-zone-hint">Drop more files or click to add</p>
                    </div>
                  ) : (
                    <div className="drop-zone-empty">
                      <div className="drop-zone-icon">
                        <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <rect x="8" y="12" width="48" height="40" rx="4" stroke="url(#drop-gradient)" strokeWidth="2" strokeDasharray="4 2" />
                          <path d="M32 24v16M24 32h16" stroke="url(#drop-gradient)" strokeWidth="2" strokeLinecap="round" />
                          <defs>
                            <linearGradient id="drop-gradient" x1="8" y1="12" x2="56" y2="52">
                              <stop stopColor="#4f46ef" />
                              <stop offset="0.5" stopColor="#2563eb" />
                              <stop offset="1" stopColor="#0ea5e9" />
                            </linearGradient>
                          </defs>
                        </svg>
                      </div>
                      <h3>Add to knowledge base</h3>
                      <p>Drop files or click to browse</p>
                      <span className="drop-zone-formats">PDF, DOC, DOCX, TXT, CSV, XLSX, JSON</span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              /* Chat messages view */
              <>
                <div className="gpt-hero gpt-hero--compact">
                  <div className="gpt-logo gpt-logo--small" aria-hidden="true">
                    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
                      <defs>
                        <linearGradient id="synergy-glyph-grad-small" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0" stopColor="#4f46ef" />
                          <stop offset="0.45" stopColor="#2563eb" />
                          <stop offset="1" stopColor="#0ea5e9" />
                        </linearGradient>
                      </defs>
                      <path
                        d="M24 4c4.4 0 8 3.6 8 8 0 2.7-1.3 5.1-3.4 6.6l0.9 2.8H36c4.4 0 8 3.6 8 8s-3.6 8-8 8h-6.1l-1.3 4.1c-0.7 2.3-2.9 3.9-5.3 3.9s-4.6-1.6-5.3-3.9l-1.3-4.1H12c-4.4 0-8-3.6-8-8s3.6-8 8-8h4.5l0.9-2.8C15.3 17.1 14 14.7 14 12c0-4.4 3.6-8 8-8z"
                        fill="url(#synergy-glyph-grad-small)"
                      />
                      <circle cx="24" cy="24" r="5.2" fill="#050a1a" opacity="0.5" />
                      <circle cx="24" cy="24" r="3" fill="#f8fbff" />
                    </svg>
                  </div>
                </div>

                <section className="message-feed" ref={feedRef}>
                  {messagesLoading ? (
                    <div className="message-loader">Loading conversation…</div>
                  ) : (
                    messages.map((message) => {
                      const attachments = messageAttachments.get(message.id) || [];
                      return (
                        <div key={message.id} className={`message-entry ${message.role}`}>
                          {attachments.length > 0 && (
                            <div className="message-attachments">
                              {attachments.map((file, idx) => (
                                <div key={idx} className="message-attachment-item">
                                  <div className="message-attachment-icon">{getFileIcon(file.name, file.type)}</div>
                                  <div className="message-attachment-content">
                                    <span className="message-attachment-name" title={file.name}>{file.name}</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                          {message.content && (
                            <div className="message-speech" dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} />
                          )}
                        </div>
                      );
                    })
                  )}
                  <div ref={bottomAnchorRef} />
                </section>

                {/* RAG Comparison Panel */}
                {showRagComparison && ragComparison && (
                  <div className="rag-comparison-panel">
                    <div className="rag-comparison-header">
                      <h4>RAG Performance Comparison</h4>
                      {ragComparison.comparison?.qubo_solver && (
                        // Which solver produced these numbers, and at what energy.
                        // A comparison panel with no attribution invites the
                        // question "compared how?" and can't answer it.
                        <span className="rag-solver-readout">
                          {ragComparison.comparison.qubo_solver.solver}
                          {typeof ragComparison.comparison.qubo_solver.energy === 'number' && (
                            <> · E={ragComparison.comparison.qubo_solver.energy.toFixed(2)}</>
                          )}
                          {ragComparison.comparison.qubo_solver.num_variables > 0 && (
                            <> · {ragComparison.comparison.qubo_solver.num_variables} vars</>
                          )}
                        </span>
                      )}
                      <button
                        type="button"
                        className="rag-comparison-close"
                        onClick={() => setShowRagComparison(false)}
                      >
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M18 6L6 18M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    
                    <div className="rag-comparison-grid">
                      {/* Traditional RAG Column */}
                      <div className="rag-comparison-column traditional">
                        <div className="rag-comparison-column-header">
                          <span className="rag-method-icon">📊</span>
                          <span>Traditional RAG</span>
                        </div>
                        <div className="rag-comparison-stats">
                          <div className="rag-stat">
                            <span className="rag-stat-label">Chunks</span>
                            <span className="rag-stat-value">{ragComparison.traditional_rag?.chunks_selected || 0}</span>
                          </div>
                          <div className="rag-stat">
                            <span className="rag-stat-label">Tokens</span>
                            <span className="rag-stat-value">{ragComparison.traditional_rag?.total_tokens || 0}</span>
                          </div>
                          <div className="rag-stat">
                            <span className="rag-stat-label">Avg Similarity</span>
                            <span className="rag-stat-value">{((ragComparison.traditional_rag?.avg_similarity || 0) * 100).toFixed(1)}%</span>
                          </div>
                          <div className="rag-stat">
                            <span className="rag-stat-label">Avg RF Score</span>
                            <span className="rag-stat-value">{((ragComparison.traditional_rag?.avg_rf_score || 0) * 100).toFixed(1)}%</span>
                          </div>
                          <div className="rag-stat warning">
                            <span className="rag-stat-label">Redundant Pairs</span>
                            <span className="rag-stat-value">{ragComparison.traditional_rag?.redundancy_pairs || 0}</span>
                          </div>
                        </div>
                      </div>

                      {/* QUBO RAG Column */}
                      <div className="rag-comparison-column qubo">
                        <div className="rag-comparison-column-header">
                          <span className="rag-method-icon">🎯</span>
                          <span>{ragComparison.comparison?.selector_label || 'Optimized Selection'}</span>
                        </div>
                        <div className="rag-comparison-stats">
                          <div className="rag-stat">
                            <span className="rag-stat-label">Chunks</span>
                            <span className="rag-stat-value">{ragComparison.qubo_rag?.chunks_selected || 0}</span>
                          </div>
                          <div className="rag-stat highlight">
                            <span className="rag-stat-label">Tokens</span>
                            <span className="rag-stat-value">{ragComparison.qubo_rag?.total_tokens || 0}</span>
                          </div>
                          <div className="rag-stat">
                            <span className="rag-stat-label">Avg Similarity</span>
                            <span className="rag-stat-value">{((ragComparison.qubo_rag?.avg_similarity || 0) * 100).toFixed(1)}%</span>
                          </div>
                          <div className="rag-stat highlight">
                            <span className="rag-stat-label">Avg RF Score</span>
                            <span className="rag-stat-value">{((ragComparison.qubo_rag?.avg_rf_score || 0) * 100).toFixed(1)}%</span>
                          </div>
                          <div className="rag-stat success">
                            <span className="rag-stat-label">Redundant Pairs</span>
                            <span className="rag-stat-value">{ragComparison.qubo_rag?.redundancy_pairs || 0}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Savings Summary */}
                    <div className="rag-comparison-summary">
                      <div className="rag-savings-item">
                        <span className="rag-savings-icon">💰</span>
                        <div className="rag-savings-content">
                          <span className="rag-savings-label">Token Savings</span>
                          <span className="rag-savings-value">
                            {(() => {
                              // Three outcomes, not two. The old branch collapsed
                              // "used more" into "Same usage", so a 1192-token
                              // INCREASE displayed as no change.
                              const abs = ragComparison.comparison?.token_savings?.absolute ?? 0;
                              const pct = ragComparison.comparison?.token_savings?.percentage ?? 0;
                              if (abs > 0) return (
                                <>
                                  <strong>{abs}</strong> tokens saved
                                  <span className="rag-savings-pct">({pct}%)</span>
                                </>
                              );
                              if (abs < 0) return (
                                <>
                                  <strong className="rag-savings-down">{Math.abs(abs)}</strong> more tokens
                                  <span className="rag-savings-pct">({Math.abs(pct)}%)</span>
                                </>
                              );
                              return <span className="rag-savings-neutral">Same usage</span>;
                            })()}
                          </span>
                        </div>
                      </div>
                      <div className="rag-savings-item">
                        <span className="rag-savings-icon">📈</span>
                        <div className="rag-savings-content">
                          {/* "Quality Improvement" presumed the answer. This card
                              reports whichever direction relevance actually moved,
                              using the backend's verdict rather than recomputing a
                              rosier one here — the old branch showed "Similar
                              quality" for any non-positive delta, so a 17% drop
                              read as no change. */}
                          <span className="rag-savings-label">Avg Relevance</span>
                          <span className="rag-savings-value">
                            {(() => {
                              const pct = ragComparison.comparison?.quality?.rf_score_improvement_pct ?? 0;
                              if (pct > 5) return <><strong>+{pct.toFixed(1)}%</strong> relevance</>;
                              if (pct < -5) return <><strong className="rag-savings-down">{pct.toFixed(1)}%</strong> relevance</>;
                              return <span className="rag-savings-neutral">Comparable</span>;
                            })()}
                          </span>
                        </div>
                      </div>
                      <div className="rag-savings-item">
                        <span className="rag-savings-icon">🔄</span>
                        <div className="rag-savings-content">
                          <span className="rag-savings-label">Redundancy</span>
                          <span className="rag-savings-value">
                            {(() => {
                              // "No redundancy" was shown whenever the reduction
                              // was not positive -- including when BOTH sides
                              // still carried redundant pairs. Report the counts.
                              const r = ragComparison.comparison?.redundancy;
                              const cut = r?.reduction ?? 0;
                              const before = r?.traditional_redundant_pairs ?? 0;
                              const after = r?.qubo_redundant_pairs ?? 0;
                              if (cut > 0) return (
                                <><strong>-{cut}</strong> redundant pairs removed</>
                              );
                              if (before === 0 && after === 0) return (
                                <span className="rag-savings-neutral">None in either</span>
                              );
                              if (cut === 0) return (
                                <span className="rag-savings-neutral">{after} pair{after === 1 ? '' : 's'}, unchanged</span>
                              );
                              return (
                                <><strong className="rag-savings-down">+{Math.abs(cut)}</strong> more redundant pairs</>
                              );
                            })()}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div
                      className={`rag-comparison-verdict is-${ragComparison.comparison?.verdict_kind || 'neutral'}`}
                    >
                      {ragComparison.comparison?.overall_verdict || 'Analysis complete'}
                    </div>
                    {/* When average relevance drops, say why in plain terms rather
                        than letting a green banner imply it didn't happen. */}
                    {ragComparison.comparison?.quality?.note && (
                      <p className="rag-verdict-note">
                        {ragComparison.comparison.quality.note}
                      </p>
                    )}
                  </div>
                )}
              </>
            )}

            <form className="gpt-composer" onSubmit={handleComposerSubmit}>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                style={{ display: 'none' }}
                onChange={handleFileSelect}
                accept=".csv,.xlsx,.xls,.json,.txt,.pdf,.doc,.docx"
              />
              <div className="composer-field">
                <button
                  type="button"
                  className="composer-btn"
                  onClick={handleAttachClick}
                  disabled={uploadingFiles || isGenerating}
                  aria-label="Attach file"
                  title="Attach file"
                >
                  {uploadingFiles ? (
                    <span className="composer-btn-icon">⏳</span>
                  ) : (
                    <svg className="composer-btn-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="20" height="20">
                      <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                </button>
                {attachedFiles.length > 0 && (
                  <div className="composer-attached-files">
                    {attachedFiles.map((file) => (
                      <div key={file.id} className="composer-attached-file-item">
                        <div className="composer-attached-file-icon">{getFileIcon(file.name, file.type)}</div>
                        <div className="composer-attached-file-content">
                          <span className="composer-attached-file-name" title={file.name}>{file.name}</span>
                        </div>
                        <button
                          type="button"
                          className="composer-attached-file-remove"
                          onClick={() => handleRemoveFile(file.id)}
                          aria-label="Remove file"
                        >
                          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="14" height="14">
                            <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <textarea
                  ref={composerTextareaRef}
                  className="composer-textarea"
                  placeholder="Ask me anything..."
                  value={composerValue}
                  onChange={handleComposerInput}
                  disabled={isGenerating || uploadingFiles}
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleComposerSubmit(e);
                    }
                  }}
                />
              </div>
              <div className="composer-controls">
                <button
                  type="submit"
                  className="composer-send"
                  disabled={isGenerating || uploadingFiles || (!composerValue.trim() && attachedFiles.length === 0)}
                >
                  {isGenerating ? (
                    <svg className="composer-spinner" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" width="20" height="20">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" strokeDasharray="60" strokeLinecap="round">
                        <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
                      </circle>
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="20" height="20">
                      <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                </button>
              </div>
            </form>
            {isGenerating && <div className="assistant-hint">Synergy is thinking…</div>}
          </main>
        </div>
      )}

      {/* Delete Chat Confirmation Modal */}
      {deleteConfirmThread && (
        <div className="delete-modal-overlay" onClick={handleDeleteThreadCancel}>
          <div className="delete-modal" onClick={(e) => e.stopPropagation()}>
            <h2>Delete chat?</h2>
            <p className="delete-modal-message">
              This will delete <strong>{deleteConfirmThread.title}</strong>.
            </p>
            <div className="delete-modal-actions">
              <button type="button" className="delete-modal-cancel" onClick={handleDeleteThreadCancel}>
                Cancel
              </button>
              <button type="button" className="delete-modal-confirm" onClick={handleDeleteThreadConfirm}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Document Confirmation Modal */}
      {deleteConfirmDoc && (
        <div className="delete-modal-overlay" onClick={handleDeleteRagDocCancel}>
          <div className="delete-modal" onClick={(e) => e.stopPropagation()}>
            <h2>Delete document?</h2>
            <p className="delete-modal-message">
              This will delete <strong>{deleteConfirmDoc.filename}</strong> and all its chunks.
            </p>
            <div className="delete-modal-actions">
              <button type="button" className="delete-modal-cancel" onClick={handleDeleteRagDocCancel}>
                Cancel
              </button>
              <button type="button" className="delete-modal-confirm" onClick={handleDeleteRagDocConfirm}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Workspace;
