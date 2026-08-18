import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient.js';
import './Dashboard.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    let isMounted = true;

    const bootstrap = async () => {
      const { data } = await supabase.auth.getSession();
      if (!isMounted) {
        return;
      }

      if (!data.session) {
        setLoading(false);
        return;
      } else {
        setSession(data.session);
        setLoading(false);
      }
    };

    bootstrap();

    const { data: listener } = supabase.auth.onAuthStateChange((_event, currentSession) => {
      if (!isMounted) return;
      
      if (!currentSession) {
        setSession(null);
      } else {
        setSession(currentSession);
      }
    });

    return () => {
      isMounted = false;
      listener?.subscription?.unsubscribe();
    };
  }, []);

  // Separate effect to handle navigation after loading and when no session
  useEffect(() => {
    if (!loading && !session) {
      navigate('/login', { replace: true });
    }
  }, [loading, session, navigate]);

  const profileName = useMemo(() => {
    if (!session) {
      return '';
    }
    const metadataName = session.user.user_metadata?.full_name;
    if (metadataName && metadataName.trim().length > 0) {
      return metadataName.trim();
    }
    return session.user.email;
  }, [session]);

  const cards = useMemo(
    () => [
      {
        id: 'welcome',
        eyebrow: 'Intelligent Document Q&A',
        title: 'Welcome to Synergy',
        description:
          'Upload your documents and ask questions naturally. Synergy uses advanced optimization to select the most relevant context—delivering precise answers while using fewer tokens and saving you money.',
        badge: 'Token-efficient',
        accent: 'var(--card-accent-blue)',
        cta: 'Next',
      },
      {
        id: 'optimization',
        eyebrow: 'QUBO + Random Forest',
        title: 'Smarter context selection',
        description:
          'Our Random Forest model scores each document chunk for relevance, while QUBO optimization eliminates redundancy and selects the ideal subset under your token budget. The result: better answers, less noise.',
        badge: 'Optimized retrieval',
        accent: 'var(--card-accent-purple)',
        cta: 'Next',
      },
      {
        id: 'workspace',
        eyebrow: 'Your Workspace',
        title: 'Start asking questions',
        description:
          'Bring your own LLM API key, upload files, and chat. Synergy continuously learns from your interactions to improve retrieval over time—no model fine-tuning required.',
        badge: 'Get started',
        accent: 'var(--card-accent-cyan)',
        cta: 'Enter workspace',
      },
    ],
    [],
  );

  const handleNext = () => {
    setActiveIndex((current) => {
      const nextIndex = current + 1;
      if (nextIndex >= cards.length) {
        navigate('/workspace');
        return current;
      }
      return nextIndex;
    });
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate('/login', { replace: true });
  };

  if (loading) {
    return (
      <div className="dashboard-page">
        <div className="dashboard-shell loading-state">
          <div className="loading-spinner" aria-hidden="true" />
          <span>Preparing your workspace&hellip;</span>
        </div>
      </div>
    );
  }

  const activeCard = cards[activeIndex];

  return (
    <div className="dashboard-page">
      <div className="dashboard-shell">
        <header className="dashboard-header">
          <div className="brand-mark">
            <div className="logo-glyph" aria-hidden="true">
              <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="6" width="28" height="28" rx="10" fill="url(#synergy-dashboard-gradient)" />
                <circle cx="20" cy="20" r="6.5" fill="none" stroke="rgba(255,255,255,0.72)" strokeWidth="1.6" />
                <circle cx="20" cy="20" r="3.1" fill="rgba(255,255,255,0.96)" />
                <defs>
                  <linearGradient id="synergy-dashboard-gradient" x1="6" y1="6" x2="34" y2="34">
                    <stop offset="0" stopColor="#4f46ef" />
                    <stop offset="0.5" stopColor="#2563eb" />
                    <stop offset="1" stopColor="#0ea5e9" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div className="brand-copy">
              <span className="brand-name">Synergy</span>
              <span className="brand-tagline">Optimized document intelligence</span>
            </div>
          </div>
          <div className="user-pill">
            <div className="avatar" aria-hidden="true">
              {profileName?.charAt(0)?.toUpperCase() || 'S'}
            </div>
            <div className="user-meta">
              <span className="user-label">Signed in as</span>
              <span className="user-name">{profileName}</span>
            </div>
            <button className="ghost-button" type="button" onClick={handleSignOut}>
              Sign out
            </button>
          </div>
        </header>

        <main className="dashboard-main">
          <section className="card-stack" style={{ '--card-accent': activeCard.accent }}>
            <article className="spotlight-card">
              <div className="card-eyebrow">{activeCard.eyebrow}</div>
              <h1 className="card-title">{activeCard.title}</h1>
              <p className="card-description">{activeCard.description}</p>
              <div className="card-footer">
                <span className="accent-pill">{activeCard.badge}</span>
                <button className="primary-ghost" type="button" onClick={handleNext}>
                  {activeCard.cta}
                </button>
              </div>
            </article>
            <footer className="card-progress" aria-label="Onboarding progress">
              {cards.map((card, index) => (
                <button
                  key={card.id}
                  type="button"
                  className={`progress-dot ${index === activeIndex ? 'active' : ''}`}
                  onClick={() => setActiveIndex(index)}
                  aria-label={`Show ${card.title}`}
                  aria-current={index === activeIndex}
                />
              ))}
            </footer>
          </section>

          <aside className="info-panel">
            <div className="panel-background" aria-hidden="true" />
            <div className="panel-copy">
              <h2>Why Synergy outperforms traditional RAG</h2>
              <p>
                Standard retrieval stuffs your context with redundant chunks. Synergy's QUBO-based optimization 
                selects only what matters—cutting token costs by up to 40% while improving answer quality.
              </p>
              <ul>
                <li>RF-scored chunk relevance with SHAP explainability</li>
                <li>QUBO optimization eliminates redundant context</li>
                <li>Continuous learning from your feedback</li>
              </ul>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;


