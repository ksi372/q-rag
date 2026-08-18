import React, { useEffect, useState } from 'react';
import VaporizeTextCycle, { Tag } from './VaporizeText.tsx';
import './IntroSplash.css';

/**
 * Boot splash — the particle-vaporize wordmark carried over from the synergy 2.0
 * shell, wrapped in a HUD frame (concentric rings, scanline, status readout) so
 * it reads as a system coming online rather than a plain title card.
 *
 * Shows once per browser session: a boot animation on every route change gets
 * old fast, and sessionStorage clears on tab close so it still greets a fresh visit.
 */
const SESSION_KEY = 'synergy.introShown';
const HOLD_MS = 2600;
const FADE_MS = 900;

export function IntroSplash({ onDone }) {
  const [visible, setVisible] = useState(() => {
    try {
      return sessionStorage.getItem(SESSION_KEY) !== '1';
    } catch {
      return true;
    }
  });
  const [fading, setFading] = useState(false);

  useEffect(() => {
    if (!visible) {
      onDone?.();
      return undefined;
    }

    // Respect a reduced-motion preference: mark it seen and get out of the way.
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      try {
        sessionStorage.setItem(SESSION_KEY, '1');
      } catch {
        /* private mode — just don't persist */
      }
      setVisible(false);
      onDone?.();
      return undefined;
    }

    const fadeTimer = setTimeout(() => setFading(true), HOLD_MS);
    const doneTimer = setTimeout(() => {
      try {
        sessionStorage.setItem(SESSION_KEY, '1');
      } catch {
        /* ignore */
      }
      setVisible(false);
      onDone?.();
    }, HOLD_MS + FADE_MS);

    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(doneTimer);
    };
  }, [visible, onDone]);

  if (!visible) return null;

  return (
    <div
      className={`intro-splash${fading ? ' is-fading' : ''}`}
      role="status"
      aria-label="Synergy is starting up"
    >
      <div className="intro-grid" aria-hidden="true" />
      <div className="intro-scan" aria-hidden="true" />

      <div className="intro-rings" aria-hidden="true">
        <span className="intro-ring intro-ring-outer" />
        <span className="intro-ring intro-ring-mid" />
        <span className="intro-ring intro-ring-inner" />
      </div>

      <div className="intro-wordmark">
        <VaporizeTextCycle
          texts={['SYNERGY']}
          font={{
            fontFamily: "'Inter', system-ui, sans-serif",
            fontSize: '78px',
            fontWeight: 700,
          }}
          color="rgb(34, 211, 238)"
          spread={4}
          density={6}
          animation={{ vaporizeDuration: 2.4, fadeInDuration: 1, waitDuration: 0.9 }}
          direction="left-to-right"
          alignment="center"
          tag={Tag.H1}
        />
      </div>

      <div className="intro-readout">
        <span className="intro-dot" aria-hidden="true" />
        <span>QUBO retrieval optimizer — initializing</span>
      </div>
    </div>
  );
}

export default IntroSplash;
