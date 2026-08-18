import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient.js';
import './ResetPassword.css';

const ResetPassword = () => {
  const navigate = useNavigate();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState({ type: null, message: '' });
  const [sessionReady, setSessionReady] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function loadSession() {
      const { data } = await supabase.auth.getSession();
      if (!isMounted) return;

      if (!data.session) {
        setStatus({
          type: 'error',
          message:
            'The reset link has expired or is invalid. Please request a new password reset email.',
        });
      } else {
        setSessionReady(true);
      }
    }

    loadSession();

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session) {
        setSessionReady(true);
      }
    });

    return () => {
      isMounted = false;
      listener.subscription.unsubscribe();
    };
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (submitting || !sessionReady) return;

    if (password.length < 8) {
      setStatus({ type: 'error', message: 'Password must be at least 8 characters long.' });
      return;
    }

    if (password !== confirmPassword) {
      setStatus({ type: 'error', message: 'Passwords do not match.' });
      return;
    }

    setSubmitting(true);
    setStatus({ type: null, message: '' });

    const { error } = await supabase.auth.updateUser({ password });

    if (error) {
      setStatus({ type: 'error', message: error.message });
      setSubmitting(false);
    } else {
      setStatus({ type: 'success', message: 'Password updated successfully. Redirecting to sign in…' });
      setTimeout(() => navigate('/login'), 1500);
    }
  };

  return (
    <div className="reset-page">
      <div className="reset-wrapper">
        <div className="reset-panel">
          <div className="reset-brand">
            <div className="reset-logo-mark" aria-hidden="true">
              <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="6" width="28" height="28" rx="10" fill="url(#synergy-reset-gradient)" />
                <circle cx="20" cy="20" r="6.5" fill="none" stroke="rgba(255,255,255,0.75)" strokeWidth="1.8" />
                <circle cx="20" cy="20" r="3.2" fill="rgba(255,255,255,0.92)" />
                <path
                  d="M13 15.5c3.4-3.9 10.6-4.2 14.1-.2"
                  fill="none"
                  stroke="rgba(152,215,255,0.9)"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                />
                <path
                  d="M26.6 24.6c-3.4 3.8-10.5 4-14-.1"
                  fill="none"
                  stroke="rgba(255,255,255,0.55)"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                />
                <defs>
                  <linearGradient id="synergy-reset-gradient" x1="6" y1="6" x2="34" y2="34" gradientUnits="userSpaceOnUse">
                    <stop offset="0" stopColor="#a855f7" />
                    <stop offset="0.45" stopColor="#6366f1" />
                    <stop offset="1" stopColor="#22d3ee" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <span className="reset-logo-text">Synergy</span>
          </div>

          <div className="reset-copy">
            <p className="reset-tagline">Update credentials</p>
            <h1 className="reset-heading">Set a new password</h1>
            <p className="reset-subtext">
              Choose a strong password to secure your Synergy account and continue working with the quantum engine.
            </p>
          </div>

          {status.message && (
            <div className={`form-alert ${status.type === 'error' ? 'error' : 'success'}`}>{status.message}</div>
          )}

          <form className="reset-form" onSubmit={handleSubmit}>
            <label className="reset-field-group">
              <span className="reset-field-label">New password</span>
              <input
                type="password"
                name="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Enter new password"
                required
              />
            </label>
            <label className="reset-field-group">
              <span className="reset-field-label">Confirm password</span>
              <input
                type="password"
                name="confirmPassword"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                placeholder="Re-enter new password"
                required
              />
            </label>

            <button className="reset-primary-btn" type="submit" disabled={submitting || !sessionReady}>
              {sessionReady ? 'Update password' : 'Validating link…'}
            </button>
          </form>

          <div className="reset-footer">
            <Link to="/login">Return to sign in</Link>
          </div>
        </div>
        <div className="reset-visual" aria-hidden="true" />
      </div>
    </div>
  );
};

export default ResetPassword;


