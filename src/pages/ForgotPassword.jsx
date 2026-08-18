import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient.js';
import './ForgotPassword.css';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState({ type: null, message: '' });

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (submitting) return;

    setSubmitting(true);
    setStatus({ type: null, message: '' });

    const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
      redirectTo: `${window.location.origin}/reset-password`,
    });

    if (error) {
      setStatus({ type: 'error', message: error.message });
    } else {
      setStatus({
        type: 'success',
        message: 'Check your email for a secure link to finish resetting your password.',
      });
    }

    setSubmitting(false);
  };

  return (
    <div className="forgot-page">
      <div className="forgot-wrapper">
        <div className="forgot-panel">
          <div className="forgot-brand">
            <div className="forgot-logo-mark" aria-hidden="true">
              <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="6" width="28" height="28" rx="10" fill="url(#synergy-forgot-gradient)" />
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
                  <linearGradient id="synergy-forgot-gradient" x1="6" y1="6" x2="34" y2="34" gradientUnits="userSpaceOnUse">
                    <stop offset="0" stopColor="#6366f1" />
                    <stop offset="0.45" stopColor="#2563eb" />
                    <stop offset="1" stopColor="#22d3ee" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <span className="forgot-logo-text">Synergy</span>
          </div>

          <div className="forgot-copy">
            <p className="forgot-tagline">Reset access</p>
            <h1 className="forgot-heading">Forget your password?</h1>
            <p className="forgot-subtext">
              Enter your account email and we’ll send a secure link to reset your credentials.
            </p>
          </div>

          {status.message && (
            <div className={`form-alert ${status.type === 'error' ? 'error' : 'success'}`}>{status.message}</div>
          )}

          <form className="forgot-form" onSubmit={handleSubmit}>
            <label className="forgot-field-group">
              <span className="forgot-field-label">E-mail</span>
              <input
                name="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                type="email"
                placeholder="example@email.com"
                required
              />
            </label>

            <button type="submit" className="forgot-primary-btn" disabled={submitting}>
              Send reset link
            </button>
          </form>

          <div className="forgot-footer">
            Remembered your password? <Link to="/login">Back to sign in</Link>
          </div>
        </div>
        <div className="forgot-visual" aria-hidden="true" />
      </div>
    </div>
  );
};

export default ForgotPassword;

