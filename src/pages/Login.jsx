import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient.js';
import './Login.css';

const Login = () => {
  const navigate = useNavigate();
  const [formValues, setFormValues] = useState({ email: '', password: '' });
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState({ type: null, message: '' });

  const handleGoogleSignIn = async () => {
    setStatus({ type: null, message: '' });
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/dashboard`,
      },
    });

    if (error) {
      setStatus({ type: 'error', message: error.message });
    }
  };

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (submitting) return;

    setSubmitting(true);
    setStatus({ type: null, message: '' });

    const { error } = await supabase.auth.signInWithPassword({
      email: formValues.email.trim(),
      password: formValues.password,
    });

    if (error) {
      setStatus({ type: 'error', message: error.message });
      setSubmitting(false);
      return;
    }

    setStatus({ type: 'success', message: 'Welcome back! Redirecting…' });
    setTimeout(() => {
      navigate('/dashboard');
    }, 1200);
  };

  return (
    <div className="login-page">
      <div className="login-wrapper">
        <div className="login-panel">
          <div className="login-brand">
            <div className="login-logo-mark" aria-hidden="true">
              <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="6" width="28" height="28" rx="10" fill="url(#synergy-login-gradient)" />
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
                  <linearGradient id="synergy-login-gradient" x1="6" y1="6" x2="34" y2="34" gradientUnits="userSpaceOnUse">
                    <stop offset="0" stopColor="#4b5df6" />
                    <stop offset="0.4" stopColor="#3f83f8" />
                    <stop offset="1" stopColor="#22d3ee" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <span className="login-logo-text">Synergy</span>
          </div>

          <div className="login-copy">
            <p className="login-tagline">Welcome back</p>
            <h1 className="login-heading">Access your workspace</h1>
            <p className="login-subtext">Enter your credentials to continue orchestration.</p>
          </div>

          {status.message && (
            <div className={`form-alert ${status.type === 'error' ? 'error' : 'success'}`}>{status.message}</div>
          )}

          <form className="login-form" onSubmit={handleSubmit}>
            <label className="login-field-group">
              <span className="login-field-label">E-mail</span>
              <input
                name="email"
                value={formValues.email}
                onChange={handleChange}
                type="email"
                placeholder="example@email.com"
                required
              />
            </label>
            <label className="login-field-group">
              <span className="login-field-label">Password</span>
              <input
                name="password"
                value={formValues.password}
                onChange={handleChange}
                type="password"
                placeholder="********"
                required
                minLength={8}
              />
              <div className="login-meta">
                <span>Must be at least 8 characters.</span>
                <Link to="/forgot-password" className="login-link-button">
                  Forgot password?
                </Link>
              </div>
            </label>

            <button type="submit" className="login-primary-btn" disabled={submitting}>
              Sign in
            </button>
          </form>

          <div className="login-divider">
            <span>or continue with</span>
          </div>

          <div className="login-social">
            <button type="button" className="oauth-btn google" onClick={handleGoogleSignIn}>
              <span className="oauth-icon">
                <svg width="20" height="20" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M17.64 9.20451C17.64 8.56639 17.5836 7.95275 17.4764 7.36365H9V10.844H13.8436C13.635 11.9685 13.0009 12.9236 12.0459 13.5617V15.819H14.9564C16.6582 14.2528 17.64 11.9456 17.64 9.20451Z"
                    fill="#4285F4"
                  />
                  <path
                    d="M9 18C11.43 18 13.4673 17.1941 14.9564 15.819L12.0459 13.5617C11.24 14.1017 10.2118 14.4227 9 14.4227C6.65591 14.4227 4.67182 12.8373 3.96409 10.71H0.95636V13.0418C2.43818 15.9832 5.48182 18 9 18Z"
                    fill="#34A853"
                  />
                  <path
                    d="M3.96409 10.71C3.78227 10.17 3.68182 9.59455 3.68182 9C3.68182 8.40545 3.78227 7.83 3.96409 7.29V4.95818H0.956364C0.347727 6.17364 0 7.54727 0 9C0 10.4527 0.347727 11.8264 0.956364 13.0418L3.96409 10.71Z"
                    fill="#FBBC05"
                  />
                  <path
                    d="M9 3.57727C10.3186 3.57727 11.4982 4.03273 12.4091 4.905L15.0218 2.29227C13.4632 0.844091 11.426 0 9 0C5.48182 0 2.43818 2.01636 0.956364 4.95818L3.96409 7.29C4.67182 5.16273 6.65591 3.57727 9 3.57727Z"
                    fill="#EA4335"
                  />
                </svg>
              </span>
              Continue with Google
            </button>
          </div>

          <div className="login-footer">
            New to Synergy? <Link to="/">Create account</Link>
          </div>
        </div>
        <div className="login-visual" aria-hidden="true" />
      </div>
    </div>
  );
};

export default Login;

