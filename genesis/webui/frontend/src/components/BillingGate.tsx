import { useEffect, useState, type ReactNode } from 'react';
import { useAuth } from '@clerk/clerk-react';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

interface BillingStatus {
  subscribed: boolean;
  status: string;
  plan: string;
  currentPeriodEnd: string | null;
}

export function BillingGate({ children }: { children: ReactNode }) {
  const { getToken } = useAuth();
  const [status, setStatus] = useState<BillingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [redirecting, setRedirecting] = useState(false);

  useEffect(() => {
    async function checkSubscription() {
      try {
        const token = await getToken();
        const res = await fetch(`${API_BASE}/billing/status`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data: BillingStatus = await res.json();
          setStatus(data);
        }
      } catch (err) {
        console.error('Failed to check billing status:', err);
      } finally {
        setLoading(false);
      }
    }
    checkSubscription();
  }, [getToken]);

  async function handleSubscribe() {
    setRedirecting(true);
    try {
      const token = await getToken();
      const res = await fetch(`${API_BASE}/billing/checkout`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      }
    } catch (err) {
      console.error('Failed to create checkout session:', err);
      setRedirecting(false);
    }
  }

  async function handleManageBilling() {
    try {
      const token = await getToken();
      const res = await fetch(`${API_BASE}/billing/portal`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      }
    } catch (err) {
      console.error('Failed to open billing portal:', err);
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-neutral-950">
        <div className="text-neutral-400">Loading...</div>
      </div>
    );
  }

  if (status?.subscribed) {
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950">
      <div className="flex flex-col items-center gap-6 rounded-xl border border-neutral-800 bg-neutral-900 p-12 shadow-2xl">
        <img src="/genesis-logo.png" alt="Genesis" className="h-20 w-20 rounded-full" />
        <h1 className="text-2xl font-bold text-white">Subscription Required</h1>
        <p className="max-w-sm text-center text-neutral-400">
          A Genesis Pro subscription is required to access evolution experiments and analytics.
        </p>
        <button
          onClick={handleSubscribe}
          disabled={redirecting}
          className="rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-500 disabled:opacity-50"
        >
          {redirecting ? 'Redirecting...' : 'Subscribe to Genesis Pro'}
        </button>
        {status?.status === 'canceled' && (
          <button
            onClick={handleManageBilling}
            className="text-sm text-neutral-500 underline transition hover:text-neutral-300"
          >
            Manage existing subscription
          </button>
        )}
      </div>
    </div>
  );
}
