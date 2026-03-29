import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SignedIn, SignedOut, SignInButton, UserButton } from '@clerk/clerk-react';
import { GenesisProvider } from './context/GenesisContext';
import GenesisLayout from './components/GenesisLayout';
import CommandMenu from './components/CommandMenu';
import { ErrorBoundary } from './components/ErrorBoundary';
import { BillingGate } from './components/BillingGate';

function SignInPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950">
      <div className="flex flex-col items-center gap-6 rounded-xl border border-neutral-800 bg-neutral-900 p-12 shadow-2xl">
        <img src="/genesis-logo.png" alt="Genesis" className="h-20 w-20 rounded-full" />
        <h1 className="text-2xl font-bold text-white">Genesis</h1>
        <p className="text-neutral-400">Sign in to access your evolution experiments</p>
        <SignInButton mode="modal">
          <button className="rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-500">
            Continue with Google
          </button>
        </SignInButton>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <SignedOut>
        <SignInPage />
      </SignedOut>
      <SignedIn>
        <BillingGate>
          <GenesisProvider>
            <div className="relative">
              <div className="absolute right-4 top-4 z-50">
                <UserButton afterSignOutUrl="/" />
              </div>
              <CommandMenu />
              <BrowserRouter>
                <Routes>
                  <Route path="/" element={<GenesisLayout />} />
                </Routes>
              </BrowserRouter>
            </div>
          </GenesisProvider>
        </BillingGate>
      </SignedIn>
    </ErrorBoundary>
  );
}
