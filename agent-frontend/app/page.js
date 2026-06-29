'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Search, BrainCircuit, Activity, Wrench, ChevronRight, Moon, Sun, Cpu } from 'lucide-react';
import { startResearch, checkSystemHealth, getMemoryCount, getMcpTools } from '@/lib/api';
import { useTheme } from '@/components/ThemeProvider';

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      className="btn-ghost p-2 rounded-lg"
      aria-label="Toggle dark mode"
    >
      {theme === 'dark'
        ? <Sun className="w-4 h-4" style={{ color: '#FF5B21' }} />
        : <Moon className="w-4 h-4" />}
    </button>
  );
}

export default function Home() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [depth, setDepth] = useState('standard');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const [health, setHealth] = useState({ backend: false, loading: true });
  const [memoryCount, setMemoryCount] = useState(0);
  const [mcpTools, setMcpTools] = useState([]);

  console.log("API BASE URL:", process.env.NEXT_PUBLIC_API_URL)

  useEffect(() => {
    async function fetchSystemStatus() {
      try {
        const h = await checkSystemHealth();
        setHealth({ backend: h.status === 'healthy', loading: false });
        if (h.status === 'healthy') {
          try { const m = await getMemoryCount(); setMemoryCount(m.count || 0); } catch (_) { }
          try { const t = await getMcpTools(); setMcpTools(t.tools || []); } catch (_) { }
        }
      } catch (_) {
        setHealth({ backend: false, loading: false });
      }
    }
    fetchSystemStatus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setIsSubmitting(true);
    setError('');
    try {
      const res = await startResearch(query, depth);
      router.push(`/task/${res.request_id}`);
    } catch (err) {
      setError(err.message || 'Failed to start research. Is the backend running?');
      setIsSubmitting(false);
    }
  };

  const DEPTH_OPTS = [
    { key: 'quick', label: 'Quick', desc: '~30s' },
    { key: 'standard', label: 'Standard', desc: '~2min' },
    { key: 'deep', label: 'Deep', desc: '~5min' },
  ];

  return (
    <main className="min-h-screen bg-canvas flex flex-col">

      {/* ── Top Nav ─────────────────────────────────── */}
      <nav style={{ backgroundColor: 'var(--bg-surface)', borderBottom: '1px solid var(--bd)' }}
        className="sticky top-0 z-40 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5" style={{ color: '#0073FE' }} />
          <span className="font-semibold text-sm" style={{ color: 'var(--tx)' }}>Ingenious Agentic</span>
        </div>
        <ThemeToggle />
      </nav>

      {/* ── Hero ────────────────────────────────────── */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 pb-24 pt-16">

        <div className="w-full max-w-2xl animate-fade-up">

          {/* Eyebrow tag */}
          <div className="flex justify-center mb-6">
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full"
              style={{ backgroundColor: 'rgba(0,115,254,0.1)', color: '#0073FE' }}>
              <BrainCircuit className="w-3.5 h-3.5" />
              Multi-Agent Research Engine
            </span>
          </div>

          {/* Headline */}
          <h1 className="text-center text-5xl font-bold tracking-tight mb-4 leading-tight"
            style={{ color: 'var(--tx)' }}>
            Research anything.{' '}
            <span style={{ color: '#0073FE' }}>Instantly.</span>
          </h1>
          <p className="text-center text-lg mb-10" style={{ color: 'var(--tx-2)' }}>
            Autonomous agents plan, search the web, analyse sources,<br className="hidden md:block" />
            and write a comprehensive report — in minutes.
          </p>

          {/* Main Search Card */}
          <div className="card p-2" style={{ boxShadow: '0 8px 40px rgba(0,0,0,0.08)' }}>
            <form onSubmit={handleSubmit}>
              {error && (
                <div className="mx-2 mt-2 mb-1 px-4 py-3 rounded-xl text-sm"
                  style={{ backgroundColor: 'rgba(255,91,33,0.08)', color: '#FF5B21', border: '1px solid rgba(255,91,33,0.18)' }}>
                  {error}
                </div>
              )}

              {/* Input Row */}
              <div className="flex gap-2 p-1">
                <div className="flex-1 relative">
                  <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5"
                    style={{ color: 'var(--tx-3)' }} />
                  <input
                    type="text"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder="e.g. How does AlphaFold 3 work?"
                    disabled={isSubmitting}
                    className="input-base w-full py-3.5 pl-12 pr-4 text-base"
                    style={{ border: 'none', boxShadow: 'none', borderRadius: '10px' }}
                    autoFocus
                  />
                </div>
                <button
                  type="submit"
                  disabled={isSubmitting || !query.trim()}
                  className="btn-blue rounded-xl px-6 py-3 font-semibold flex items-center gap-2 disabled:opacity-50"
                >
                  {isSubmitting ? 'Starting…' : 'Run Agents'}
                  {!isSubmitting && <ChevronRight className="w-4 h-4" />}
                </button>
              </div>

              {/* Depth Selector */}
              <div className="flex items-center gap-3 px-3 pb-2 pt-1">
                <span className="text-xs font-medium" style={{ color: 'var(--tx-3)' }}>Depth:</span>
                <div className="flex gap-1 p-1 rounded-lg" style={{ backgroundColor: 'var(--bg-muted)' }}>
                  {DEPTH_OPTS.map(d => (
                    <button
                      key={d.key}
                      type="button"
                      onClick={() => setDepth(d.key)}
                      disabled={isSubmitting}
                      className="px-3 py-1 rounded-md text-xs font-medium transition-all flex items-center gap-1"
                      style={{
                        backgroundColor: depth === d.key ? 'var(--bg-surface)' : 'transparent',
                        color: depth === d.key ? 'var(--tx)' : 'var(--tx-3)',
                        boxShadow: depth === d.key ? '0 1px 4px rgba(0,0,0,0.08)' : 'none',
                      }}
                    >
                      {d.label}
                      <span style={{ color: 'var(--tx-3)', fontSize: '10px' }}>{d.desc}</span>
                    </button>
                  ))}
                </div>
              </div>
            </form>
          </div>

          {/* System Status Row */}
          <div className="mt-8 grid grid-cols-3 gap-3">

            {/* API */}
            <div className="card p-4 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: health.loading ? 'var(--bg-muted)' : health.backend ? 'rgba(63,167,89,0.12)' : 'rgba(255,91,33,0.1)' }}>
                <Activity className="w-4 h-4"
                  style={{ color: health.loading ? 'var(--tx-3)' : health.backend ? '#3FA759' : '#FF5B21' }} />
              </div>
              <div>
                <p className="text-xs font-semibold" style={{ color: 'var(--tx)' }}>API</p>
                <p className="text-xs" style={{ color: 'var(--tx-3)' }}>
                  {health.loading ? 'Checking…' : health.backend ? 'Online' : 'Offline'}
                </p>
              </div>
            </div>

            {/* Memory */}
            <div className="card p-4 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: 'rgba(0,115,254,0.1)' }}>
                <BrainCircuit className="w-4 h-4" style={{ color: '#0073FE' }} />
              </div>
              <div>
                <p className="text-xs font-semibold" style={{ color: 'var(--tx)' }}>Memory</p>
                <p className="text-xs" style={{ color: 'var(--tx-3)' }}>{memoryCount} reports</p>
              </div>
            </div>

            {/* MCP */}
            <div className="card p-4 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: 'rgba(255,91,33,0.1)' }}>
                <Wrench className="w-4 h-4" style={{ color: '#FF5B21' }} />
              </div>
              <div>
                <p className="text-xs font-semibold" style={{ color: 'var(--tx)' }}>MCP</p>
                <p className="text-xs" style={{ color: 'var(--tx-3)' }}>{mcpTools.length} tools</p>
              </div>
            </div>

          </div>
        </div>
      </div>
    </main>
  );
}
