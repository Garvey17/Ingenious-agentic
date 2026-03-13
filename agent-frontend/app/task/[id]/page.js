'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import ReactMarkdown from 'react-markdown';
import {
    ArrowLeft, Clock, CheckCircle2, Circle, Loader2,
    ExternalLink, BrainCircuit, BookOpen, Moon, Sun, Cpu
} from 'lucide-react';
import { getResearchStatus, getResearchState } from '@/lib/api';
import { useTheme } from '@/components/ThemeProvider';

const AGENTS = [
    { key: 'planner', label: 'Planner', desc: 'Analysing the goal & building a search strategy', color: '#0073FE' },
    { key: 'researcher', label: 'Researcher', desc: 'Searching the web and collecting sources', color: '#FF5B21' },
    { key: 'analyst', label: 'Analyst', desc: 'Extracting facts and deriving insights', color: '#0073FE' },
    { key: 'writer', label: 'Writer', desc: 'Authoring the structured research report', color: '#FF5B21' },
    { key: 'critic', label: 'Critic', desc: 'Reviewing quality and scoring the report', color: '#3FA759' },
];

const POLL_MS = 2500;

function elapsed(start) {
    const s = Math.floor((Date.now() - start) / 1000);
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

function StatusBadge({ status }) {
    const cfg = {
        pending: { bg: 'rgba(255,91,33,0.1)', color: '#FF5B21' },
        in_progress: { bg: 'rgba(0,115,254,0.1)', color: '#0073FE' },
        running: { bg: 'rgba(0,115,254,0.1)', color: '#0073FE' },
        completed: { bg: 'rgba(63,167,89,0.12)', color: '#3FA759' },
        approved: { bg: 'rgba(63,167,89,0.12)', color: '#3FA759' },
        failed: { bg: 'rgba(255,91,33,0.1)', color: '#FF5B21' },
    };
    const c = cfg[status] || cfg.pending;
    return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold capitalize"
            style={{ backgroundColor: c.bg, color: c.color }}>
            {(status === 'running' || status === 'in_progress') && <Loader2 className="w-3 h-3 animate-spin" />}
            {(status === 'completed' || status === 'approved') && <CheckCircle2 className="w-3 h-3" />}
            {status?.replace('_', ' ')}
        </span>
    );
}

function AgentStep({ agent, isActive, isDone }) {
    return (
        <div className="flex items-start gap-3 p-3 rounded-xl transition-all duration-300"
            style={{
                backgroundColor: isActive ? `${agent.color}0D` : 'transparent',
                border: isActive ? `1px solid ${agent.color}30` : '1px solid transparent',
            }}>
            <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-all"
                style={{
                    backgroundColor: isDone ? '#3FA759' : isActive ? agent.color : 'var(--bg-muted)',
                    border: !isDone && !isActive ? '1.5px solid var(--bd)' : 'none',
                }}>
                {isDone ? <CheckCircle2 className="w-3.5 h-3.5 text-white" /> :
                    isActive ? <Loader2 className="w-3.5 h-3.5 text-white animate-spin" /> :
                        <Circle className="w-3.5 h-3.5" style={{ color: 'var(--tx-3)' }} />}
            </div>
            <div>
                <p className="text-sm font-semibold"
                    style={{ color: isActive ? agent.color : isDone ? '#3FA759' : 'var(--tx-2)' }}>
                    {agent.label}
                </p>
                <p className="text-xs mt-0.5" style={{ color: 'var(--tx-3)' }}>{agent.desc}</p>
            </div>
        </div>
    );
}

function ThemeToggle() {
    const { theme, toggle } = useTheme();
    return (
        <button onClick={toggle} className="btn-ghost p-2 rounded-lg" aria-label="Toggle theme">
            {theme === 'dark'
                ? <Sun className="w-4 h-4" style={{ color: '#FF5B21' }} />
                : <Moon className="w-4 h-4" style={{ color: 'var(--tx-2)' }} />}
        </button>
    );
}

export default function TaskPage() {
    const { id: requestId } = useParams();
    const router = useRouter();

    const [taskData, setTaskData] = useState(null);
    const [stateData, setStateData] = useState(null);
    const [loadError, setLoadError] = useState('');
    const [startTime] = useState(Date.now());
    const [elapsedStr, setElapsedStr] = useState('0s');

    const intervalRef = useRef(null);
    const isDone = taskData?.status === 'completed' || taskData?.status === 'approved' || taskData?.status === 'failed';

    // Elapsed clock
    useEffect(() => {
        const t = setInterval(() => setElapsedStr(elapsed(startTime)), 1000);
        return () => clearInterval(t);
    }, [startTime]);

    const poll = useCallback(async () => {
        try {
            const [s, st] = await Promise.all([
                getResearchStatus(requestId),
                getResearchState(requestId).catch(() => null),
            ]);
            setTaskData(s);
            if (st) setStateData(st);
        } catch (e) {
            setLoadError(e.message);
            clearInterval(intervalRef.current);
        }
    }, [requestId]);

    useEffect(() => {
        poll();
        intervalRef.current = setInterval(poll, POLL_MS);
        return () => clearInterval(intervalRef.current);
    }, [poll]);

    useEffect(() => {
        if (isDone) clearInterval(intervalRef.current);
    }, [isDone]);

    const currentStep = stateData?.current_step || null;
    const agentIndex = AGENTS.findIndex(a => a.key === currentStep);
    const report = stateData?.report;
    const sources = stateData?.sources || [];
    const pastResearch = stateData?.past_research || [];

    return (
        <div className="min-h-screen flex flex-col" style={{ backgroundColor: 'var(--bg)' }}>

            {/* Nav */}
            <header className="sticky top-0 z-40 px-6 py-3 flex items-center justify-between"
                style={{ backgroundColor: 'var(--bg-surface)', borderBottom: '1px solid var(--bd)' }}>
                <div className="flex items-center gap-4">
                    <button onClick={() => router.push('/')}
                        className="btn-ghost flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm">
                        <ArrowLeft className="w-4 h-4" />
                        New Research
                    </button>
                    {taskData?.topic && (
                        <span className="text-sm font-medium truncate max-w-xs hidden md:block"
                            style={{ color: 'var(--tx-2)' }}>
                            {taskData.topic}
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--tx-3)' }}>
                        <Clock className="w-3.5 h-3.5" />
                        {elapsedStr}
                    </div>
                    {taskData && <StatusBadge status={taskData.status} />}
                    <ThemeToggle />
                </div>
            </header>

            {/* Body */}
            <div className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 flex gap-8">

                {/* ── Left: Pipeline ─────────────────────────── */}
                <aside className="w-64 flex-shrink-0 space-y-3">
                    {/* Topic card */}
                    {taskData?.topic && (
                        <div className="card p-4">
                            <p className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--tx-3)' }}>Goal</p>
                            <p className="text-sm leading-relaxed" style={{ color: 'var(--tx)' }}>{taskData.topic}</p>
                        </div>
                    )}

                    {/* Stepper */}
                    <div className="card p-3 space-y-1">
                        <p className="text-xs font-semibold uppercase tracking-wider px-1 mb-2" style={{ color: 'var(--tx-3)' }}>
                            Agent Pipeline
                        </p>
                        {AGENTS.map((ag, idx) => (
                            <AgentStep
                                key={ag.key}
                                agent={ag}
                                isActive={idx === agentIndex}
                                isDone={isDone ? idx <= agentIndex : idx < agentIndex}
                            />
                        ))}
                    </div>

                    {/* Memory badge */}
                    {pastResearch.length > 0 && (
                        <div className="card p-4"
                            style={{ backgroundColor: 'rgba(0,115,254,0.05)', borderColor: 'rgba(0,115,254,0.2)' }}>
                            <div className="flex items-center gap-2 mb-1">
                                <BrainCircuit className="w-4 h-4" style={{ color: '#0073FE' }} />
                                <span className="text-xs font-semibold" style={{ color: '#0073FE' }}>Memory Injected</span>
                            </div>
                            <p className="text-xs" style={{ color: 'var(--tx-3)' }}>
                                {pastResearch.length} past report(s) given to Planner
                            </p>
                        </div>
                    )}

                    {loadError && (
                        <div className="card p-3 text-xs"
                            style={{ backgroundColor: 'rgba(255,91,33,0.07)', color: '#FF5B21', borderColor: 'rgba(255,91,33,0.2)' }}>
                            Polling error: {loadError}
                        </div>
                    )}
                </aside>

                {/* ── Right: Results ─────────────────────────── */}
                <section className="flex-1 min-w-0 space-y-6">

                    {/* Running skeleton */}
                    {!report && (
                        <div className="card p-8 flex flex-col items-center justify-center gap-6 text-center">
                            {sources.length === 0 ? (
                                <>
                                    <div className="w-12 h-12 rounded-full flex items-center justify-center"
                                        style={{ backgroundColor: 'rgba(0,115,254,0.1)' }}>
                                        <Loader2 className="w-6 h-6 animate-spin" style={{ color: '#0073FE' }} />
                                    </div>
                                    <div>
                                        <p className="font-semibold" style={{ color: 'var(--tx)' }}>Agents at work…</p>
                                        <p className="text-sm mt-1" style={{ color: 'var(--tx-3)' }}>Results will appear here as the pipeline runs.</p>
                                    </div>
                                    <div className="w-full max-w-sm space-y-2">
                                        <div className="skeleton h-3" />
                                        <div className="skeleton h-3 w-4/5" />
                                        <div className="skeleton h-3 w-3/5" />
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div className="w-full">
                                        <div className="flex items-center gap-2 mb-4">
                                            <BookOpen className="w-4 h-4" style={{ color: '#0073FE' }} />
                                            <span className="text-sm font-semibold" style={{ color: 'var(--tx)' }}>
                                                Sources Collected ({sources.length})
                                            </span>
                                        </div>
                                        <div className="space-y-1.5 max-h-72 overflow-y-auto">
                                            {sources.slice(0, 12).map((s, i) => (
                                                <a key={i} href={s.url} target="_blank" rel="noopener noreferrer"
                                                    className="flex items-start gap-2 p-2 rounded-lg transition-colors"
                                                    style={{ color: 'var(--tx-2)' }}
                                                    onMouseEnter={e => e.currentTarget.style.backgroundColor = 'var(--bg-muted)'}
                                                    onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                                                    <ExternalLink className="w-3 h-3 flex-shrink-0 mt-0.5" style={{ color: '#0073FE' }} />
                                                    <div className="min-w-0">
                                                        <p className="text-xs font-medium truncate">{s.title || s.url}</p>
                                                        <p className="text-xs truncate" style={{ color: 'var(--tx-3)' }}>{s.url}</p>
                                                    </div>
                                                </a>
                                            ))}
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    )}

                    {/* Final Report */}
                    {report && (
                        <div className="space-y-5 animate-fade-up">
                            {/* Header */}
                            <div className="card p-6">
                                <div className="flex items-start gap-4 mb-6 pb-6" style={{ borderBottom: '1px solid var(--bd)' }}>
                                    <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                                        style={{ backgroundColor: 'rgba(0,115,254,0.1)' }}>
                                        <Cpu className="w-5 h-5" style={{ color: '#0073FE' }} />
                                    </div>
                                    <div>
                                        <h1 className="text-2xl font-bold leading-tight" style={{ color: 'var(--tx)' }}>
                                            {report.topic || taskData?.topic}
                                        </h1>
                                        {report.executive_summary && (
                                            <p className="mt-2 leading-relaxed text-sm" style={{ color: 'var(--tx-2)' }}>
                                                {report.executive_summary}
                                            </p>
                                        )}
                                    </div>
                                </div>

                                {/* Key Findings */}
                                {report.key_findings?.length > 0 && (
                                    <div className="rounded-xl p-4 mb-5"
                                        style={{ backgroundColor: 'rgba(0,115,254,0.05)', border: '1px solid rgba(0,115,254,0.15)' }}>
                                        <p className="text-xs font-bold uppercase tracking-wider mb-3" style={{ color: '#0073FE' }}>
                                            Key Findings
                                        </p>
                                        <ul className="space-y-2">
                                            {report.key_findings.map((f, i) => (
                                                <li key={i} className="flex items-start gap-2 text-sm" style={{ color: 'var(--tx)' }}>
                                                    <span style={{ color: '#0073FE', fontWeight: 700 }}>→</span> {f}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Sections */}
                                {report.sections?.length > 0 && (
                                    <div className="prose max-w-none">
                                        {report.sections.map((sec, i) => (
                                            <div key={i} className="mb-6">
                                                <h2 className="text-base font-bold mb-2" style={{ color: 'var(--tx)' }}>
                                                    <span style={{ color: '#FF5B21', marginRight: '6px' }}>#{i + 1}</span>
                                                    {sec.title}
                                                </h2>
                                                <ReactMarkdown>{sec.content}</ReactMarkdown>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Sources */}
                            {sources.length > 0 && (
                                <div className="card p-5">
                                    <div className="flex items-center gap-2 mb-4">
                                        <BookOpen className="w-4 h-4" style={{ color: '#3FA759' }} />
                                        <span className="text-sm font-semibold" style={{ color: 'var(--tx)' }}>
                                            Sources Analysed ({sources.length})
                                        </span>
                                    </div>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                        {sources.slice(0, 10).map((s, i) => (
                                            <a key={i} href={s.url} target="_blank" rel="noopener noreferrer"
                                                className="flex items-start gap-2 p-3 rounded-xl transition-all"
                                                style={{ border: '1px solid var(--bd)' }}
                                                onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--bd-hover)'; e.currentTarget.style.backgroundColor = 'var(--bg-muted)'; }}
                                                onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--bd)'; e.currentTarget.style.backgroundColor = 'transparent'; }}>
                                                <ExternalLink className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: '#3FA759' }} />
                                                <div className="min-w-0">
                                                    <p className="text-xs font-medium truncate" style={{ color: 'var(--tx)' }}>{s.title || 'Source'}</p>
                                                    <p className="text-xs truncate" style={{ color: 'var(--tx-3)' }}>{s.url}</p>
                                                </div>
                                            </a>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                </section>
            </div>
        </div>
    );
}
