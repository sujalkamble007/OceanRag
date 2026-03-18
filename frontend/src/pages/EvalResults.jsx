import { useState, useEffect } from 'react';
import apiClient from '../api/client';
import { FlaskConical, Download, Filter, ChevronDown } from 'lucide-react';

const PHASE_LABELS = {
    'A': 'Phase A — Chunking',
    'B': 'Phase B — Embedding',
    'C': 'Phase C — Retriever',
    'D': 'Phase D — LLM',
    'FULL': 'Full Matrix',
    'QUICK': 'Quick Eval',
};

const METRIC_COLS = [
    { key: 'precision_at_k', label: 'P@K', color: 'text-blue-400' },
    { key: 'recall_at_k', label: 'R@K', color: 'text-cyan-400' },
    { key: 'mrr', label: 'MRR', color: 'text-purple-400' },
    { key: 'hit_rate', label: 'Hit Rate', color: 'text-indigo-400' },
    { key: 'rouge_l', label: 'ROUGE-L', color: 'text-amber-400' },
    { key: 'bleu', label: 'BLEU', color: 'text-orange-400' },
    { key: 'bertscore', label: 'BERTScore', color: 'text-pink-400' },
    { key: 'faithfulness', label: 'Faithful', color: 'text-green-400' },
];

const EvalResults = () => {
    const [experiments, setExperiments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedPhase, setSelectedPhase] = useState('all');
    const [sortKey, setSortKey] = useState('precision_at_k');
    const [sortDir, setSortDir] = useState('desc');

    useEffect(() => {
        fetchExperiments();
    }, []);

    const fetchExperiments = async () => {
        try {
            const res = await apiClient.get('/eval/experiments');
            setExperiments(res.data.results || []);
        } catch (err) {
            console.error('Failed to fetch eval results:', err);
        } finally {
            setLoading(false);
        }
    };

    const filtered = experiments.filter(exp => {
        if (selectedPhase === 'all') return true;
        return exp.phase === selectedPhase;
    });

    const sorted = [...filtered].sort((a, b) => {
        const aVal = a[sortKey] ?? 0;
        const bVal = b[sortKey] ?? 0;
        return sortDir === 'desc' ? bVal - aVal : aVal - bVal;
    });

    const handleSort = (key) => {
        if (sortKey === key) {
            setSortDir(sortDir === 'desc' ? 'asc' : 'desc');
        } else {
            setSortKey(key);
            setSortDir('desc');
        }
    };

    const handleExport = () => {
        if (sorted.length === 0) return;
        const keys = ['phase', 'chunk_strategy', 'embedding_model', 'retriever_type', 'llm_name', 'top_k',
            ...METRIC_COLS.map(c => c.key), 'latency_seconds', 'cost_per_query'];
        const csvContent =
            keys.join(',') + '\n' +
            sorted.map(row => keys.map(k => `"${row[k] ?? ''}"`).join(',')).join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.setAttribute('href', URL.createObjectURL(blob));
        link.setAttribute('download', 'oceanrag_eval_results.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const phases = ['all', ...new Set(experiments.map(e => e.phase).filter(Boolean))];

    if (loading) return <div className="p-6 text-ocean-400">Loading evaluation results...</div>;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex justify-between items-end bg-ocean-800/50 p-6 rounded-2xl border border-ocean-700">
                <div>
                    <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
                        <FlaskConical className="text-ocean-400" /> Research Evaluation Results
                    </h1>
                    <p className="text-sm text-slate-400 max-w-2xl">
                        Systematic comparison of chunking strategies, embedding models, retrievers, and LLMs.
                        All metrics stored per experiment run. Click column headers to sort.
                    </p>
                </div>
                <button
                    onClick={handleExport}
                    className="flex items-center gap-2 bg-ocean-400 hover:bg-ocean-300 text-ocean-900 px-4 py-2 rounded-xl text-sm font-bold transition-colors shadow-[0_0_15px_rgba(100,255,218,0.2)]"
                >
                    <Download size={16} /> Export CSV
                </button>
            </div>

            {/* Phase Filter */}
            <div className="flex items-center gap-2 flex-wrap">
                <Filter size={16} className="text-slate-400" />
                {phases.map(phase => (
                    <button
                        key={phase}
                        onClick={() => setSelectedPhase(phase)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                            selectedPhase === phase
                                ? 'bg-ocean-400 text-ocean-900 shadow-[0_0_10px_rgba(100,255,218,0.3)]'
                                : 'bg-ocean-800 text-slate-400 border border-ocean-700 hover:border-ocean-500'
                        }`}
                    >
                        {phase === 'all' ? 'All Phases' : PHASE_LABELS[phase] || phase}
                    </button>
                ))}
                <span className="text-xs text-slate-500 ml-2">{sorted.length} results</span>
            </div>

            {/* Results Table */}
            <div className="bg-ocean-800/30 border border-ocean-700 rounded-2xl overflow-hidden shadow-xl">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left text-slate-300">
                        <thead className="text-[11px] uppercase bg-ocean-900/80 text-ocean-400 border-b border-ocean-700 font-semibold tracking-wider">
                            <tr>
                                <th className="px-4 py-3 sticky left-0 bg-ocean-900/80 z-10">#</th>
                                <th className="px-4 py-3">Phase</th>
                                <th className="px-4 py-3">Chunk</th>
                                <th className="px-4 py-3">Embedding</th>
                                <th className="px-4 py-3">Retriever</th>
                                <th className="px-4 py-3">LLM</th>
                                <th className="px-3 py-3 text-center">K</th>
                                {METRIC_COLS.map(col => (
                                    <th
                                        key={col.key}
                                        className="px-3 py-3 text-center cursor-pointer hover:text-white transition-colors select-none"
                                        onClick={() => handleSort(col.key)}
                                    >
                                        <div className="flex items-center justify-center gap-1">
                                            {col.label}
                                            {sortKey === col.key && (
                                                <ChevronDown size={12} className={`transition-transform ${sortDir === 'asc' ? 'rotate-180' : ''}`} />
                                            )}
                                        </div>
                                    </th>
                                ))}
                                <th className="px-3 py-3 text-center cursor-pointer hover:text-white" onClick={() => handleSort('latency_seconds')}>
                                    Latency
                                </th>
                                <th className="px-3 py-3 text-center cursor-pointer hover:text-white" onClick={() => handleSort('cost_per_query')}>
                                    Cost
                                </th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-ocean-700/50">
                            {sorted.length === 0 ? (
                                <tr>
                                    <td colSpan={16} className="px-6 py-16 text-center text-slate-500">
                                        <div className="space-y-2">
                                            <p className="text-lg">No evaluation results yet</p>
                                            <p className="text-xs">Run <code className="bg-ocean-800 px-2 py-1 rounded text-ocean-400">python run_research.py --phase A</code> to start</p>
                                        </div>
                                    </td>
                                </tr>
                            ) : sorted.map((exp, idx) => (
                                <tr key={exp.id || idx} className={`hover:bg-ocean-700/30 transition-colors ${idx === 0 ? 'bg-ocean-400/5' : ''}`}>
                                    <td className="px-4 py-3 sticky left-0 bg-ocean-800/80 z-10">
                                        <span className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold ${
                                            idx === 0 ? 'bg-ocean-400 text-ocean-900' : 'bg-ocean-800 border border-ocean-600 text-slate-400'
                                        }`}>{idx + 1}</span>
                                    </td>
                                    <td className="px-4 py-3">
                                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                                            exp.phase === 'A' ? 'bg-blue-900/50 text-blue-400' :
                                            exp.phase === 'B' ? 'bg-purple-900/50 text-purple-400' :
                                            exp.phase === 'C' ? 'bg-amber-900/50 text-amber-400' :
                                            exp.phase === 'D' ? 'bg-green-900/50 text-green-400' :
                                            'bg-ocean-800 text-slate-400'
                                        }`}>{exp.phase || '—'}</span>
                                    </td>
                                    <td className="px-4 py-3 text-xs whitespace-nowrap">{exp.chunk_strategy || '—'}</td>
                                    <td className="px-4 py-3 text-xs whitespace-nowrap">{exp.embedding_model || '—'}</td>
                                    <td className="px-4 py-3 text-xs whitespace-nowrap">{exp.retriever_type || '—'}</td>
                                    <td className="px-4 py-3 text-xs whitespace-nowrap font-medium text-slate-200">{exp.llm_name || '—'}</td>
                                    <td className="px-3 py-3 text-center text-xs">{exp.top_k ?? '—'}</td>
                                    {METRIC_COLS.map(col => {
                                        const val = exp[col.key];
                                        const isHigh = val != null && val >= 0.8;
                                        return (
                                            <td key={col.key} className="px-3 py-3 text-center text-xs">
                                                <span className={`font-mono ${isHigh ? col.color + ' font-bold' : 'text-slate-400'}`}>
                                                    {val != null ? val.toFixed(3) : '—'}
                                                </span>
                                            </td>
                                        );
                                    })}
                                    <td className="px-3 py-3 text-center text-xs text-slate-400 font-mono">
                                        {exp.latency_seconds != null ? `${exp.latency_seconds.toFixed(1)}s` : '—'}
                                    </td>
                                    <td className="px-3 py-3 text-center text-xs font-mono">
                                        {exp.cost_per_query != null ? (
                                            exp.cost_per_query === 0 ? (
                                                <span className="text-green-400 font-semibold">FREE</span>
                                            ) : `$${exp.cost_per_query.toFixed(4)}`
                                        ) : '—'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default EvalResults;
