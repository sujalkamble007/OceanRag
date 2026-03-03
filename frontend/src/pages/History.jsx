import { useState, useEffect } from 'react';
import apiClient from '../api/client';
import { Search, Hash, Clock, Database, ChevronDown, ChevronRight, MessageSquare } from 'lucide-react';

const History = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [expandedId, setExpandedId] = useState(null);
    const [details, setDetails] = useState({});

    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        try {
            const res = await apiClient.get('/history/');
            setHistory(res.data.records);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const fetchDetails = async (id) => {
        if (expandedId === id) {
            setExpandedId(null);
            return;
        }

        if (!details[id]) {
            try {
                const res = await apiClient.get(`/history/${id}`);
                setDetails(prev => ({ ...prev, [id]: res.data }));
            } catch (err) {
                console.error(err);
            }
        }
        setExpandedId(id);
    };

    if (loading) return <div className="p-6 text-ocean-400">Loading history...</div>;

    return (
        <div className="max-w-5xl mx-auto space-y-6">
            <div className="bg-ocean-800/50 p-6 rounded-2xl border border-ocean-700">
                <h1 className="text-2xl font-bold text-white mb-2">Query History</h1>
                <p className="text-sm text-slate-400">Review past conversations, latencies, and generation costs.</p>

                {/* Simple search bar UI */}
                <div className="mt-6 relative">
                    <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />
                    <input
                        type="text"
                        placeholder="Search history... (UI Only)"
                        className="w-full bg-ocean-900 border border-ocean-700 text-white rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-1 focus:ring-ocean-400/50 text-sm"
                    />
                </div>
            </div>

            <div className="space-y-3">
                {history.length === 0 ? (
                    <div className="p-8 text-center bg-ocean-800/20 border border-ocean-700/50 rounded-xl text-slate-500">
                        No query history found
                    </div>
                ) : history.map((record) => (
                    <div key={record.id} className="bg-ocean-800/30 border border-ocean-700 rounded-xl overflow-hidden shadow-sm hover:border-ocean-600 transition-colors">
                        {/* Header row (clickable) */}
                        <div
                            className="p-4 flex items-center justify-between cursor-pointer"
                            onClick={() => fetchDetails(record.id)}
                        >
                            <div className="flex items-center gap-4 min-w-0 pr-4">
                                <div className="w-10 h-10 rounded-lg bg-ocean-700 flex items-center justify-center shrink-0 text-slate-400">
                                    <MessageSquare size={18} />
                                </div>
                                <div className="truncate">
                                    <h4 className="text-sm font-semibold text-slate-200 truncate">{record.query_text}</h4>
                                    <div className="flex items-center gap-3 mt-1 text-[11px] text-slate-500 font-medium">
                                        <span className="flex items-center gap-1"><Clock size={10} /> {record.run_at ? new Date(record.run_at).toLocaleString() : 'Unknown'}</span>
                                        <span className="flex items-center gap-1"><Hash size={10} /> {record.llm_name}</span>
                                        <span className="text-ocean-400">{record.latency_seconds?.toFixed(2)}s</span>
                                    </div>
                                </div>
                            </div>
                            <div className="text-slate-500 shrink-0">
                                {expandedId === record.id ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
                            </div>
                        </div>

                        {/* Expanded Details */}
                        {expandedId === record.id && details[record.id] && (
                            <div className="p-6 bg-ocean-900 border-t border-ocean-700">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div>
                                        <h5 className="text-xs uppercase tracking-wider text-ocean-400 font-semibold mb-2">Assistant Answer</h5>
                                        <div className="bg-ocean-800/50 p-4 rounded-lg text-sm text-slate-300 leading-relaxed border border-ocean-700">
                                            {details[record.id].answer_text}
                                        </div>
                                    </div>
                                    <div>
                                        <h5 className="text-xs uppercase tracking-wider text-ocean-400 font-semibold mb-2">Technical Details</h5>
                                        <div className="bg-ocean-800/50 p-4 rounded-lg border border-ocean-700 space-y-3">
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-400">Retriever Type</span>
                                                <span className="text-slate-200 font-medium">{details[record.id].retriever_type}</span>
                                            </div>
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-400">Embedding</span>
                                                <span className="text-slate-200 font-medium">{details[record.id].embedding_model}</span>
                                            </div>
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-400">Total Tokens</span>
                                                <span className="text-slate-200 font-medium">
                                                    {(details[record.id].input_tokens || 0) + (details[record.id].output_tokens || 0)}
                                                </span>
                                            </div>
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-400">Estimated Cost</span>
                                                <span className="text-green-400 font-medium">${details[record.id].cost_usd?.toFixed(5)}</span>
                                            </div>
                                            <div className="border-t border-ocean-700 pt-3 mt-3">
                                                <span className="block text-xs text-slate-400 mb-1">Sources</span>
                                                <div className="flex flex-wrap gap-1">
                                                    {details[record.id].sources?.map((s, i) => (
                                                        <span key={i} className="text-[10px] px-2 py-1 rounded bg-ocean-700 text-slate-300">
                                                            {s.split('/').pop()}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default History;
