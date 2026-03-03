import { useState, useEffect } from 'react';
import apiClient from '../api/client';
import { Award, Download, Filter, Target, Cpu, Hash } from 'lucide-react';
import useAuthStore from '../store/authStore';

const Experiments = () => {
    const { user } = useAuthStore();
    const [leaderboard, setLeaderboard] = useState([]);
    const [loading, setLoading] = useState(true);

    if (user?.role === 'common_user') {
        return (
            <div className="p-8 text-center mt-20">
                <h2 className="text-2xl font-bold text-slate-300">Access Denied</h2>
                <p className="text-slate-500 mt-2">Your role does not have permission to view evaluation experiments.</p>
            </div>
        );
    }

    useEffect(() => {
        const fetchLeaderboard = async () => {
            try {
                const res = await apiClient.get('/experiments/leaderboard');
                setLeaderboard(res.data.leaderboard);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchLeaderboard();
    }, []);

    const handleExport = () => {
        // Generate simple CSV
        if (leaderboard.length === 0) return;
        const keys = Object.keys(leaderboard[0]);
        const csvContent =
            keys.join(",") + "\n" +
            leaderboard.map(row => keys.map(k => `"${row[k]}"`).join(",")).join("\n");

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "oceanrag_experiments.csv");
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    if (loading) return <div className="p-6 text-ocean-400">Loading experiments...</div>;

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-end bg-ocean-800/50 p-6 rounded-2xl border border-ocean-700">
                <div>
                    <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
                        <Award className="text-ocean-400" /> Evaluation Matrix
                    </h1>
                    <p className="text-sm text-slate-400 max-w-2xl">
                        RAGAS metrics and IR evaluations (Precision, Recall, Faithfulness) ranked by composite score.
                        Used to determine optimal chunking, embedding, and retriever combinations.
                    </p>
                </div>
                <button
                    onClick={handleExport}
                    className="flex items-center gap-2 bg-ocean-400 hover:bg-ocean-300 text-ocean-900 px-4 py-2 rounded-xl text-sm font-bold transition-colors shadow-[0_0_15px_rgba(100,255,218,0.2)]"
                >
                    <Download size={16} /> Export CSV
                </button>
            </div>

            <div className="bg-ocean-800/30 border border-ocean-700 rounded-2xl overflow-hidden shadow-xl">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left text-slate-300">
                        <thead className="text-xs uppercase bg-ocean-900/80 text-ocean-400 border-b border-ocean-700 font-semibold tracking-wider">
                            <tr>
                                <th className="px-6 py-4">Rank</th>
                                <th className="px-6 py-4">Configuration</th>
                                <th className="px-6 py-4">Retrieval Metrics</th>
                                <th className="px-6 py-4">Generation Metrics</th>
                                <th className="px-6 py-4 text-right">Composite Score</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-ocean-700/50">
                            {leaderboard.length === 0 ? (
                                <tr>
                                    <td colSpan="5" className="px-6 py-12 text-center text-slate-500">
                                        No experiments found. Run Phase 4 Evaluation first.
                                    </td>
                                </tr>
                            ) : leaderboard.map((exp, idx) => (
                                <tr key={exp.id} className={`hover:bg-ocean-700/30 transition-colors ${idx === 0 ? 'bg-ocean-400/5' : ''}`}>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        {idx === 0 ? (
                                            <span className="flex items-center justify-center w-8 h-8 rounded-full bg-ocean-400 text-ocean-900 font-bold">1</span>
                                        ) : (
                                            <span className="flex items-center justify-center w-8 h-8 rounded-full bg-ocean-800 border border-ocean-600 font-semibold text-slate-400">{exp.rank}</span>
                                        )}
                                    </td>

                                    <td className="px-6 py-4">
                                        <div className="space-y-1">
                                            <div className="flex items-center gap-2 text-slate-200 font-medium">
                                                <Cpu size={14} className="text-purple-400" /> {exp.llm_name}
                                            </div>
                                            <div className="flex items-center gap-2 text-xs text-slate-400">
                                                <Filter size={12} /> {exp.retriever_type} | Top {exp.top_k}
                                            </div>
                                            <div className="flex items-center gap-2 text-xs text-slate-500">
                                                <Hash size={12} /> {exp.chunk_strategy}
                                            </div>
                                        </div>
                                    </td>

                                    <td className="px-6 py-4">
                                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                                            <div className="flex justify-between">
                                                <span className="text-slate-500">P@K:</span>
                                                <span className="text-slate-300 font-medium">{exp.precision_at_k?.toFixed(3)}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-slate-500">R@K:</span>
                                                <span className="text-slate-300 font-medium">{exp.recall_at_k?.toFixed(3)}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-slate-500">MRR:</span>
                                                <span className="text-slate-300 font-medium">{exp.mrr?.toFixed(3)}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-slate-500">Hit %:</span>
                                                <span className="text-slate-300 font-medium text-blue-400">{(exp.hit_rate * 100)?.toFixed(1)}%</span>
                                            </div>
                                        </div>
                                    </td>

                                    <td className="px-6 py-4">
                                        <div className="grid grid-cols-1 gap-y-1 text-xs">
                                            <div className="flex justify-between w-32">
                                                <span className="text-slate-500">Faithful:</span>
                                                <span className={`${(exp.faithfulness || 0) > 0.8 ? 'text-green-400' : 'text-slate-300'} font-medium`}>
                                                    {exp.faithfulness?.toFixed(3) || 'N/A'}
                                                </span>
                                            </div>
                                            <div className="flex justify-between w-32">
                                                <span className="text-slate-500">Relevant:</span>
                                                <span className="text-slate-300 font-medium">{exp.answer_relevancy?.toFixed(3) || 'N/A'}</span>
                                            </div>
                                        </div>
                                    </td>

                                    <td className="px-6 py-4 text-right whitespace-nowrap">
                                        <div className="text-lg font-bold text-ocean-400">
                                            {exp.composite_score?.toFixed(4)}
                                        </div>
                                        {idx === 0 && <div className="text-[10px] text-ocean-300 uppercase tracking-widest font-semibold mt-1">Recommended</div>}
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

export default Experiments;
