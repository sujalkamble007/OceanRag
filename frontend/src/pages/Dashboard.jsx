import { useState, useEffect } from 'react';
import apiClient from '../api/client';
import { Clock, Download, Database, Cpu, Activity, ThumbsUp } from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
    ScatterChart, Scatter, ZAxis, Legend, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

const MetricCard = ({ title, value, icon: Icon, colorClass }) => (
    <div className="bg-ocean-800/50 border border-ocean-700 rounded-2xl p-5 flex items-center gap-4">
        <div className={`p-4 rounded-xl flex items-center justify-center shrink-0 border shadow-lg ${colorClass}`}>
            <Icon size={24} />
        </div>
        <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">{title}</p>
            <h3 className="text-2xl font-bold text-slate-100 mt-1">{value}</h3>
        </div>
    </div>
);

const Dashboard = () => {
    const [stats, setStats] = useState(null);
    const [chartData, setChartData] = useState(null);
    const [feedbackStats, setFeedbackStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [statsRes, chartsRes, feedbackRes] = await Promise.all([
                    apiClient.get('/dashboard/stats'),
                    apiClient.get('/dashboard/charts'),
                    apiClient.get('/dashboard/feedback-stats').catch(() => ({ data: null }))
                ]);
                setStats(statsRes.data);
                if (chartsRes.data.data) {
                    setChartData(chartsRes.data.data);
                }
                setFeedbackStats(feedbackRes.data);
            } catch (error) {
                console.error("Failed to load dashboard data", error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <div className="flex items-center justify-center h-full text-ocean-400"><Database className="animate-spin" size={32} /></div>;

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-ocean-900 border border-ocean-600 p-3 rounded-lg shadow-xl text-xs">
                    <p className="font-bold text-white mb-2">{label || payload[0].payload.name}</p>
                    {payload.map((entry, index) => (
                        <p key={index} style={{ color: entry.color }} className="font-medium">
                            {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
                        </p>
                    ))}
                </div>
            );
        }
        return null;
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center bg-ocean-800/50 p-6 rounded-2xl border border-ocean-700">
                <div>
                    <h1 className="text-2xl font-bold text-white">System Analytics</h1>
                    <p className="text-sm text-slate-400 mt-1">Real-time performance metrics and evaluation data</p>
                </div>
                <button className="flex items-center gap-2 bg-ocean-700 hover:bg-ocean-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors border border-ocean-600">
                    <Download size={16} /> Export PDF
                </button>
            </div>

            {stats && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <MetricCard title="Documents" value={stats.total_documents} icon={Database} colorClass="bg-blue-500/20 text-blue-400 border-blue-500/30" />
                    <MetricCard title="Vector Chunks" value={(stats.total_chunks).toLocaleString()} icon={Cpu} colorClass="bg-purple-500/20 text-purple-400 border-purple-500/30" />
                    <MetricCard title="Total Queries" value={stats.total_queries} icon={Activity} colorClass="bg-green-500/20 text-green-400 border-green-500/30" />
                    <MetricCard title="Avg Latency" value={`${stats.avg_latency}s`} icon={Clock} colorClass="bg-orange-500/20 text-orange-400 border-orange-500/30" />
                </div>
            )}

            {feedbackStats && (
                <div className="bg-ocean-800/50 border border-ocean-700 rounded-2xl p-6 flex justify-around">
                    <div className="text-center">
                        <p className="text-slate-400 text-xs uppercase tracking-wider mb-2">Total Feedback</p>
                        <h4 className="text-2xl font-bold text-white">{feedbackStats.total_feedback}</h4>
                    </div>
                    <div className="text-center">
                        <p className="text-slate-400 text-xs uppercase tracking-wider mb-2">Thumbs Up</p>
                        <h4 className="text-2xl font-bold text-green-400 flex justify-center items-center gap-2"><ThumbsUp size={18} /> {feedbackStats.thumbs_up}</h4>
                    </div>
                    <div className="text-center">
                        <p className="text-slate-400 text-xs uppercase tracking-wider mb-2">Avg Rating</p>
                        <h4 className="text-2xl font-bold text-ocean-400">{feedbackStats.avg_rating > 0 ? "+" : ""}{feedbackStats.avg_rating}</h4>
                    </div>
                </div>
            )}

            {!chartData ? (
                <div className="bg-ocean-800/50 border border-ocean-700 rounded-2xl p-12 text-center text-slate-400">
                    No chart data available. Please run Phase 4 Evaluation Module first.
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                    {/* Scatter Chart */}
                    <div className="bg-ocean-800/50 border border-ocean-700 rounded-2xl p-6">
                        <h3 className="text-base font-bold text-slate-200 mb-6">Precision vs Recall Analysis</h3>
                        <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#233554" />
                                    <XAxis type="number" dataKey="recall" name="Recall@K" stroke="#64748b" domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} />
                                    <YAxis type="number" dataKey="precision" name="Precision@K" stroke="#64748b" domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} />
                                    <ZAxis type="category" dataKey="name" name="Config" />
                                    <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
                                    <Legend />
                                    {chartData.precision_recall_scatter?.map((series, i) => (
                                        <Scatter key={series.retriever} name={series.retriever} data={series.data} fill={['#64ffda', '#3b82f6', '#8b5cf6'][i % 3]} />
                                    ))}
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Bar Chart */}
                    <div className="bg-ocean-800/50 border border-ocean-700 rounded-2xl p-6">
                        <h3 className="text-base font-bold text-slate-200 mb-6">Average Latency by LLM (seconds)</h3>
                        <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData.latency_bar} margin={{ top: 20, right: 20, bottom: 0, left: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#233554" vertical={false} />
                                    <XAxis dataKey="llm_name" stroke="#64748b" />
                                    <YAxis stroke="#64748b" />
                                    <RechartsTooltip cursor={{ fill: '#233554' }} content={<CustomTooltip />} />
                                    <Bar dataKey="avg_latency" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Radar Chart */}
                    {chartData.retriever_radar && (
                        <div className="bg-ocean-800/50 border border-ocean-700 rounded-2xl p-6 lg:col-span-2">
                            <h3 className="text-base font-bold text-slate-200 mb-6">Retriever Multi-Metric Profile</h3>
                            <div className="h-96">
                                <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData.retriever_radar}>
                                        <PolarGrid stroke="#233554" />
                                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                        <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
                                        <RechartsTooltip content={<CustomTooltip />} />
                                        <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                        <Radar name="Similarity" dataKey="similarity" stroke="#64ffda" fill="#64ffda" fillOpacity={0.4} />
                                        <Radar name="MMR" dataKey="mmr" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.4} />
                                        <Radar name="Hybrid" dataKey="hybrid" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.4} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default Dashboard;
