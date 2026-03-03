import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, Settings2, FileText, ChevronDown, ThumbsUp, ThumbsDown } from 'lucide-react';
import apiClient from '../api/client';
import useAuthStore from '../store/authStore';

const Chat = () => {
    const { user } = useAuthStore();
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Hello! I am the OceanRAG assistant. Ask me anything about Deep-Sea Governance & UNCLOS regulations.' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);

    // Controls
    const [llmKey, setLlmKey] = useState('groq-llama8b');
    const [retrieverType, setRetrieverType] = useState('mmr');
    const [topK, setTopK] = useState(5);

    const endOfMessagesRef = useRef(null);

    // Auto-scroll
    useEffect(() => {
        endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        // Add a placeholder assistant message that will be streamed into
        setMessages(prev => [...prev, { role: 'assistant', content: '', streaming: true }]);

        try {
            const token = useAuthStore.getState().token;
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            const response = await fetch(`${apiUrl}/query/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify({
                    query: userMessage.content,
                    llm_key: llmKey,
                    retriever_type: retrieverType,
                    top_k: Number(topK),
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line in buffer

                let eventType = '';
                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        eventType = line.slice(7).trim();
                    } else if (line.startsWith('data: ') && eventType) {
                        const data = JSON.parse(line.slice(6));

                        if (eventType === 'token') {
                            setMessages(prev => {
                                const updated = [...prev];
                                const last = updated[updated.length - 1];
                                if (last && last.role === 'assistant') {
                                    updated[updated.length - 1] = { ...last, content: last.content + data.token };
                                }
                                return updated;
                            });
                        } else if (eventType === 'done') {
                            setMessages(prev => {
                                const updated = [...prev];
                                const last = updated[updated.length - 1];
                                if (last && last.role === 'assistant') {
                                    updated[updated.length - 1] = {
                                        ...last,
                                        streaming: false,
                                        sources: data.sources || [],
                                        latency: data.latency_seconds || 0,
                                        llm: data.llm_name || llmKey,
                                        recordId: data.record_id || 0,
                                    };
                                }
                                return updated;
                            });
                        }
                        eventType = '';
                    }
                }
            }
        } catch (err) {
            setMessages(prev => {
                // If the last message is the streaming placeholder, remove it and add error
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last && last.role === 'assistant' && last.streaming) {
                    updated.pop();
                }
                return [...updated, {
                    role: 'system',
                    content: `Error: ${err.message}`
                }];
            });
        } finally {
            setLoading(false);
        }
    };

    const handleFeedback = async (recordId, rating) => {
        if (!recordId) return;
        try {
            await apiClient.post('/feedback/', {
                qa_log_id: recordId,
                rating: rating,
                comment: ""
            });
            // Simple visual feedback (alert for now, could be a toast)
            alert(rating === 1 ? "Thanks for the positive feedback!" : "Thanks for the feedback. We'll improve.");
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className="flex h-full gap-6">
            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col bg-ocean-800/50 border border-ocean-700 rounded-2xl overflow-hidden shadow-xl">
                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>

                            {msg.role !== 'user' && (
                                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${msg.role === 'system' ? 'bg-red-500/20 text-red-400' : 'bg-ocean-700 border border-ocean-600'}`}>
                                    {msg.role === 'system' ? '⚠️' : <Bot size={20} className="text-ocean-400" />}
                                </div>
                            )}

                            <div className={`max-w-[80%] rounded-2xl p-5 shadow-sm ${msg.role === 'user'
                                ? 'bg-ocean-400 text-ocean-900 rounded-tr-sm'
                                : msg.role === 'system'
                                    ? 'bg-red-500/10 border border-red-500/20 text-red-200 rounded-tl-sm'
                                    : 'bg-ocean-700/50 border border-ocean-600 text-slate-200 rounded-tl-sm'
                                }`}>
                                <div className="whitespace-pre-wrap leading-relaxed">{msg.content}{msg.streaming && <span className="animate-pulse text-ocean-400">▌</span>}</div>

                                {/* Metadata & Sources for Assistant */}
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="mt-4 pt-4 border-t border-ocean-600/50">
                                        <p className="text-xs font-semibold uppercase tracking-wider text-ocean-400 mb-2 flex items-center gap-1">
                                            <FileText size={12} /> Sources Consulted
                                        </p>
                                        <div className="flex flex-wrap gap-2">
                                            {msg.sources.map((src, i) => {
                                                // src can be a plain string OR an object with a filename field
                                                const label = (typeof src === 'string' ? src : (src?.filename || src?.filepath || String(src) || ''));
                                                const display = label.split('/').pop() || label;
                                                return (
                                                    <div key={i} className="text-[11px] px-2 py-1 rounded bg-ocean-900 border border-ocean-700 text-slate-300">
                                                        {display}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}

                                {msg.role === 'assistant' && msg.latency && (
                                    <div className="mt-3 flex items-center justify-between text-[10px] text-slate-400 font-medium">
                                        <div className="flex items-center gap-2">
                                            <span className="flex items-center gap-1 bg-ocean-900 px-2 py-0.5 rounded border border-ocean-700">
                                                <Sparkles size={10} className="text-ocean-400" /> {msg.llm}
                                            </span>
                                            <span>{msg.latency.toFixed(2)}s</span>
                                        </div>
                                        {msg.recordId && (
                                            <div className="flex gap-1">
                                                <button onClick={() => handleFeedback(msg.recordId, 1)} className="p-1 hover:text-green-400 hover:bg-green-400/10 rounded transition-colors" title="Helpful">
                                                    <ThumbsUp size={12} />
                                                </button>
                                                <button onClick={() => handleFeedback(msg.recordId, -1)} className="p-1 hover:text-red-400 hover:bg-red-400/10 rounded transition-colors" title="Not Helpful">
                                                    <ThumbsDown size={12} />
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            {msg.role === 'user' && (
                                <div className="w-10 h-10 rounded-xl bg-ocean-400 flex items-center justify-center shrink-0">
                                    <User size={20} className="text-ocean-900" />
                                </div>
                            )}

                        </div>
                    ))}
                    {loading && !messages.some(m => m.role === 'assistant' && m.streaming && m.content) && (
                        <div className="flex gap-4 justify-start">
                            <div className="w-10 h-10 rounded-xl bg-ocean-700 border border-ocean-600 flex items-center justify-center shrink-0">
                                <Bot size={20} className="text-ocean-400 animate-pulse" />
                            </div>
                            <div className="bg-ocean-700/50 border border-ocean-600 rounded-2xl rounded-tl-sm p-5 flex items-center gap-2">
                                <div className="w-2 h-2 bg-ocean-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                <div className="w-2 h-2 bg-ocean-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                <div className="w-2 h-2 bg-ocean-400 rounded-full animate-bounce"></div>
                            </div>
                        </div>
                    )}
                    <div ref={endOfMessagesRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 bg-ocean-800 border-t border-ocean-700">
                    <form onSubmit={handleSend} className="relative flex items-center">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask a question about deep-sea mining regulations..."
                            className="w-full bg-ocean-900 border border-ocean-700 text-white rounded-xl pl-4 pr-12 py-4 focus:outline-none focus:ring-2 focus:ring-ocean-400/50 transition-all placeholder:text-slate-500"
                            disabled={loading}
                        />
                        <button
                            type="submit"
                            disabled={loading || !input.trim()}
                            className="absolute right-2 p-2 bg-ocean-400 hover:bg-ocean-300 disabled:bg-ocean-700 disabled:text-slate-500 text-ocean-900 rounded-lg transition-colors"
                        >
                            <Send size={18} />
                        </button>
                    </form>
                </div>
            </div>

            {/* Right Sidebar Controls */}
            <div className="w-80 bg-ocean-800/50 border border-ocean-700 rounded-2xl p-6 flex flex-col gap-6 shadow-xl hidden lg:flex">
                <div className="flex items-center gap-2 mb-2 pb-4 border-b border-ocean-700">
                    <Settings2 size={20} className="text-ocean-400" />
                    <h2 className="text-lg font-bold text-white">Parameters</h2>
                </div>

                <div className="space-y-4">
                    <div>
                        <label className="block text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Language Model</label>
                        <div className="relative">
                            <select
                                value={llmKey}
                                onChange={(e) => setLlmKey(e.target.value)}
                                className="w-full bg-ocean-900 border border-ocean-700 text-slate-200 rounded-xl px-4 py-2.5 appearance-none focus:outline-none focus:border-ocean-400 disabled:opacity-50"
                                disabled={user?.role === 'common_user'}
                            >
                                <option value="groq-llama8b">Llama 3 (8B) - Fast</option>
                                {user?.role !== 'common_user' && <option value="groq-llama70b">Llama 3 (70B) - Smart</option>}
                                {user?.role !== 'common_user' && <option value="zephyr-7b">Zephyr 7B - HF</option>}
                                <option value="qwen2.5-72b">Qwen 2.5 72B - HF</option>
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-slate-400">
                                <ChevronDown size={14} />
                            </div>
                        </div>
                        {user?.role === 'common_user' && (
                            <p className="text-[10px] text-ocean-400 mt-1">Upgrade role to unlock more models</p>
                        )}
                    </div>

                    <div>
                        <label className="block text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Retriever Engine</label>
                        <div className="relative">
                            <select
                                value={retrieverType}
                                onChange={(e) => setRetrieverType(e.target.value)}
                                className="w-full bg-ocean-900 border border-ocean-700 text-slate-200 rounded-xl px-4 py-2.5 appearance-none focus:outline-none focus:border-ocean-400 disabled:opacity-50"
                                disabled={user?.role === 'common_user'}
                            >
                                <option value="similarity">Similarity (Fast)</option>
                                {user?.role !== 'common_user' && <option value="mmr">MMR (Diverse)</option>}
                                {user?.role !== 'common_user' && <option value="hybrid">Hybrid (Contextual)</option>}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-slate-400">
                                <ChevronDown size={14} />
                            </div>
                        </div>
                    </div>

                    <div>
                        <label className="block text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2 flex justify-between">
                            <span>Top-K Chunks</span>
                            <span className="text-ocean-400">{topK}</span>
                        </label>
                        <input
                            type="range"
                            min="1"
                            max={user?.role === 'common_user' ? "3" : user?.role === 'student' ? "5" : "10"}
                            step="1"
                            value={topK}
                            onChange={(e) => setTopK(e.target.value)}
                            className="w-full accent-ocean-400"
                        />
                        <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                            <span>1</span>
                            <span>{user?.role === 'common_user' ? "3" : user?.role === 'student' ? "5" : "10"} (Max)</span>
                        </div>
                    </div>
                </div>

                <div className="mt-auto p-4 bg-ocean-900/50 border border-ocean-700/50 rounded-xl">
                    <p className="text-xs text-slate-400 leading-relaxed">
                        <span className="font-semibold text-ocean-400">Note:</span> Your current role is <span className="uppercase text-slate-200">{user?.role}</span>. Generation complexity and available parameters are automatically adjusted based on your permissions.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Chat;
