import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { Brain, Waves, ExternalLink, Zap, Shield, Search, ArrowRight, Github } from 'lucide-react';
import useAuthStore from '../store/authStore';

const fadeIn = {
  hidden: { opacity: 0, y: 30 },
  visible: (custom = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.8,
      delay: custom * 0.1,
      ease: [0.25, 0.4, 0.2, 1]
    }
  })
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.2 }
  }
};

const Landing = () => {
  const { isAuthenticated } = useAuthStore();
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-ocean-900 text-slate-200 overflow-x-hidden w-full relative">
      {/* Decorative gradient backgrounds */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-ocean-400/20 rounded-full blur-[150px]"></div>
        <div className="absolute top-[40%] right-[-10%] w-96 h-96 bg-blue-500/10 rounded-full blur-[150px]"></div>
        <div className="absolute bottom-[-10%] left-[20%] w-[40rem] h-[40rem] bg-indigo-500/10 rounded-full blur-[150px]"></div>
      </div>

      {/* Navbar */}
      <nav className="relative z-10 border-b border-ocean-800/50 bg-ocean-900/50 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-ocean-800 flex items-center justify-center border border-ocean-700 shadow-lg">
              <Waves size={20} className="text-ocean-400" />
            </div>
            <span className="font-bold text-xl tracking-tight text-white flex items-center">
              Ocean<span className="text-ocean-400">RAG</span>
              {/* <span className="ml-3 px-2 py-0.5 rounded-full bg-ocean-800 border border-ocean-700 text-[10px] text-ocean-400 uppercase tracking-wider font-semibold">
                Beta
              </span> */}
            </span>
          </div>

          <div className="flex items-center gap-6">
            <a href="https://github.com/sujalkamble007/OceanRag" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white transition-colors">
              <Github size={20} />
            </a>
            {isAuthenticated ? (
              <button 
                onClick={() => navigate('/chat')}
                className="px-6 py-2.5 rounded-xl bg-ocean-800 hover:bg-ocean-700 text-white font-medium border border-ocean-600 transition-all shadow-lg"
              >
                Go to Dashboard
              </button>
            ) : (
              <div className="flex items-center gap-3">
                <Link to="/login" className="px-5 py-2.5 text-slate-300 hover:text-white font-medium transition-colors">
                  Sign In
                </Link>
                <Link to="/register" className="px-5 py-2.5 rounded-xl bg-ocean-400 hover:bg-ocean-300 text-ocean-900 font-bold transition-all shadow-[0_0_20px_rgba(100,255,218,0.2)]">
                  Get Started
                </Link>
              </div>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative z-10">
        <div className="max-w-7xl mx-auto px-6 pt-32 pb-24 flex flex-col items-center text-center">
          <motion.div
            initial="hidden"
            animate="visible"
            custom={0}
            variants={fadeIn}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-ocean-700 bg-ocean-800/50 text-ocean-400 text-sm font-medium mb-8 backdrop-blur-md"
          >
            <Zap size={16} />
            <span>Now with LLaMA 70B & Multi-Agent Evaluation</span>
          </motion.div>

          <motion.h1
            initial="hidden"
            animate="visible"
            custom={1}
            variants={fadeIn}
            className="text-6xl md:text-8xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-br from-white via-slate-200 to-slate-400 mb-8 max-w-5xl leading-[1.1]"
          >
            Information governance, <br className="hidden md:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-ocean-400 via-blue-400 to-emerald-400">
              deeply evolved.
            </span>
          </motion.h1>

          <motion.p
            initial="hidden"
            animate="visible"
            custom={2}
            variants={fadeIn}
            className="text-xl text-slate-400 max-w-2xl mb-12 leading-relaxed"
          >
            A production-grade Retrieval-Augmented Generation framework 
            designed to navigate massive document corpora with absolute precision, security, and enterprise compliance.
          </motion.p>

          <motion.div
            initial="hidden"
            animate="visible"
            custom={3}
            variants={fadeIn}
            className="flex flex-col sm:flex-row items-center gap-4"
          >
            <button
               onClick={() => navigate(isAuthenticated ? '/chat' : '/register')}
               className="group flex items-center justify-center gap-3 px-8 py-4 w-full sm:w-auto rounded-2xl bg-ocean-400 hover:bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] hover:bg-ocean-300 text-ocean-900 font-black text-lg transition-all shadow-[0_0_30px_rgba(0,210,255,0.3)] hover:shadow-[0_0_40px_rgba(0,210,255,0.5)] tracking-wide"
            >
              Start Chatting
              <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </button>
            <button
              onClick={() => document.getElementById('features').scrollIntoView({ behavior: 'smooth' })}
              className="px-8 py-4 w-full sm:w-auto rounded-2xl bg-ocean-800/80 hover:bg-ocean-700/80 text-ocean-400 font-bold tracking-wide text-lg border border-ocean-700 backdrop-blur-lg transition-all hover:shadow-[0_0_20px_rgba(0,210,255,0.2)]"
            >
              Explore Architecture
            </button>
          </motion.div>
        </div>

        {/* Features Showcase */}
        <div id="features" className="max-w-7xl mx-auto px-6 py-32 border-t border-ocean-800/50">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeIn}
            className="text-center mb-20"
          >
            <h2 className="text-4xl font-bold text-white mb-6 tracking-tight">Built for serious research.</h2>
            <p className="text-slate-400 max-w-2xl mx-auto text-lg">
              Compare multiple LLMs, chunking strategies, and embedding models head-to-head in our Analytics dashboard.
            </p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            {[
              {
                icon: Search,
                title: "Advanced Retrievers",
                description: "Seamlessly switch between Semantic Similarity, Maximal Marginal Relevance (MMR), and Hybrid search patterns."
              },
              {
                icon: Brain,
                title: "Multi-Model Ops",
                description: "Test LLaMA, Qwen, constraints, and more, comparing their latency vs precise extraction rates in real-time."
              },
              {
                icon: Shield,
                title: "RAGAS Evaluated",
                description: "We automatically compute Faithfulness, Answer Relevancy, Precision@K, and ROUGE-L for every single chunk."
              }
            ].map((feat, i) => (
              <motion.div
                key={i}
                variants={fadeIn}
                className="bg-ocean-800/40 border border-ocean-700/50 p-8 rounded-3xl hover:bg-ocean-800/60 transition-colors group"
              >
                <div className="w-14 h-14 bg-ocean-900 border border-ocean-700 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform group-hover:border-ocean-400/30">
                  <feat.icon size={28} className="text-ocean-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{feat.title}</h3>
                <p className="text-slate-400 leading-relaxed">{feat.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-ocean-800/50 bg-ocean-900 pt-16 pb-8">
        <div className="max-w-7xl mx-auto px-6 flex flex-col items-center justify-center">
          <div className="flex items-center gap-2 mb-6">
            <Waves size={24} className="text-ocean-400" />
            <span className="font-bold text-xl text-white">OceanRAG</span>
          </div>
          <p className="text-slate-500 text-sm">
            © {new Date().getFullYear()} OceanRAG Engine. Built for advanced deep-sea context modeling.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
