import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import apiClient from '../api/client';
import { Anchor } from 'lucide-react';

const Register = () => {
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        role: 'student'
    });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const navigate = useNavigate();

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await apiClient.post('/auth/register', formData);
            // On success, redirect to login
            navigate('/login');
        } catch (err) {
            setError(err.response?.data?.detail || 'Registration failed.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-ocean-900 px-4 relative overflow-hidden">
            {/* Decorative background blur */}
            <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-ocean-400/10 rounded-full blur-[120px] pointer-events-none"></div>
            <div className="absolute bottom-1/4 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-[120px] pointer-events-none"></div>

            <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: [0.25, 0.4, 0.2, 1] }}
                className="w-full max-w-md bg-ocean-800/80 backdrop-blur-xl border border-ocean-700 p-8 rounded-2xl shadow-2xl relative z-10"
            >
                <div className="flex flex-col items-center mb-8">
                    <div className="w-16 h-16 bg-ocean-700 rounded-2xl flex items-center justify-center mb-4 border border-ocean-400/20 shadow-lg">
                        <Anchor size={32} className="text-ocean-400" />
                    </div>
                    <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Create Account</h1>
                    <p className="text-slate-400 text-sm">Join OceanRAG platform</p>
                </div>

                {error && (
                    <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex items-center gap-2">
                        <span className="block w-2 h-2 rounded-full bg-red-500 shrink-0"></span>
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Username</label>
                        <input
                            type="text"
                            name="username"
                            value={formData.username}
                            onChange={handleChange}
                            className="w-full bg-ocean-900/50 border border-ocean-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-ocean-400/50 focus:border-ocean-400/50 transition-all placeholder:text-slate-600"
                            placeholder="johndoe"
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Email</label>
                        <input
                            type="email"
                            name="email"
                            value={formData.email}
                            onChange={handleChange}
                            className="w-full bg-ocean-900/50 border border-ocean-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-ocean-400/50 focus:border-ocean-400/50 transition-all placeholder:text-slate-600"
                            placeholder="you@example.com"
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Password</label>
                        <input
                            type="password"
                            name="password"
                            value={formData.password}
                            onChange={handleChange}
                            className="w-full bg-ocean-900/50 border border-ocean-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-ocean-400/50 focus:border-ocean-400/50 transition-all placeholder:text-slate-600"
                            placeholder="••••••••"
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Role</label>
                        <div className="relative">
                            <select
                                name="role"
                                value={formData.role}
                                onChange={handleChange}
                                className="w-full bg-ocean-900/50 border border-ocean-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-ocean-400/50 focus:border-ocean-400/50 transition-all appearance-none"
                            >
                                <option value="student">Student</option>
                                <option value="researcher">Researcher</option>
                                <option value="common_user">Common User</option>
                                {/* Admin role is assigned via backend only for security */}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-slate-400">
                                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" /></svg>
                            </div>
                        </div>
                    </div>
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-ocean-400 hover:bg-ocean-400/90 text-ocean-900 font-extrabold tracking-wide py-3 rounded-xl transition-all mt-4 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(0,210,255,0.3)]"
                    >
                        {loading ? 'Creating Account...' : 'Register'}
                    </button>
                </form>

                <p className="mt-8 text-center text-sm text-slate-400">
                    Already have an account?{' '}
                    <Link to="/login" className="text-ocean-400 hover:underline hover:text-ocean-300 font-medium">
                        Sign in here
                    </Link>
                </p>
            </motion.div>
        </div>
    );
};

export default Register;
