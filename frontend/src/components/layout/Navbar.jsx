import { NavLink } from 'react-router-dom';
import { LogOut, User, MessageSquare, BarChart2, TestTube2 } from 'lucide-react';
import useAuthStore from '../../store/authStore';

const Navbar = () => {
    const { user, logout } = useAuthStore();
    const canViewResults = ['admin', 'researcher', 'student'].includes(user?.role);

    const handleLogout = () => {
        logout();
        window.location.href = '/login';
    };

    const navLinks = [
        { to: '/chat', icon: MessageSquare, label: 'Chat Engine' },
    ];

    const rightLinks = [
        { to: '/dashboard', icon: BarChart2, label: 'Analytics' },
        ...(canViewResults ? [{ to: '/eval-results', icon: TestTube2, label: 'Research Results' }] : []),
    ];

    return (
        <header className="h-16 px-6 bg-ocean-800 border-b border-ocean-700 flex flex-row items-center justify-between shrink-0">
            {/* Left: Logo + Primary Nav */}
            <div className="flex items-center gap-6">
                <NavLink to="/chat" className="flex items-center gap-2 shrink-0">
                    <span className="text-xl font-bold text-ocean-400 tracking-tight leading-none">🌊 Ocean<span className="text-white">RAG</span></span>
                </NavLink>

                <nav className="flex items-center gap-1">
                    {navLinks.map(({ to, icon: Icon, label }) => (
                        <NavLink
                            key={to}
                            to={to}
                            className={({ isActive }) =>
                                `flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${isActive
                                    ? 'bg-ocean-700/60 text-ocean-400 border border-ocean-600'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-ocean-700/30'
                                }`
                            }
                        >
                            <Icon size={16} className="shrink-0" />
                            <span>{label}</span>
                        </NavLink>
                    ))}
                </nav>
            </div>

            {/* Right: Secondary Nav + Profile + Logout */}
            <div className="flex items-center gap-2">
                {/* Analytics & Research Results */}
                <nav className="flex items-center gap-1 mr-4">
                    {rightLinks.map(({ to, icon: Icon, label }) => (
                        <NavLink
                            key={to}
                            to={to}
                            className={({ isActive }) =>
                                `flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${isActive
                                    ? 'bg-ocean-700/60 text-ocean-400 border border-ocean-600'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-ocean-700/30'
                                }`
                            }
                        >
                            <Icon size={16} className="shrink-0" />
                            <span className="hidden md:inline">{label}</span>
                        </NavLink>
                    ))}
                </nav>

                {/* Divider */}
                <div className="h-6 w-px bg-ocean-700 mx-1"></div>

                {/* Profile */}
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-ocean-900 border border-ocean-700">
                    <div className="w-6 h-6 rounded-full bg-ocean-700 flex items-center justify-center">
                        <User size={14} className="text-ocean-400" />
                    </div>
                    <div className="flex flex-col">
                        <span className="text-sm font-medium text-slate-200 leading-tight">{user?.username}</span>
                        <span className="text-[10px] text-ocean-400 uppercase tracking-widest leading-tight">{user?.role}</span>
                    </div>
                </div>

                {/* Logout */}
                <button
                    onClick={handleLogout}
                    className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors"
                    title="Logout"
                >
                    <LogOut size={20} />
                </button>
            </div>
        </header>
    );
};

export default Navbar;
