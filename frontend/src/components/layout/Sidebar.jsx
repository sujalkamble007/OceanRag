import { NavLink } from 'react-router-dom';
import { MessageSquare, BarChart2, Clock, FlaskConical, TestTube2 } from 'lucide-react';
import useAuthStore from '../../store/authStore';

const Sidebar = () => {
    const { user } = useAuthStore();
    const canViewExperiments = ['admin', 'researcher', 'student'].includes(user?.role);

    const links = [
        { to: '/chat', icon: MessageSquare, label: 'Chat Engine' },
        { to: '/dashboard', icon: BarChart2, label: 'Analytics' },
        { to: '/history', icon: Clock, label: 'Q&A History' },
    ];

    if (canViewExperiments) {
        links.push({ to: '/experiments', icon: FlaskConical, label: 'Experiments' });
        links.push({ to: '/eval-results', icon: TestTube2, label: 'Research Results' });
    }

    return (
        <div className="w-64 bg-ocean-800 border-r border-ocean-700 flex flex-col h-full">
            <div className="p-6">
                <h1 className="text-2xl font-bold text-ocean-400 tracking-tight flex items-center gap-2">
                    🌊 Ocean<span className="text-white">RAG</span>
                </h1>
                <p className="text-xs text-slate-400 mt-1 uppercase tracking-widest font-semibold">Deep-Sea Governance</p>
            </div>

            <nav className="flex-1 px-4 space-y-2 mt-4">
                {links.map((link) => {
                    const Icon = link.icon;
                    return (
                        <NavLink
                            key={link.to}
                            to={link.to}
                            className={({ isActive }) =>
                                `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${isActive
                                    ? 'bg-ocean-700/50 text-ocean-400 font-medium border border-ocean-700 shadow-sm'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-ocean-700/30'
                                }`
                            }
                        >
                            <Icon size={20} className="shrink-0" />
                            <span>{link.label}</span>
                        </NavLink>
                    );
                })}
            </nav>

            <div className="p-4 m-4 rounded-xl bg-ocean-900 border border-ocean-700 text-sm">
                <p className="text-slate-400">Phase 5 Release</p>
                <p className="text-xs text-slate-500 mt-1">v5.0.0</p>
            </div>
        </div>
    );
};

export default Sidebar;
