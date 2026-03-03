import { LogOut, User } from 'lucide-react';
import useAuthStore from '../../store/authStore';

const Navbar = () => {
    const { user, logout } = useAuthStore();

    const handleLogout = () => {
        logout();
        window.location.href = '/login';
    };

    return (
        <header className="h-16 px-6 bg-ocean-800 border-b border-ocean-700 flex flex-row items-center justify-between shrink-0">
            <div className="flex items-center">
                {/* Breadcrumb or View Title could go here based on route */}
            </div>

            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-ocean-900 border border-ocean-700">
                    <div className="w-6 h-6 rounded-full bg-ocean-700 flex items-center justify-center">
                        <User size={14} className="text-ocean-400" />
                    </div>
                    <div className="flex flex-col">
                        <span className="text-sm font-medium text-slate-200 leading-tight">{user?.username}</span>
                        <span className="text-[10px] text-ocean-400 uppercase tracking-widest leading-tight">{user?.role}</span>
                    </div>
                </div>

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
