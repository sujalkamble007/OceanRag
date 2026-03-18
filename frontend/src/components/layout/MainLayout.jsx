import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';

const MainLayout = () => {
    return (
        <div className="flex flex-col h-screen bg-ocean-900 text-slate-300 font-sans overflow-hidden">
            <Navbar />
            <main className="flex-1 overflow-auto p-6">
                <Outlet />
            </main>
        </div>
    );
};

export default MainLayout;
