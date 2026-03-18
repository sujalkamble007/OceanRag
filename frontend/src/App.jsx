import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import useAuthStore from './store/authStore';

// Layouts
import MainLayout from './components/layout/MainLayout';

// Pages
import Landing from './pages/Landing';
import Login from './pages/Login';
import Register from './pages/Register';
import Chat from './pages/Chat';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import Experiments from './pages/Experiments';
import EvalResults from './pages/EvalResults';

const ProtectedRoute = ({ children, allowedRoles }) => {
  const { isAuthenticated, user } = useAuthStore();

  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (allowedRoles && !allowedRoles.includes(user?.role)) return <Navigate to="/" replace />;

  return children;
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected Routes inside MainLayout */}
        <Route element={<ProtectedRoute><MainLayout /></ProtectedRoute>}>
          <Route path="/chat" element={<Chat />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<History />} />

          {/* Admin/Researcher/Student only */}
          <Route
            path="/experiments"
            element={
              <ProtectedRoute allowedRoles={['admin', 'researcher', 'student']}>
                <Experiments />
              </ProtectedRoute>
            }
          />
          <Route
            path="/eval-results"
            element={
              <ProtectedRoute allowedRoles={['admin', 'researcher', 'student']}>
                <EvalResults />
              </ProtectedRoute>
            }
          />
        </Route>
        
        {/* Catch-all redirect */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
