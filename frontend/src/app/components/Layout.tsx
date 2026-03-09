import { Outlet, Link, useLocation } from 'react-router';
import { 
  FileText, 
  Database, 
  Search, 
  BarChart3, 
  Cpu,
  MessageSquare,
  Settings,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { cn } from './ui/utils';
import { useState } from 'react';
import { Button } from './ui/button';

const navItems = [
  { path: '/', label: 'RAG Chat', icon: MessageSquare },
  { path: '/chunking', label: 'Chunking', icon: FileText },
  { path: '/index', label: 'Index', icon: Database },
  { path: '/retrieval', label: 'Retrieval', icon: Search },
  { path: '/eval', label: 'End-to-End Eval', icon: BarChart3 },
  {
    path: '/component-eval',
    label: 'Component Eval',
    icon: Cpu,
    children: [
      { path: '/component-eval?section=chunk', label: 'Chunk Evaluation' },
      { path: '/component-eval?section=retrieval', label: 'Retrieval Evaluation' },
    ],
  },
];

export default function Layout() {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen bg-slate-50">
      {/* Sidebar */}
      <aside className={cn(
        "bg-slate-900 text-white flex flex-col transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}>
        {/* Logo */}
        <div className="p-4 border-b border-slate-800">
          {!collapsed ? (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
                <Search className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold">RAG Lab</h1>
                <p className="text-xs text-slate-400">Experiment Platform</p>
              </div>
            </div>
          ) : (
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center mx-auto">
              <Search className="w-5 h-5 text-white" />
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            const isSectionActive = location.pathname === '/component-eval' && item.path === '/component-eval';

            return (
              <div key={item.path}>
                <Link
                  to={item.path}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all group",
                    isActive
                      ? "bg-blue-600 text-white"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  )}
                  title={collapsed ? item.label : undefined}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {!collapsed && (
                    <span className="text-sm font-medium">{item.label}</span>
                  )}
                </Link>

                {!collapsed && isSectionActive && item.children?.length ? (
                  <div className="mt-2 ml-8 space-y-1 rounded-lg border border-slate-800/80 bg-slate-900/40 p-1.5">
                    {item.children.map((child: any) => {
                      const childActive = location.search.includes('section=retrieval')
                        ? child.path.includes('section=retrieval')
                        : child.path.includes('section=chunk');
                      return (
                        <Link
                          key={child.path}
                          to={child.path}
                          className={cn(
                            'flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-all',
                            childActive
                              ? 'bg-blue-500/20 text-blue-100 border border-blue-400/30'
                              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/70 border border-transparent'
                          )}
                        >
                          <span className={cn('h-1.5 w-1.5 rounded-full', childActive ? 'bg-blue-300' : 'bg-slate-500')} />
                          <span>{child.label}</span>
                        </Link>
                      );
                    })}
                  </div>
                ) : null}
              </div>
            );
          })}
        </nav>

        {/* Settings & Collapse */}
        <div className="p-3 border-t border-slate-800 space-y-1">
          <button className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-slate-300 hover:bg-slate-800 hover:text-white transition-all w-full">
            <Settings className="w-5 h-5 flex-shrink-0" />
            {!collapsed && <span className="text-sm font-medium">Settings</span>}
          </button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setCollapsed(!collapsed)}
            className="w-full text-slate-400 hover:text-white hover:bg-slate-800"
          >
            {collapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <>
                <ChevronLeft className="w-4 h-4 mr-2" />
                <span className="text-xs">Collapse</span>
              </>
            )}
          </Button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
