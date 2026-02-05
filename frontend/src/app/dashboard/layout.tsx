'use client';

import { ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard,
    Brain,
    TrendingUp,
    Users,
    BarChart3,
    Target,
    Settings,
    Bell,
    Search,
    Menu,
    X,
    ChevronDown,
    LogOut,
} from 'lucide-react';
import { useState } from 'react';

const navigation = [
    { name: 'Overview', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Creative Predictor', href: '/dashboard/creative', icon: Brain },
    { name: 'ROAS Optimizer', href: '/dashboard/roas', icon: TrendingUp },
    { name: 'Client Health', href: '/dashboard/churn', icon: Users },
    { name: 'Attribution', href: '/dashboard/attribution', icon: BarChart3 },
    { name: 'Audience', href: '/dashboard/audience', icon: Target },
];

function Sidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
    const pathname = usePathname();

    return (
        <>
            {/* Mobile backdrop */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-40 lg:hidden"
                    onClick={onClose}
                />
            )}

            {/* Sidebar */}
            <aside
                className={`
          fixed top-0 left-0 z-50 h-full w-64 bg-neutral-900 border-r border-neutral-800
          transform transition-transform duration-300 ease-in-out
          lg:translate-x-0 lg:static lg:z-0
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
            >
                {/* Logo */}
                <div className="flex items-center justify-between h-16 px-4 border-b border-neutral-800">
                    <Link href="/dashboard" className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                            <span className="text-white font-bold text-sm">AC</span>
                        </div>
                        <span className="text-lg font-semibold text-white">AdClass AI</span>
                    </Link>
                    <button onClick={onClose} className="lg:hidden text-neutral-400 hover:text-white">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Navigation */}
                <nav className="p-4 space-y-1">
                    {navigation.map((item) => {
                        const isActive = pathname === item.href ||
                            (item.href !== '/dashboard' && pathname?.startsWith(item.href));

                        return (
                            <Link
                                key={item.name}
                                href={item.href}
                                className={`
                  flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                  transition-colors duration-200
                  ${isActive
                                        ? 'bg-primary-500/10 text-primary-400'
                                        : 'text-neutral-400 hover:text-white hover:bg-neutral-800'
                                    }
                `}
                            >
                                <item.icon className="w-5 h-5" />
                                {item.name}
                            </Link>
                        );
                    })}
                </nav>

                {/* Bottom section */}
                <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-neutral-800">
                    <Link
                        href="/settings"
                        className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
                    >
                        <Settings className="w-5 h-5" />
                        Settings
                    </Link>
                </div>
            </aside>
        </>
    );
}

function Header({ onMenuClick }: { onMenuClick: () => void }) {
    return (
        <header className="h-16 border-b border-neutral-800 bg-neutral-900/80 backdrop-blur-sm sticky top-0 z-30">
            <div className="h-full px-4 flex items-center justify-between gap-4">
                {/* Left section */}
                <div className="flex items-center gap-4">
                    <button
                        onClick={onMenuClick}
                        className="lg:hidden text-neutral-400 hover:text-white"
                    >
                        <Menu className="w-5 h-5" />
                    </button>

                    {/* Search */}
                    <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-neutral-800 border border-neutral-700 w-64">
                        <Search className="w-4 h-4 text-neutral-500" />
                        <input
                            type="text"
                            placeholder="Search..."
                            className="bg-transparent border-none outline-none text-sm text-white placeholder:text-neutral-500 w-full"
                        />
                        <kbd className="hidden md:inline-flex px-1.5 py-0.5 text-xs bg-neutral-700 rounded text-neutral-400">
                            ⌘K
                        </kbd>
                    </div>
                </div>

                {/* Right section */}
                <div className="flex items-center gap-4">
                    {/* Notifications */}
                    <button className="relative text-neutral-400 hover:text-white">
                        <Bell className="w-5 h-5" />
                        <span className="absolute -top-1 -right-1 w-4 h-4 bg-danger-500 rounded-full text-[10px] font-medium text-white flex items-center justify-center">
                            3
                        </span>
                    </button>

                    {/* User menu */}
                    <button className="flex items-center gap-2 px-2 py-1 rounded-lg hover:bg-neutral-800 transition-colors">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center text-white text-sm font-medium">
                            JD
                        </div>
                        <span className="hidden md:block text-sm text-white">John Doe</span>
                        <ChevronDown className="w-4 h-4 text-neutral-500" />
                    </button>
                </div>
            </div>
        </header>
    );
}

export default function DashboardLayout({ children }: { children: ReactNode }) {
    const [sidebarOpen, setSidebarOpen] = useState(false);

    return (
        <div className="min-h-screen bg-neutral-950 flex">
            <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

            <div className="flex-1 flex flex-col min-w-0">
                <Header onMenuClick={() => setSidebarOpen(true)} />

                <main className="flex-1 p-4 md:p-6 lg:p-8 overflow-y-auto">
                    {children}
                </main>
            </div>
        </div>
    );
}
