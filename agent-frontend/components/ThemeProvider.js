'use client';

import { createContext, useContext, useEffect, useState } from 'react';

console.log("API BASE URL:", process.env.NEXT_PUBLIC_API_URL)

const ThemeContext = createContext({ theme: 'light', toggle: () => { } });

export function ThemeProvider({ children }) {
    const [theme, setTheme] = useState('light');

    // On first mount, read saved preference or system preference
    useEffect(() => {
        const stored = localStorage.getItem('theme');
        if (stored === 'dark' || stored === 'light') {
            setTheme(stored);
        } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            setTheme('dark');
        }
    }, []);

    // Apply class to <html> element
    useEffect(() => {
        const root = document.documentElement;
        if (theme === 'dark') {
            root.classList.add('dark');
        } else {
            root.classList.remove('dark');
        }
        localStorage.setItem('theme', theme);
    }, [theme]);

    const toggle = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

    return (
        <ThemeContext.Provider value={{ theme, toggle }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    return useContext(ThemeContext);
}
