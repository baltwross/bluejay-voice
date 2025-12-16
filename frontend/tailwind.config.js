/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Terminator HUD Color Palette
        terminator: {
          red: '#ff0033',
          'red-glow': '#ff003366',
          cyan: '#00ffff',
          'cyan-glow': '#00ffff44',
          dark: '#0a0a0f',
          darker: '#050508',
          surface: '#12121a',
          border: '#1f1f2e',
          text: '#e0e0e8',
          'text-dim': '#6b6b7b',
        },
      },
      fontFamily: {
        mono: ['Terminator', 'JetBrains Mono', 'monospace'],
        display: ['Terminator', 'sans-serif'],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'scan-line': 'scan-line 3s linear infinite',
        'flicker': 'flicker 0.15s infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '0.6', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.02)' },
        },
        'scan-line': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        'flicker': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.95' },
        },
      },
      boxShadow: {
        'glow-red': '0 0 20px rgba(255, 0, 51, 0.5)',
        'glow-cyan': '0 0 20px rgba(0, 255, 255, 0.3)',
        'glow-red-lg': '0 0 40px rgba(255, 0, 51, 0.6)',
        'glow-cyan-lg': '0 0 40px rgba(0, 255, 255, 0.4)',
      },
    },
  },
  plugins: [],
};

