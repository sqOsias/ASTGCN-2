/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-dark': '#0a0e17',
        'cyber-darker': '#060910',
        'cyber-blue': '#00d4ff',
        'cyber-purple': '#a855f7',
        'cyber-green': '#22c55e',
        'cyber-yellow': '#eab308',
        'cyber-red': '#ef4444',
        'cyber-orange': '#f97316',
      }
    },
  },
  plugins: [],
}
