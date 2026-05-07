/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Thesis-light palette. Variable names retained for backwards
        // compatibility with existing class usage; values shifted from
        // neon "cyber" tones to print-friendly tokens.
        'cyber-dark': '#f6f8fb',     // page background
        'cyber-darker': '#ffffff',   // panel / header surface
        'cyber-blue': '#2563eb',     // primary accent (blue-600)
        'cyber-purple': '#7c3aed',   // secondary accent (violet-600)
        'cyber-green': '#16a34a',    // success (green-600)
        'cyber-yellow': '#d97706',   // warning (amber-600)
        'cyber-red': '#dc2626',      // danger (red-600)
        'cyber-orange': '#ea580c',   // alt warning (orange-600)
      }
    },
  },
  plugins: [],
}
