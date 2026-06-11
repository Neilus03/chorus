import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { chorusFilesPlugin } from './vite.chorusFilesPlugin';

export default defineConfig({
  plugins: [react(), chorusFilesPlugin()],
  css: {
    postcss: {},
  },
  server: {
    watch: {
      usePolling: true,
      interval: 1000,
    },
  },
  test: {
    environment: 'node',
    globals: true,
    exclude: ['node_modules/**', 'dist/**', 'tests/e2e/**'],
  },
});
