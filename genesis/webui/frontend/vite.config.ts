import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/list_databases': 'http://localhost:8000',
      '/get_programs': 'http://localhost:8000',
      '/get_meta_files': 'http://localhost:8000',
      '/get_meta_content': 'http://localhost:8000',
      '/download_meta_pdf': 'http://localhost:8000',
    },
  },
});
