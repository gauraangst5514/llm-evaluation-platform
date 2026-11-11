import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In Docker Compose the API is reachable as the `api` service; locally it is on
// localhost. The dev server proxies /api/* so the browser never needs CORS.
const apiTarget = process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
