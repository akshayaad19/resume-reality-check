import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output goes straight into app/static, which FastAPI serves via
// StaticFiles — one Docker image, one deployed service, no separate
// frontend host or CORS setup needed.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../app/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/analyze": "http://localhost:8000",
    },
  },
});
