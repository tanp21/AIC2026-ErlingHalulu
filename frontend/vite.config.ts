import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/static/",
  plugins: [react()],
  build: {
    outDir: "../static/dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/search": "http://127.0.0.1:8000",
      "/compare": "http://127.0.0.1:8000",
      "/thumb": "http://127.0.0.1:8000",
      "/frame": "http://127.0.0.1:8000",
      "/video": "http://127.0.0.1:8000",
      "/indexes": "http://127.0.0.1:8000"
    }
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts"
  }
});
