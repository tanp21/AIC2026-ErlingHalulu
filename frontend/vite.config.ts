import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// @types/node is not a dependency, and this is the only place the config reads the shell.
declare const process: { env: Record<string, string | undefined> };

// In production HAProxy authenticates and sets X-AIC-User; nothing does that in front of
// `vite dev`, so every backend call would 401. Set AIC_DEV_USER to one of AIC_ALLOWED_USERS
// to stand in for it. Dev-server only — `vite build` never reads this block.
const backend = {
  target: process.env.AIC_DEV_BACKEND ?? "http://127.0.0.1:8000",
  headers: process.env.AIC_DEV_USER ? { "X-AIC-User": process.env.AIC_DEV_USER } : undefined,
};
const routes = ["/search", "/compare", "/thumb", "/frame", "/video", "/transcript",
  "/shots", "/clip", "/indexes", "/profiles", "/session", "/api"];

export default defineConfig({
  base: "/static/",
  plugins: [react()],
  build: {
    outDir: "../static/dist",
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(routes.map((route) => [route, backend])),
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts"
  }
});
