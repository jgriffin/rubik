import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// COOP/COEP headers enable SharedArrayBuffer, which is a prerequisite
// for multi-threaded WASM in onnxruntime-web. The OnnxSolver currently
// pins numThreads=1 (single-threaded WASM), so SAB isn't strictly
// required yet — but turning the headers on now means Block D can flip
// the thread count without a separate config change. WebGPU is
// unaffected by these headers.
//
// `optimizeDeps.exclude("onnxruntime-web")` keeps Vite's dev-mode
// pre-bundler away from ort-web's ESM loader, which dynamically imports
// its own .mjs workers and breaks under esbuild bundling.
const ORT_HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    headers: ORT_HEADERS,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  preview: {
    headers: ORT_HEADERS,
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
