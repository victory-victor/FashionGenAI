// Thin wrapper around fetch(): every page talks to FastAPI through these functions.
const API = (() => {
  function getUrl(path) {
    const base = (window.FASHIONGEN_CONFIG.API_BASE || "").replace(/\/+$/, "");
    const cleanPath = path.startsWith("/") ? path : `/${path}`;
    return base ? `${base}${cleanPath}` : cleanPath;
  }

  async function request(path, options = {}, timeoutMs = 120000, retries = 2) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const fullUrl = getUrl(path);
      const res = await fetch(fullUrl, {
        headers: { "Content-Type": "application/json" }, signal: ctrl.signal, ...options });

      // If Render returns 502/503/504 (cold start wakeup), retry up to 2 times
      if ((res.status === 502 || res.status === 503 || res.status === 504) && retries > 0) {
        clearTimeout(timer);
        await new Promise(r => setTimeout(r, 3500));
        return request(path, options, timeoutMs, retries - 1);
      }

      if (!res.ok) {
        let msg = `Request failed (${res.status})`;
        try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
        const err = new Error(msg); err.status = res.status; throw err;
      }
      return res;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The server took too long to respond. On free hosting the first request can take a minute while it wakes up - try again.");
      if (e instanceof TypeError) throw new Error("Cannot reach the backend. Check that it is running and that the API address in js/config.js is correct.");
      throw e;
    } finally { clearTimeout(timer); }
  }
  const json = async (path, opts, t) => (await request(path, opts, t)).json();
  const post = (path, body, t) => json(path, { method: "POST", body: JSON.stringify(body) }, t);

  return {
    get BASE() { return window.FASHIONGEN_CONFIG.API_BASE; },
    getUrl,
    health: () => json("/api/health", {}, 15000),
    categories: () => json("/api/categories"),
    parsePrompt: (prompt, strategy = "few_shot", compare = false) => post("/api/parse-prompt", { prompt, strategy, compare }),
    generate: (payload) => post("/api/generate", payload),
    generateVae: (payload) => post("/api/generate-vae", payload),
    samples: () => json("/api/samples"),
    compare: (classId, count, seed) => json(`/api/compare?class_id=${classId}&count=${count}&seed=${seed}`),
    trainingMetrics: () => json("/api/training-metrics"),
    evaluation: () => json("/api/evaluation"),
    modelInfo: () => json("/api/model-info"),
    zipBlob: async (batchId) => (await request(`/api/batch/${batchId}/zip`)).blob(),
  };
})();
