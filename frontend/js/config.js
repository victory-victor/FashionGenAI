// Backend URL. Edit PRODUCTION_API after deploying the FastAPI service (e.g. on Render).
window.FASHIONGEN_CONFIG = (function () {
  const PRODUCTION_API = "https://fashiongenai-c7hr.onrender.com/";
  const h = location.hostname, p = location.port;
  let base;
  if (location.pathname.startsWith("/app")) base = "";              // served by FastAPI itself
  else if (h === "localhost" || h === "127.0.0.1" || h === "") base = "http://localhost:8000";
  else base = PRODUCTION_API;
  return { API_BASE: base };
})();
