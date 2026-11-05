// Shared UI: navigation bar, footer, backend status pill, small helpers.
const NAV = [["index.html", "Home"], ["generator.html", "Generator"], ["comparison.html", "Compare"],
             ["training.html", "Training"], ["evaluation.html", "Evaluation"], ["about.html", "About"]];

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  kids.flat().forEach(c => n.append(c instanceof Node ? c : document.createTextNode(c ?? "")));
  return n;                                    // textContent only -> no HTML injection
}
const fmt = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? "–" : (x * 100).toFixed(0) + "%");

function downloadDataUrl(url, filename) {
  const a = document.createElement("a"); a.href = url; a.download = filename; document.body.append(a); a.click(); a.remove();
}

function buildChrome() {
  const page = location.pathname.split("/").pop() || "index.html";
  const nav = document.getElementById("nav");
  if (nav) {
    nav.className = "nav";
    nav.innerHTML = `<div class="wrap"><a class="brand" href="index.html"><span class="logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/></svg></span>FashionGen AI</a>
      <button class="burger" aria-label="Open menu" aria-expanded="false">☰</button>
      <nav class="links" aria-label="Main"></nav><span class="pill" id="api-pill"><i class="dot"></i><span>Checking backend…</span></span></div>`;
    const links = nav.querySelector(".links");
    NAV.forEach(([href, label]) => links.append(el("a", { href, class: page === href ? "active" : "" }, label)));
    const burger = nav.querySelector(".burger");
    burger.addEventListener("click", () => { const o = links.classList.toggle("open"); burger.setAttribute("aria-expanded", o); });
  }
  const foot = document.getElementById("footer");
  if (foot) foot.innerHTML = `<footer><div class="wrap">FashionGen AI · P14 Text-to-Conditional-GAN Bridge · College Generative AI mini-project.
    Fashion-MNIST © Zalando Research (MIT licence). Generated images are 28×28 research samples, not product photos.</div></footer>`;
  API.health().then(h => {
    const pill = document.getElementById("api-pill"); if (!pill) return;
    pill.querySelector(".dot").className = "dot " + (h.cgan_ready ? "ok" : "bad");
    pill.querySelector("span").textContent = h.cgan_ready ? "Backend online" : "Backend online · model not trained";
  }).catch(() => {
    const pill = document.getElementById("api-pill"); if (!pill) return;
    pill.querySelector(".dot").className = "dot bad"; pill.querySelector("span").textContent = "Backend offline";
  });
}
document.addEventListener("DOMContentLoaded", buildChrome);

// ---- home page: example gallery built from REAL cGAN output ----
async function loadHomeGallery() {
  const box = document.getElementById("home-gallery"); if (!box) return;
  try {
    const { samples } = await API.samples();
    box.replaceChildren(...samples.map(s => el("figure", {}, el("img", { src: s.data_url, alt: `cGAN generated ${s.category}`, loading: "lazy" }), el("figcaption", {}, s.category))));
  } catch (e) { box.replaceChildren(el("div", { class: "empty", style: "grid-column:1/-1" }, "Example images appear once the backend is running with trained weights. " + e.message)); }
}
document.addEventListener("DOMContentLoaded", loadHomeGallery);
