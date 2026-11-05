// Generator page: parse text -> generate with cGAN -> show CLIP scores -> download.
const state = { batchId: null };
const $ = (id) => document.getElementById(id);
const EXAMPLES = ["Show me four ankle boots.", "Generate three sneakers.", "Create five handbags.", "Show me two dresses."];

async function init() {
  EXAMPLES.forEach(t => $("examples").append(el("button", { class: "chip", type: "button", onclick: () => { $("prompt").value = t; $("prompt").focus(); } }, t)));
  try {
    const { categories, max_images } = await API.categories();
    categories.forEach(c => $("category").append(el("option", { value: c.class_id }, c.name)));
    for (let i = 1; i <= max_images; i++) $("count").append(el("option", { value: i }, String(i)));
  } catch (e) { showStatus(e.message, true); }
  $("gen-form").addEventListener("submit", onSubmit);
}

function showStatus(html, isError = false, steps = null) {
  const s = $("status"); s.className = "status show" + (isError ? " error" : "");
  s.replaceChildren(el("div", {}, html));
  if (steps) { s.prepend(el("span", { class: "spinner", style: "margin-right:.6rem;vertical-align:middle" })); s.append(steps); }
}
function stepList(active, done) {
  const names = ["Understanding your request", "Generating images with cGAN", "Scoring images with CLIP"];
  return el("ul", { class: "steps" }, names.map((n, i) => el("li", { class: i < done ? "done" : i === active ? "active" : "" }, n)));
}

async function onSubmit(ev) {
  ev.preventDefault();
  const text = $("prompt").value.trim(), catSel = $("category").value, cntSel = $("count").value;
  if (!text && catSel === "") return showStatus("Type a request or pick a category first.", true);
  const btn = $("gen-btn"); btn.disabled = true; btn.textContent = "Working…";
  $("results").replaceChildren(el("div", { class: "empty" }, "Generating…"));
  $("summary").replaceChildren(); $("bert-box").replaceChildren();
  try {
    let parsed = null, classId, count;
    if (catSel === "") {
      showStatus("Starting…", false, stepList(0, 0));
      parsed = await API.parsePrompt(text);                       // step 1: LLM / parser
      classId = parsed.class_id; count = cntSel ? +cntSel : parsed.count;
    } else { classId = +catSel; count = cntSel ? +cntSel : 1; }
    showStatus("Generating…", false, stepList(1, 1));
    const data = await API.generate({ class_id: classId, count, evaluate: true });   // steps 2+3
    data.parsed = parsed || { source: "category selector", category: data.category, class_id: classId, count };
    data.bert_check = parsed ? parsed.bert_check : null;
    render(data);
    $("status").className = "status";
  } catch (e) {
    showStatus(e.message, true);
    $("results").replaceChildren(el("div", { class: "empty" }, "Nothing was generated. Fix the issue above and try again."));
  } finally { btn.disabled = false; btn.textContent = "Generate images"; }
}

function render(d) {
  state.batchId = d.batch_id;
  const cs = d.clip_summary, p = d.parsed;
  $("summary").replaceChildren(
    el("span", { class: "badge" }, "Category: " + d.category + " (id " + d.class_id + ")"),
    el("span", { class: "badge blue" }, "Quantity: " + d.count),
    el("span", { class: "badge" }, "Parsed by: " + p.source),
    el("span", { class: "badge ok" }, `Generated in ${d.timing_ms.generation} ms`),
    cs ? el("span", { class: "badge blue" }, "Mean CLIP similarity: " + fmt(cs.mean_similarity)) :
         el("span", { class: "badge warn" }, "CLIP unavailable" + (d.clip_status.error ? " – " + d.clip_status.error : " (disabled)")));
  const bc = d.bert_check;
  $("bert-box").replaceChildren(bc ? el("div", { class: "note" }, `${bc.model} cross-check (${bc.method}): top guess "${bc.top[0].category}" (${pct(bc.top[0].score)}) – ` + (bc.top[0].class_id === d.class_id ? "agrees with the parser." : "differs from the parser.")) : "");

  const tiles = d.images.map(im => {
    const c = im.clip;
    return el("div", { class: "tile" },
      el("img", { src: im.data_url, alt: `Generated ${d.category} ${im.index}` }),
      el("div", { class: "meta" },
        el("b", {}, `${d.category} #${im.index}`),
        c ? [el("span", {}, `CLIP similarity ${fmt(c.similarity)}`),
             el("div", { class: "score-bar", title: "CLIP probability for requested class" }, el("i", { style: `width:${Math.min(100, c.class_match_prob * 100)}%` })),
             el("span", {}, `Class match ${pct(c.class_match_prob)} · CLIP top-1: ${c.top1_category}`)] : el("span", {}, "No CLIP score"),
        el("button", { class: "btn sm ghost", type: "button", onclick: () => downloadDataUrl(im.data_url, im.filename) }, "Download PNG")));
  });
  const zipBtn = el("button", { class: "btn sm", type: "button", style: "margin-bottom:1rem", onclick: onZip }, "Download all (.zip)");
  $("results").replaceChildren(zipBtn, el("div", { class: "gallery" }, tiles),
    el("p", { class: "note" }, "CLIP similarity shows image–text agreement. It is not a verified accuracy score; 28×28 grayscale images are far outside CLIP's training data."));
}

async function onZip(ev) {
  const b = ev.currentTarget; b.disabled = true;
  try { const blob = await API.zipBlob(state.batchId); const u = URL.createObjectURL(blob);
        downloadDataUrl(u, `fashiongen_${state.batchId.slice(0, 8)}.zip`); setTimeout(() => URL.revokeObjectURL(u), 5000); }
  catch (e) { showStatus(e.message, true); } finally { b.disabled = false; }
}
document.addEventListener("DOMContentLoaded", init);
