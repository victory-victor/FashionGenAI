// Comparison, Training and Evaluation pages. All numbers come from the backend.
const $ = (id) => document.getElementById(id);
const PURPLE = "#9b87f5", BLUE = "#4f8cff", GREEN = "#3ddc97", AMBER = "#ffb454";
let charts = {};

function status(msg, isError = true) {
  const s = $("status"); if (!s) return;
  s.className = "status show" + (isError ? " error" : ""); s.textContent = msg;
}
function line(canvasId, labels, datasets, yTitle) {
  if (charts[canvasId]) charts[canvasId].destroy();
  Chart.defaults.color = "#a3accf"; Chart.defaults.borderColor = "rgba(255,255,255,.08)";
  charts[canvasId] = new Chart($(canvasId), { type: "line",
    data: { labels, datasets: datasets.map(d => ({ ...d, tension: .25, pointRadius: labels.length > 40 ? 0 : 3, borderWidth: 2 })) },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: "Epoch" } }, y: { title: { display: !!yTitle, text: yTitle } } } } });
}
function bar(canvasId, labels, datasets, max) {
  if (charts[canvasId]) charts[canvasId].destroy();
  Chart.defaults.color = "#a3accf"; Chart.defaults.borderColor = "rgba(255,255,255,.08)";
  charts[canvasId] = new Chart($(canvasId), { type: "bar", data: { labels, datasets },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max } } } });
}
const stat = (value, label) => el("div", { class: "card stat" }, el("b", {}, String(value)), el("span", {}, label));
function table(t, head, rows) {
  t.replaceChildren(el("thead", {}, el("tr", {}, head.map(h => el("th", {}, h)))),
                    el("tbody", {}, rows.map(r => el("tr", {}, r.map(c => el("td", {}, c))))));
}

// ---------------- Comparison ----------------
async function initComparison() {
  try {
    const { categories, max_images } = await API.categories();
    categories.forEach(c => $("cmp-cat").append(el("option", { value: c.class_id }, c.name)));
    for (let i = 1; i <= max_images; i++) $("cmp-count").append(el("option", { value: i }, String(i)));
    $("cmp-count").value = Math.min(6, max_images); $("cmp-cat").value = 9;
  } catch (e) { status(e.message); }
  $("cmp-btn").addEventListener("click", runCompare);
  API.modelInfo().then(i => {
    $("arch-gan").textContent = (i.cgan_generator.architecture || "Not trained yet") + "\n\n" + (i.cgan_discriminator.architecture || "");
    $("arch-vae").textContent = i.cvae.architecture || "Not trained yet";
  }).catch(e => { $("arch-gan").textContent = e.message; $("arch-vae").textContent = e.message; });
  runCompare();
}
async function runCompare() {
  const btn = $("cmp-btn"); btn.disabled = true; btn.textContent = "Comparing…"; $("status").className = "status";
  try {
    const r = await API.compare($("cmp-cat").value, $("cmp-count").value, Math.floor(Math.random() * 10000));
    for (const [key, out, sum] of [["cgan", "cgan-out", "cgan-sum"], ["vae", "vae-out", "vae-sum"]]) {
      $(out).replaceChildren(...r[key].images.map(im => el("div", { class: "tile" }, el("img", { src: im.data_url, alt: `${key} ${r[key].category}` }),
        im.clip ? el("div", { class: "meta" }, `sim ${fmt(im.clip.similarity)}`) : "")));
      const s = r[key].clip_summary;
      $(sum).textContent = s ? `Mean CLIP similarity ${fmt(s.mean_similarity)} · top-1 agreement ${pct(s.top1_agreement)} · generated in ${r[key].timing_ms.generation} ms`
                             : "CLIP not available on this server – images are shown without scores.";
    }
  } catch (e) { status(e.message); } finally { btn.disabled = false; btn.textContent = "Compare models"; }
}

// ---------------- Training ----------------
async function initTraining() {
  let m;
  try { m = await API.trainingMetrics(); } catch (e) { return status(e.message); }
  const g = m.cgan, v = m.vae, base = API.BASE + m.sample_base_url;
  const stats = [];
  if (g) {
    const last = g.epochs[g.epochs.length - 1];
    stats.push(stat(`${g.completed_epochs}/${g.total_epochs}`, "cGAN epochs completed"),
               stat(fmt(last.g_loss, 3), "Final generator loss"), stat(fmt(last.d_loss, 3), "Final discriminator loss"),
               stat(g.train_images.toLocaleString(), `Training images · ${g.test_images.toLocaleString()} test`));
    const labels = g.epochs.map(e => e.epoch);
    line("chart-g", labels, [{ label: "Generator loss", data: g.epochs.map(e => e.g_loss), borderColor: PURPLE }], "BCE loss");
    line("chart-d", labels, [{ label: "Discriminator loss", data: g.epochs.map(e => e.d_loss), borderColor: BLUE }], "BCE loss");
    line("chart-score", labels, [{ label: "D(real)", data: g.epochs.map(e => e.d_real_score), borderColor: GREEN },
                                 { label: "D(fake)", data: g.epochs.map(e => e.d_fake_score), borderColor: AMBER }], "Probability real");
  } else status("cGAN has not been trained yet. Run python -m training.train_gan");
  if (v) {
    stats.push(stat(`${v.completed_epochs}/${v.total_epochs}`, "VAE epochs completed"));
    const labels = v.epochs.map(e => e.epoch);
    line("chart-vae", labels, [{ label: "Train loss", data: v.epochs.map(e => e.loss), borderColor: PURPLE },
      { label: "Test loss", data: v.epochs.map(e => e.test_loss), borderColor: BLUE },
      { label: "KL", data: v.epochs.map(e => e.kl_loss), borderColor: AMBER, borderDash: [5, 4] }], "Loss per image");
  }
  $("train-stats").replaceChildren(...stats);
  const grids = [...(g ? g.samples.map(s => ({ ...s, model: "cGAN" })) : []), ...(v ? v.samples.map(s => ({ ...s, model: "VAE" })) : [])];
  const show = (s, btns) => { $("grid-img").src = base + s.file; $("grid-title").textContent = `${s.model} · epoch ${s.epoch}`;
    btns.forEach(b => b.btn.className = "btn sm " + (b.s === s ? "" : "ghost")); };
  const btns = grids.map(s => ({ s, btn: el("button", { class: "btn sm ghost", type: "button" }, `${s.model} ep ${s.epoch}`) }));
  btns.forEach(b => { b.btn.addEventListener("click", () => show(b.s, btns)); $("grid-tabs").append(b.btn); });
  const firstFinal = grids.filter(s => s.model === "cGAN").pop() || grids[0]; if (firstFinal) show(firstFinal, btns);
  API.modelInfo().then(i => {
    const rows = [["Dataset", `${i.dataset.name}, ${i.dataset.image_size}`], ["Device", i.device]];
    if (i.cgan_generator.weights_present) rows.push(["Generator params", i.cgan_generator.parameters.toLocaleString()], ["Discriminator params", i.cgan_discriminator.parameters.toLocaleString()]);
    if (i.cvae.weights_present) rows.push(["CVAE params", i.cvae.parameters.toLocaleString()]);
    if (g) rows.push(["cGAN batch size / lr", `${g.batch_size} / ${g.learning_rate}`], ["Training device", g.device]);
    $("model-info").replaceChildren(el("div", { class: "table-wrap" }, Object.assign(el("table", { style: "min-width:0" }), {})));
    table($("model-info").querySelector("table"), ["Item", "Value"], rows);
  }).catch(e => { $("model-info").textContent = e.message; });
}

// ---------------- Evaluation ----------------
async function initEvaluation() {
  let r;
  try { r = await API.evaluation(); } catch (e) { return status(e.message); }
  const t = r.totals, cw = r.classwise;
  $("eval-stats").replaceChildren(stat(t.images_evaluated.toLocaleString(), "Images evaluated"), stat(fmt(t.cgan_mean_similarity), "cGAN mean CLIP similarity"),
    stat(fmt(t.vae_mean_similarity), "VAE mean CLIP similarity"), stat(fmt(t.cgan_mean_ms_per_image, 1) + " ms", "cGAN time per image (VAE " + fmt(t.vae_mean_ms_per_image, 1) + " ms)"));
  const labels = cw.map(c => c.category);
  bar("chart-class", labels, [{ label: "cGAN", data: cw.map(c => c.cgan.mean_similarity), backgroundColor: PURPLE },
                              { label: "VAE", data: cw.map(c => c.vae.mean_similarity), backgroundColor: BLUE }]);
  bar("chart-top1", labels, [{ label: "cGAN", data: cw.map(c => c.cgan.top1_agreement), backgroundColor: PURPLE },
                             { label: "VAE", data: cw.map(c => c.vae.top1_agreement), backgroundColor: BLUE }], 1);
  table($("prompt-table"), ["Prompt", "Requested", "Generated", "Images", "cGAN sim.", "cGAN top-1", "VAE sim.", "Time (ms)"],
    r.prompt_experiment.map(p => p.error ? [p.prompt, "–", "parse error", "0", "–", "–", "–", "–"] :
      [p.prompt, p.requested_category, p.generated_category, `${p.generated_count}/${p.requested_count}`, fmt(p.cgan.mean_similarity),
       pct(p.cgan.top1_agreement), fmt(p.vae.mean_similarity), fmt(p.total_processing_ms, 0)]));
  $("weakest").replaceChildren(...r.weakest_categories.map(w => el("p", {}, `${w.category}: mean similarity ${fmt(w.cgan_mean_similarity)}`)),
    el("p", { class: "note" }, "Ranked by lowest cGAN CLIP similarity. Low CLIP scores can reflect CLIP’s trouble with tiny grayscale images as much as GAN quality."));
  $("failures").replaceChildren(...(r.failures.length ? r.failures.map(f => el("p", {}, `“${f.prompt}” – ${f.reason}`)) : [el("p", {}, "No failures recorded in this run.")]));
  table($("class-table"), ["Class", "Images", "cGAN sim.", "cGAN top-1", "VAE sim.", "VAE top-1", "cGAN ms/img", "VAE ms/img"],
    cw.map(c => [c.category, c.n_images, fmt(c.cgan.mean_similarity), pct(c.cgan.top1_agreement), fmt(c.vae.mean_similarity), pct(c.vae.top1_agreement), c.cgan_ms_per_image, c.vae_ms_per_image]));
  $("status").className = "status show"; $("status").textContent = `Run on ${new Date(r.generated_at).toLocaleString()} · CLIP model ${r.clip_model} · seed ${r.seed}. ${r.note}`;
}
