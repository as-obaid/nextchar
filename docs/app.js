// NextChar demo: live next-character predictions, sampling, and transformer attention.
// All model work runs in web workers (worker.js); this file only drives the page.

(() => {
  "use strict";

  const MODELS = ["bigram", "lstm", "transformer"];
  const LABELS = { bigram: "Bigram", lstm: "LSTM", transformer: "Transformer" };
  const PRESETS = ["Good morrow, my l", "KING RICHARD II", "Thou art", "ROMEO:\n", "KING RICHARD III:\n"];
  const TOP_K = 8;
  const TAPE = 18;

  const $ = (id) => document.getElementById(id);
  const make = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };
  const visible = (c) => (c === " " ? "␣" : c === "\n" ? "↵" : c);
  const quoted = (c) => `“${visible(c)}”`;
  const isSpace = (c) => c === " " || c === "\n";
  const pct = (p) => `${(p * 100).toFixed(1)}%`;
  const fmt = (n) => n.toLocaleString("en-US");

  const ui = { cards: {}, samples: {} };
  const state = {
    manifest: null,
    vocab: [],
    stoi: new Map(),
    block: 128,
    ids: [],
    predictors: {},
    generators: {},
    samples: {},
    run: 0,
    running: new Set(),
    att: { source: "sample", layer: 2, head: "avg", text: [], selected: 0, result: null, version: 0 },
  };

  // ---------- workers ----------

  // A worker that keeps only the latest request of each type, so typing fast
  // never builds a queue of stale predictions.
  function predictor(name, handlers) {
    const worker = new Worker(`worker.js?model=${name}`);
    const w = { busy: false, pending: new Map(), sent: {} };
    const pump = () => {
      if (w.busy || w.pending.size === 0) return;
      const [type, msg] = w.pending.entries().next().value;
      w.pending.delete(type);
      w.busy = true;
      w.sent[type] = msg;
      worker.postMessage(msg);
    };
    worker.onmessage = ({ data }) => {
      if (data.type === "error") return showLoadError(data.message);
      if (data.type === "ready") return;
      w.busy = false;
      handlers[data.type](data, w.sent[data.type]);
      pump();
    };
    worker.onerror = (e) => showLoadError(e.message);
    w.request = (msg) => {
      w.pending.set(msg.type, msg);
      pump();
    };
    return w;
  }

  function generator(name) {
    const worker = new Worker(`worker.js?model=${name}`);
    worker.onmessage = ({ data }) => {
      if (data.type === "error") return showLoadError(data.message);
      if (data.run !== state.run) return;
      if (data.type === "token") onToken(name, data.id);
      if (data.type === "done") onDone(name, data.ms);
    };
    worker.onerror = (e) => showLoadError(e.message);
    return worker;
  }

  function showLoadError(detail) {
    if (document.querySelector(".error")) return;
    const box = make("p", "error");
    box.textContent =
      "The model files did not load. Serve the docs folder over HTTP (for example, run " +
      "“python3 -m http.server” inside it) instead of opening index.html from disk." +
      (detail ? ` (${detail})` : "");
    document.querySelector("main").prepend(box);
  }

  // ---------- context ----------

  function encode(text) {
    const ids = [];
    const skipped = new Set();
    for (const c of text) {
      const id = state.stoi.get(c);
      if (id === undefined) skipped.add(c);
      else ids.push(id);
    }
    return { ids, skipped };
  }

  const decode = (ids) => ids.map((i) => state.vocab[i]).join("");

  function onContext() {
    const text = $("context").value;
    const { ids, skipped } = encode(text);
    state.ids = ids;

    let note = `${fmt(ids.length)} character${ids.length === 1 ? "" : "s"}.`;
    if (skipped.size) {
      note += ` Skipped ${[...skipped].map((c) => `“${c}”`).join(" ")}: the models never saw ${skipped.size === 1 ? "it" : "them"} in training.`;
    }
    $("context-note").textContent = note;
    for (const b of document.querySelectorAll(".presets button")) {
      b.setAttribute("aria-pressed", String(b.dataset.text === text));
    }

    for (const name of MODELS) {
      if (ids.length) state.predictors[name].request({ type: "predict", ids });
      else renderEmptyPrediction(name);
    }
    if (state.att.source === "context" || !state.samples.transformer) setAttentionText();
  }

  function appendChar(c) {
    const box = $("context");
    box.value += c;
    onContext();
  }

  // ---------- prediction ----------

  function buildPredictionCard(name) {
    const spec = state.manifest.models[name];
    const card = make("article", "model");
    card.dataset.model = name;

    const head = make("div", "model-head");
    const title = make("div", "model-name");
    title.append(make("span", "swatch"), make("span", null, LABELS[name]));
    head.append(title, make("div", "model-meta", `${fmt(spec.params)} parameters · val loss ${spec.val_loss.toFixed(3)}`));

    const sees = make("div", "sees");
    const seesLabel = make("div", "sees-label");
    const tape = make("div", "tape");
    tape.setAttribute("aria-hidden", "true");
    sees.append(seesLabel, tape);

    const guess = make("div", "guess");
    const glyph = make("div", "guess-glyph");
    const guessText = make("div", "guess-text");
    const guessPct = make("strong");
    guessText.append(guessPct, make("span", null, "top guess"));
    guess.append(glyph, guessText);

    const bars = make("ol", "bars");
    bars.setAttribute("aria-label", `${LABELS[name]}: most likely next characters`);
    const rows = [];
    for (let k = 0; k < TOP_K; k++) {
      const li = make("li");
      const button = make("button", "bar-row");
      button.type = "button";
      const g = make("span", "glyph");
      const track = make("span", "track");
      const fill = make("span", "fill");
      track.append(fill);
      const p = make("span", "pct");
      button.append(g, track, p);
      button.addEventListener("click", () => appendChar(button.dataset.char));
      li.append(button);
      bars.append(li);
      rows.push({ button, g, fill, p });
    }

    card.append(head, sees, guess, bars);
    ui.cards[name] = { card, seesLabel, tape, glyph, guessPct, rows };
    return card;
  }

  function renderTape(name, ids) {
    const { seesLabel, tape } = ui.cards[name];
    const n = ids.length;
    const span = name === "bigram" ? 1 : state.block;
    seesLabel.textContent =
      name === "bigram" ? "Sees only the last character"
      : n <= span ? `Sees all ${fmt(n)} character${n === 1 ? "" : "s"}`
      : `Sees the last ${span} of ${fmt(n)} characters`;
    tape.replaceChildren();
    const start = Math.max(0, n - TAPE);
    for (let i = start; i < n; i++) {
      const c = state.vocab[ids[i]];
      const cell = make("span", isSpace(c) ? "ws" : null, visible(c));
      if (i >= n - span) cell.classList.add("seen");
      tape.append(cell);
    }
  }

  function renderPrediction(name, probs, ids) {
    const card = ui.cards[name];
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, TOP_K);
    renderTape(name, ids);
    const best = state.vocab[order[0]];
    card.glyph.textContent = visible(best);
    card.glyph.classList.toggle("ws", isSpace(best));
    card.guessPct.textContent = pct(probs[order[0]]);
    order.forEach((id, k) => {
      const row = card.rows[k];
      const c = state.vocab[id];
      row.button.hidden = false;
      row.button.dataset.char = c;
      row.button.classList.toggle("top", k === 0);
      row.button.setAttribute("aria-label", `Add ${c === " " ? "space" : c === "\n" ? "new line" : quoted(c)}, ${pct(probs[id])}`);
      row.g.textContent = visible(c);
      row.g.classList.toggle("ws", isSpace(c));
      row.fill.style.width = `${probs[id] * 100}%`;
      row.p.textContent = pct(probs[id]);
    });
  }

  function renderEmptyPrediction(name) {
    const card = ui.cards[name];
    card.seesLabel.textContent = "Type at least one character";
    card.tape.replaceChildren();
    card.glyph.textContent = "";
    card.guessPct.textContent = "–";
    for (const row of card.rows) row.button.hidden = true;
  }

  // ---------- sampling ----------

  function buildSampleCard(name) {
    const spec = state.manifest.models[name];
    const card = make("article", "model");
    card.dataset.model = name;
    const head = make("div", "model-head");
    const title = make("div", "model-name");
    title.append(make("span", "swatch"), make("span", null, LABELS[name]));
    head.append(title, make("div", "model-meta", `${fmt(spec.params)} parameters · val loss ${spec.val_loss.toFixed(3)}`));
    const out = make("pre", "out");
    const prompt = make("span", "prompt");
    const gen = make("span", "gen");
    const text = document.createTextNode("");
    gen.append(text);
    out.append(prompt, gen);
    const status = make("div", "status");
    status.setAttribute("aria-live", "off");
    card.append(head, out, status);
    ui.samples[name] = { prompt, text, status };
    return card;
  }

  function settings() {
    const clamp = (v, lo, hi, fallback) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback);
    return {
      temperature: clamp(parseFloat($("temperature").value), 0.1, 1.5, 0.8),
      length: Math.round(clamp(parseInt($("length").value, 10), 20, 1000, 300)),
      seed: clamp(parseInt($("seed").value, 10), 0, 999999, 1337) >>> 0,
    };
  }

  function startSampling() {
    const ids = state.ids;
    if (!ids.length) {
      $("context-note").textContent = "Type at least one character to sample from.";
      return;
    }
    stopSampling();
    const run = ++state.run;
    const { temperature, length, seed } = settings();
    for (const name of MODELS) {
      state.samples[name] = { promptIds: ids.slice(), ids: [], length, done: false };
      const view = ui.samples[name];
      view.prompt.textContent = decode(ids);
      view.text.data = "";
      view.status.textContent = `0 / ${length}`;
      state.running.add(name);
      state.generators[name].postMessage({ type: "generate", run, ids, length, temperature, seed });
    }
    setRunning(true);
    if (state.att.source === "sample") setAttentionText();
  }

  // A worker cannot be interrupted mid-loop, so stopping replaces it.
  function stopSampling() {
    for (const name of state.running) {
      state.generators[name].terminate();
      state.generators[name] = generator(name);
      const s = state.samples[name];
      ui.samples[name].status.textContent = `Stopped at ${s.ids.length} / ${s.length}`;
    }
    state.running.clear();
    setRunning(false);
  }

  function setRunning(on) {
    const button = $("generate");
    button.textContent = on ? "Stop" : "Generate";
    button.classList.toggle("running", on);
  }

  function onToken(name, id) {
    const s = state.samples[name];
    s.ids.push(id);
    const view = ui.samples[name];
    view.text.appendData(state.vocab[id]);
    view.status.textContent = `${s.ids.length} / ${s.length}`;
  }

  function onDone(name, ms) {
    const s = state.samples[name];
    s.done = true;
    const took = ms < 1000 ? `${Math.max(1, Math.round(ms))} ms` : `${(ms / 1000).toFixed(1)} s`;
    ui.samples[name].status.textContent = `${s.length} characters in ${took}`;
    state.running.delete(name);
    if (state.running.size === 0) setRunning(false);
    if (name === "transformer" && state.att.source === "sample") setAttentionText();
  }

  // ---------- attention ----------

  function segmented(container, label, options, current, onPick) {
    container.setAttribute("role", "group");
    container.replaceChildren(make("span", "seg-label", label));
    const buttons = options.map(([value, text]) => {
      const b = make("button", null, text);
      b.type = "button";
      b.setAttribute("aria-pressed", String(value === current));
      b.addEventListener("click", () => {
        buttons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        onPick(value);
      });
      container.append(b);
      return b;
    });
  }

  function attentionSourceIds() {
    const sample = state.samples.transformer;
    if (state.att.source === "sample" && sample) return sample.promptIds.concat(sample.ids);
    return state.ids;
  }

  function setAttentionText() {
    const ids = attentionSourceIds();
    state.att.text = ids;
    state.att.version++;
    state.att.selected = ids.length; // the slot for the next character
    state.att.result = null;
    const box = $("att-text");
    box.replaceChildren();
    ids.forEach((id, i) => {
      const c = state.vocab[id];
      const span = make("span", "ch", c === "\n" ? "↵\n" : c);
      span.dataset.i = i;
      if (c === "\n") span.classList.add("ws");
      if (i === 0) span.classList.add("first");
      box.append(span);
    });
    const next = make("span", "next");
    next.dataset.i = ids.length;
    next.title = "The next character";
    box.append(next);
    box.scrollTop = box.scrollHeight;
    requestAttention();
  }

  function requestAttention() {
    const { text, selected } = state.att;
    if (!text.length || selected < 1) {
      $("att-caption").textContent = "Type a context to see where the transformer looks.";
      $("att-top").replaceChildren();
      return;
    }
    const start = Math.max(0, selected - state.block);
    const { version } = state.att;
    state.predictors.transformer.request({ type: "attention", ids: text.slice(start, selected), start, selected, version });
  }

  function onAttention(data, sent) {
    if (sent.version !== state.att.version || sent.selected !== state.att.selected) return;
    state.att.result = { probs: data.probs, weights: data.weights, start: sent.start, selected: sent.selected };
    renderAttention();
  }

  function renderAttention() {
    const { result, text, layer, head } = state.att;
    if (!result) return;
    const { weights, probs, start, selected } = result;
    const heads = weights[layer - 1];
    const T = heads[0].length;
    const w = new Float64Array(T);
    for (let j = 0; j < T; j++) {
      w[j] = head === "avg" ? heads.reduce((s, h) => s + h[j], 0) / heads.length : heads[head - 1][j];
    }
    const max = Math.max(...w);

    for (const span of $("att-text").children) {
      const i = Number(span.dataset.i);
      span.classList.toggle("selected", i === selected);
      if (!span.classList.contains("ch")) continue;
      const inside = i >= start && i < selected;
      span.classList.toggle("outside", !inside);
      // Square root so weaker links stay visible next to the strongest one.
      span.style.setProperty("--w", inside ? Math.sqrt(w[i - start] / max).toFixed(3) : "0");
    }

    const n = text.length;
    const caption = $("att-caption");
    caption.replaceChildren();
    const where = `Layer ${layer}, ${head === "avg" ? `average of ${heads.length} heads` : `head ${head}`}.`;
    if (selected < n) {
      const c = state.vocab[text[selected]];
      caption.append(
        "Predicting ", make("strong", null, quoted(c)),
        ` (character ${fmt(selected + 1)} of ${fmt(n)}) from the ${T} before it. The transformer gave it ${pct(probs[text[selected]])}. ${where}`,
      );
    } else {
      const best = [...probs.keys()].reduce((a, b) => (probs[b] > probs[a] ? b : a), 0);
      caption.append(
        `Predicting the next character from the last ${T}. Top guess `,
        make("strong", null, quoted(state.vocab[best])),
        ` at ${pct(probs[best])}. ${where}`,
      );
    }

    const list = $("att-top");
    list.replaceChildren();
    [...w.keys()].sort((a, b) => w[b] - w[a]).slice(0, 5).forEach((j) => {
      const c = state.vocab[text[start + j]];
      const li = make("li");
      const track = make("span", "track");
      const fill = make("span", "fill");
      fill.style.width = `${w[j] * 100}%`;
      track.append(fill);
      const back = selected - (start + j);
      li.append(make("span", `glyph${isSpace(c) ? " ws" : ""}`, visible(c)), make("span", "dist", `${back} back`), track, make("span", "pct", pct(w[j])));
      list.append(li);
    });
  }

  function selectAttention(i) {
    const n = state.att.text.length;
    const s = Math.min(n, Math.max(1, i));
    if (!n || s === state.att.selected) return;
    state.att.selected = s;
    // Keep the selection in view without scrolling the page itself.
    const box = $("att-text");
    const target = box.querySelector(`[data-i="${s}"]`);
    if (target) {
      if (target.offsetTop < box.scrollTop) box.scrollTop = target.offsetTop - 16;
      else if (target.offsetTop + target.offsetHeight > box.scrollTop + box.clientHeight) {
        box.scrollTop = target.offsetTop + target.offsetHeight - box.clientHeight + 16;
      }
    }
    requestAttention();
  }

  // ---------- scores ----------

  function buildScores() {
    const { models, vocab } = state.manifest;
    const uniform = Math.log(vocab.length);
    const scaleMax = Math.ceil(uniform * 2) / 2;
    $("vocab-size").textContent = vocab.length;
    $("uniform-loss").textContent = uniform.toFixed(2);
    $("steps").textContent = fmt(models.bigram.steps);

    const table = $("scores");
    const head = make("div", "score-row head");
    head.setAttribute("role", "row");
    ["Model", "Validation loss", "", "Parameters", "Training time"].forEach((t, k) => {
      const cell = make("span", k >= 2 ? "num" : null, t);
      cell.setAttribute("role", "columnheader");
      head.append(cell);
    });
    table.append(head);

    for (const name of MODELS) {
      const m = models[name];
      const row = make("div", "score-row");
      row.dataset.model = name;
      row.setAttribute("role", "row");
      const title = make("div", "model-name");
      title.append(make("span", "swatch"), make("span", null, LABELS[name]));
      const track = make("div", "score-track");
      const fill = make("span", "fill");
      fill.style.width = `${(m.val_loss / scaleMax) * 100}%`;
      const mark = make("span", "uniform");
      mark.style.left = `${(uniform / scaleMax) * 100}%`;
      mark.title = `Uniform guess: ${uniform.toFixed(2)}`;
      track.append(fill, mark);
      const cells = [
        title,
        track,
        make("span", "num loss", m.val_loss.toFixed(3)),
        make("span", "num meta params", `${fmt(m.params)}`),
        make("span", "num meta time", `${m.train_time_s.toFixed(1)} s`),
      ];
      cells.forEach((c) => c.setAttribute("role", "cell"));
      row.append(...cells);
      table.append(row);
    }
  }

  // ---------- start ----------

  async function init() {
    try {
      const response = await fetch("assets/manifest.json");
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      state.manifest = await response.json();
    } catch (err) {
      showLoadError(String(err.message || err));
      return;
    }
    state.vocab = state.manifest.vocab;
    state.block = state.manifest.block_size;
    state.vocab.forEach((c, i) => state.stoi.set(c, i));

    for (const name of MODELS) {
      $("predictions").append(buildPredictionCard(name));
      $("samples").append(buildSampleCard(name));
      state.generators[name] = generator(name);
      state.predictors[name] = predictor(name, {
        predict: (data, sent) => renderPrediction(name, data.probs, sent.ids),
        attention: onAttention,
      });
    }
    buildScores();

    const presets = document.querySelector(".presets");
    for (const text of PRESETS) {
      const b = make("button", null, text.replace(/\n/g, "↵"));
      b.type = "button";
      b.dataset.text = text;
      b.addEventListener("click", () => {
        $("context").value = text;
        onContext();
      });
      presets.append(b);
    }

    segmented($("att-source"), "Text", [["sample", "Transformer sample"], ["context", "Context"]], state.att.source, (v) => {
      state.att.source = v;
      setAttentionText();
    });
    const { n_layer: layers, n_head: heads } = state.manifest.models.transformer.hparams;
    const numbered = (n) => Array.from({ length: n }, (_, i) => [i + 1, String(i + 1)]);
    state.att.layer = layers;  // open on the last layer
    segmented($("att-layer"), "Layer", numbered(layers), state.att.layer, (v) => {
      state.att.layer = v;
      renderAttention();
    });
    segmented($("att-head"), "Head", [["avg", "All"], ...numbered(heads)], state.att.head, (v) => {
      state.att.head = v;
      renderAttention();
    });

    const attText = $("att-text");
    attText.addEventListener("click", (e) => {
      const target = e.target.closest("[data-i]");
      if (target) selectAttention(Number(target.dataset.i));
    });
    attText.addEventListener("keydown", (e) => {
      const keys = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity };
      if (!(e.key in keys)) return;
      e.preventDefault();
      const step = keys[e.key];
      selectAttention(Number.isFinite(step) ? state.att.selected + step : step);
    });

    $("context").addEventListener("input", onContext);
    $("temperature").addEventListener("input", () => {
      $("temperature-out").textContent = Number($("temperature").value).toFixed(2);
    });
    $("controls").addEventListener("submit", (e) => {
      e.preventDefault();
      if (state.running.size) stopSampling();
      else startSampling();
    });

    // Open in a working state: a sample context, its predictions, and a sample run.
    $("context").value = PRESETS[0];
    onContext();
    startSampling();
  }

  init();
})();
