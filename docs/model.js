// Forward passes for the three models in models.py, in plain JavaScript.
//
// Used by the demo's web workers and by scripts/web_parity.py, which checks every
// model against PyTorch. Weights are stored as float32; arithmetic runs in float64.
//
// Each model exposes stepper(): push(id) appends a character and logits() returns
// next-character logits. Like models.generate, a model only ever sees the last
// block_size characters: the stepper runs incrementally while the text fits in
// the window and recomputes the window from scratch once it slides.

(function (root) {
  "use strict";

  function tensors(spec, buffer) {
    const all = new Float32Array(buffer);
    const out = {};
    for (const [name, { shape, offset }] of Object.entries(spec.tensors)) {
      out[name] = all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
    }
    return out;
  }

  // y = W x + b for a row-major W of shape [nOut, nIn], like nn.Linear.
  function linear(W, b, x, xOff, nIn, nOut, y, yOff) {
    for (let o = 0; o < nOut; o++) {
      let s = b ? b[o] : 0;
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) s += W[row + i] * x[xOff + i];
      y[yOff + o] = s;
    }
  }

  function layerNorm(x, xOff, n, w, b, y, yOff) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[xOff + i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (x[xOff + i] - mean) ** 2;
    const inv = 1 / Math.sqrt(variance / n + 1e-5);
    for (let i = 0; i < n; i++) y[yOff + i] = (x[xOff + i] - mean) * inv * w[i] + b[i];
  }

  // Abramowitz & Stegun 7.1.26, absolute error below 1.5e-7. nn.GELU() uses exact erf.
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * ax);
    const poly =
      t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign * (1 - poly * Math.exp(-ax * ax));
  }

  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  function bigram(spec, w, V) {
    const table = w["table.weight"];
    return {
      stepper() {
        let last = 0;
        return {
          push(id) {
            last = id;
          },
          logits() {
            return Float64Array.from(table.subarray(last * V, last * V + V));
          },
        };
      },
    };
  }

  function lstm(spec, w, V, B) {
    const { n_embd: E, hidden_size: H, n_layer: L } = spec.hparams;
    const G = 4 * H; // gates in PyTorch order: input, forget, cell, output
    const W_ih = [], W_hh = [], bias = [];
    for (let l = 0; l < L; l++) {
      W_ih.push(w[`lstm.weight_ih_l${l}`]);
      W_hh.push(w[`lstm.weight_hh_l${l}`]);
      const b = new Float64Array(G);
      for (let i = 0; i < G; i++) b[i] = w[`lstm.bias_ih_l${l}`][i] + w[`lstm.bias_hh_l${l}`][i];
      bias.push(b);
    }
    // Layer 0's input term depends only on the character, so compute it once per character.
    const input0 = new Float64Array(V * G);
    for (let v = 0; v < V; v++) {
      linear(W_ih[0], bias[0], w["embed.weight"], v * E, E, G, input0, v * G);
    }
    const gates = new Float64Array(G);

    const fresh = () => ({
      h: Array.from({ length: L }, () => new Float64Array(H)),
      c: Array.from({ length: L }, () => new Float64Array(H)),
    });

    function step(state, id) {
      for (let l = 0; l < L; l++) {
        const h = state.h[l], c = state.c[l];
        if (l === 0) gates.set(input0.subarray(id * G, id * G + G));
        else linear(W_ih[l], bias[l], state.h[l - 1], 0, H, G, gates, 0);
        const Wh = W_hh[l];
        for (let o = 0; o < G; o++) {
          let s = 0;
          const row = o * H;
          for (let i = 0; i < H; i++) s += Wh[row + i] * h[i];
          gates[o] += s;
        }
        for (let j = 0; j < H; j++) {
          const i = sigmoid(gates[j]);
          const f = sigmoid(gates[H + j]);
          const g = Math.tanh(gates[2 * H + j]);
          const o = sigmoid(gates[3 * H + j]);
          c[j] = f * c[j] + i * g;
          h[j] = o * Math.tanh(c[j]);
        }
      }
    }

    return {
      stepper() {
        const ids = [];
        let state = fresh();
        return {
          push(id) {
            ids.push(id);
            if (ids.length <= B) {
              step(state, id);
            } else {
              // Every window starts from a zero state, as in training.
              state = fresh();
              for (const t of ids.slice(-B)) step(state, t);
            }
          },
          logits() {
            const out = new Float64Array(V);
            linear(w["head.weight"], w["head.bias"], state.h[L - 1], 0, H, V, out, 0);
            return out;
          },
        };
      },
    };
  }

  function transformer(spec, w, V, B) {
    const { n_embd: C, n_head: NH, n_layer: L, ff_dim: F } = spec.hparams;
    const hd = C / NH;
    const scale = 1 / Math.sqrt(hd);
    const blocks = Array.from({ length: L }, (_, l) => {
      const p = (name) => w[`blocks.${l}.${name}`];
      return {
        ln1w: p("ln1.weight"), ln1b: p("ln1.bias"),
        qkvW: p("attn.qkv.weight"), qkvB: p("attn.qkv.bias"),
        projW: p("attn.proj.weight"), projB: p("attn.proj.bias"),
        ln2w: p("ln2.weight"), ln2b: p("ln2.bias"),
        ff0W: p("ff.0.weight"), ff0B: p("ff.0.bias"),
        ff2W: p("ff.2.weight"), ff2B: p("ff.2.bias"),
      };
    });
    const tok = w["tok_emb.weight"], pos = w["pos_emb.weight"];
    const hC = new Float64Array(C), tmpC = new Float64Array(C), hF = new Float64Array(F);
    const y = new Float64Array(C), scores = new Float64Array(B);

    function embed(id, t, x, xOff) {
      for (let i = 0; i < C; i++) x[xOff + i] = tok[id * C + i] + pos[t * C + i];
    }

    // Causal attention for one query over key/value rows 0..n-1 (row stride `stride`).
    // Writes the concatenated head outputs to y; fills att[head] when att is given.
    function attend(q, qOff, K, kOff, Vs, vOff, stride, n, att) {
      for (let head = 0; head < NH; head++) {
        const a = att ? att[head] : scores;
        const ho = head * hd;
        let max = -Infinity;
        for (let s = 0; s < n; s++) {
          let d = 0;
          const k = kOff + s * stride + ho;
          for (let j = 0; j < hd; j++) d += q[qOff + ho + j] * K[k + j];
          a[s] = d * scale;
          if (a[s] > max) max = a[s];
        }
        let sum = 0;
        for (let s = 0; s < n; s++) {
          a[s] = Math.exp(a[s] - max);
          sum += a[s];
        }
        for (let s = 0; s < n; s++) a[s] /= sum;
        for (let j = 0; j < hd; j++) {
          let v = 0;
          for (let s = 0; s < n; s++) v += a[s] * Vs[vOff + s * stride + ho + j];
          y[ho + j] = v;
        }
      }
    }

    // The rest of a block after attention: projection, residual, feedforward, residual.
    function finishBlock(blk, x, xOff) {
      linear(blk.projW, blk.projB, y, 0, C, C, tmpC, 0);
      for (let i = 0; i < C; i++) x[xOff + i] += tmpC[i];
      layerNorm(x, xOff, C, blk.ln2w, blk.ln2b, hC, 0);
      linear(blk.ff0W, blk.ff0B, hC, 0, C, F, hF, 0);
      for (let i = 0; i < F; i++) hF[i] = gelu(hF[i]);
      linear(blk.ff2W, blk.ff2B, hF, 0, F, C, tmpC, 0);
      for (let i = 0; i < C; i++) x[xOff + i] += tmpC[i];
    }

    function headLogits(x, xOff) {
      layerNorm(x, xOff, C, w["ln_f.weight"], w["ln_f.bias"], hC, 0);
      const out = new Float64Array(V);
      linear(w["head.weight"], w["head.bias"], hC, 0, C, V, out, 0);
      return out;
    }

    // Full forward pass over a window of at most B ids. Only the last position's
    // output is needed, so the final block skips every other query.
    function runWindow(ids, captureAttention) {
      const T = ids.length;
      const x = new Float64Array(T * C);
      const qkv = new Float64Array(T * 3 * C);
      const attention = [];
      ids.forEach((id, t) => embed(id, t, x, t * C));
      blocks.forEach((blk, l) => {
        for (let t = 0; t < T; t++) {
          layerNorm(x, t * C, C, blk.ln1w, blk.ln1b, hC, 0);
          linear(blk.qkvW, blk.qkvB, hC, 0, C, 3 * C, qkv, t * 3 * C);
        }
        for (let t = l === L - 1 ? T - 1 : 0; t < T; t++) {
          const att = captureAttention && t === T - 1
            ? Array.from({ length: NH }, () => new Float64Array(T))
            : null;
          attend(qkv, t * 3 * C, qkv, C, qkv, 2 * C, 3 * C, t + 1, att);
          if (att) attention.push(att);
          finishBlock(blk, x, t * C);
        }
      });
      return { x: x.subarray((T - 1) * C), attention };
    }

    return {
      stepper() {
        const ids = [];
        const K = blocks.map(() => new Float64Array(B * C));
        const Vc = blocks.map(() => new Float64Array(B * C));
        const qkvRow = new Float64Array(3 * C);
        let last = new Float64Array(C);
        return {
          push(id) {
            ids.push(id);
            if (ids.length > B) {
              // Positions shift once the window slides, so recompute it.
              last = runWindow(ids.slice(-B), false).x;
              return;
            }
            const t = ids.length - 1;
            const x = new Float64Array(C);
            embed(id, t, x, 0);
            blocks.forEach((blk, l) => {
              layerNorm(x, 0, C, blk.ln1w, blk.ln1b, hC, 0);
              linear(blk.qkvW, blk.qkvB, hC, 0, C, 3 * C, qkvRow, 0);
              K[l].set(qkvRow.subarray(C, 2 * C), t * C);
              Vc[l].set(qkvRow.subarray(2 * C), t * C);
              attend(qkvRow, 0, K[l], 0, Vc[l], 0, C, t + 1, null);
              finishBlock(blk, x, 0);
            });
            last = x;
          },
          logits() {
            return headLogits(last, 0);
          },
        };
      },
      // Next-character logits plus the last position's attention weights in every
      // layer: weights[layer][head] is a Float64Array over the window.
      attention(ids) {
        const { x, attention } = runWindow(ids.slice(-B), true);
        return { logits: headLogits(x, 0), weights: attention };
      },
    };
  }

  const BUILDERS = { bigram, lstm, transformer };

  function build(manifest, name, buffer) {
    const spec = manifest.models[name];
    return BUILDERS[name](spec, tensors(spec, buffer), manifest.vocab.length, manifest.block_size);
  }

  // Logits for the character after `ids`. Only the last block_size ids matter.
  function contextLogits(model, ids, blockSize) {
    const s = model.stepper();
    for (const id of ids.slice(-blockSize)) s.push(id);
    return s.logits();
  }

  function softmax(logits, temperature = 1) {
    const p = new Float64Array(logits.length);
    let max = -Infinity;
    for (const v of logits) if (v / temperature > max) max = v / temperature;
    let sum = 0;
    for (let i = 0; i < p.length; i++) {
      p[i] = Math.exp(logits[i] / temperature - max);
      sum += p[i];
    }
    for (let i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  }

  // mulberry32: a small seeded PRNG, so a seed gives the same sample every time.
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sample(probs, random) {
    const u = random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (u < cumulative) return i;
    }
    return probs.length - 1;
  }

  const api = { build, contextLogits, softmax, rng, sample };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NextChar = api;
})(typeof self !== "undefined" ? self : globalThis);
