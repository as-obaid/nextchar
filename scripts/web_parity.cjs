// Compare docs/model.js with PyTorch reference values. Run via scripts/web_parity.py.

const fs = require("fs");
const path = require("path");
const NextChar = require("../docs/model.js");

const ref = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const assets = path.join(__dirname, "..", "docs", "assets");
const manifest = JSON.parse(fs.readFileSync(path.join(assets, "manifest.json"), "utf8"));

const maxAbsDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
let worst = 0;

function report(label, diff) {
  worst = Math.max(worst, diff);
  console.log(`${label.padEnd(48)} max |diff| ${diff.toExponential(2)}`);
}

for (const [name, expected] of Object.entries(ref.models)) {
  const bytes = fs.readFileSync(path.join(assets, manifest.models[name].file));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const model = NextChar.build(manifest, name, buffer);

  for (const [seq, { ids, logits }] of Object.entries(expected.sequences)) {
    const stepper = model.stepper();
    let diff = 0;
    ids.forEach((id, t) => {
      stepper.push(id);
      diff = Math.max(diff, maxAbsDiff(stepper.logits(), logits[t]));
    });
    report(`${name} logits, ${seq} (${ids.length} steps)`, diff);
    const direct = NextChar.contextLogits(model, ids, manifest.block_size);
    report(`${name} logits, ${seq}, last window only`, maxAbsDiff(direct, logits[ids.length - 1]));
  }

  for (const [seq, { ids, weights }] of Object.entries(expected.attention || {})) {
    const got = model.attention(ids);
    let diff = 0;
    weights.forEach((layer, l) => layer.forEach((head, h) => {
      diff = Math.max(diff, maxAbsDiff(got.weights[l][h], head));
    }));
    report(`${name} attention, ${seq}`, diff);
    report(`${name} attention logits, ${seq}`, maxAbsDiff(got.logits, expected.sequences[seq].logits.at(-1)));
  }
}

const ok = worst < ref.tolerance;
console.log(`\n${ok ? "PASS" : "FAIL"}: worst ${worst.toExponential(2)} (tolerance ${ref.tolerance})`);
process.exit(ok ? 0 : 1);
