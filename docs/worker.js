// One model per worker: next-character prediction, attention weights, and sampling.
// The model name comes from the URL: worker.js?model=lstm

importScripts("model.js");

const name = new URLSearchParams(self.location.search).get("model");

const ready = (async () => {
  const manifest = await (await fetch("assets/manifest.json")).json();
  const response = await fetch(`assets/${manifest.models[name].file}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return { manifest, model: NextChar.build(manifest, name, await response.arrayBuffer()) };
})();

ready.then(
  () => self.postMessage({ type: "ready" }),
  (err) => self.postMessage({ type: "error", message: String(err) }),
);

self.onmessage = async ({ data: msg }) => {
  const { manifest, model } = await ready;
  const B = manifest.block_size;

  if (msg.type === "predict") {
    const probs = NextChar.softmax(NextChar.contextLogits(model, msg.ids, B));
    self.postMessage({ type: "predict", req: msg.req, probs });
  } else if (msg.type === "attention") {
    const { logits, weights } = model.attention(msg.ids);
    self.postMessage({ type: "attention", req: msg.req, probs: NextChar.softmax(logits), weights });
  } else if (msg.type === "generate") {
    const random = NextChar.rng(msg.seed);
    const stepper = model.stepper();
    for (const id of msg.ids.slice(-B)) stepper.push(id);
    const start = performance.now();
    for (let i = 0; i < msg.length; i++) {
      const id = NextChar.sample(NextChar.softmax(stepper.logits(), msg.temperature), random);
      stepper.push(id);
      self.postMessage({ type: "token", run: msg.run, id });
    }
    self.postMessage({ type: "done", run: msg.run, ms: performance.now() - start });
  }
};
