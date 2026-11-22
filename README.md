# binary neural network

A 2-2-1 neural network learning XOR by backpropagation, shown one layer at a time.

**Live: https://zaffer.github.io/neural-network-visualisation/**

Step a layer, a sample or an epoch, or play. While paused, the α slider re-applies the last
update at a different learning rate, so you can see how far one step moves every weight.

## Files

```
index.html   the panels: native controls in fieldsets, bound to the store with data-* attributes
app.js       the store (plainpanel signals), the training actions, and the effects that draw
net.js       the network as pure functions: init, forward, gradients, apply
stage.js     the 3D view: three.js behind a narrow API (setModel, setActivations, setLabels)
style.css    layout only; the page works with it switched off (view → 🚫 no CSS)
```

No build step. [plainpanel](https://github.com/Zaffer/plainpanel) and three.js load from jsDelivr,
pinned in the import map. To run it locally, serve the folder, for example
`python3 -m http.server`, and open http://localhost:8000.
