// 训练历程报告模板的无浏览器冒烟:最小 DOM 桩执行 report.html 中的脚本。
// 用法:node tests/_journey_dom_stub.js <report.html>
// 输出一行 JSON:{ok, sections:{id:children}, hidden:[...], bad_attrs:[...], error}
"use strict";
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const dataM = html.match(/<script type="application\/json" id="journey-data">([\s\S]*?)<\/script>/);
const codeM = html.match(/<script>\n?"use strict";([\s\S]*?)<\/script>\s*<\/body>/);
const bad = [];
const nodes = {};
function cls() { const s = new Set(); return { add: (...a) => a.forEach(x => s.add(x)), remove: (...a) => a.forEach(x => s.delete(x)),
  toggle: x => (s.has(x) ? s.delete(x) : s.add(x)), contains: x => s.has(x), _s: s }; }
function mk(tag, id) {
  const n = { tag, id, attrs: {}, children: [], style: {}, dataset: {}, classList: cls(), _text: "", _html: "",
    offsetWidth: 120, offsetHeight: 40,
    setAttribute(k, v) { this.attrs[k] = v; if (/NaN|undefined|Infinity/.test(String(v))) bad.push(`${tag}.${k}=${v}`); },
    getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {},
    set textContent(v) { this._text = v; }, get textContent() { return this._text; },
    set innerHTML(v) { this._html = v; if (/undefined|>NaN|NaN<|NaN%|\[object/.test(String(v))) bad.push(`innerHTML: ${String(v).slice(0, 160)}`); },  // 正文里的"NaN 原生"等字样不算
    get innerHTML() { return this._html; },
    set className(v) { v.split(/\s+/).filter(Boolean).forEach(x => this.classList.add(x)); }, get className() { return [...this.classList._s].join(" "); },
    set href(v) { this.attrs.href = v; }, get href() { return this.attrs.href; },
    set id(v) { this._id = v; if (v) nodes[v] = this; }, get id() { return this._id; },
  };
  if (id) { n._id = id; nodes[id] = n; }
  return n;
}
global.window = { innerWidth: 1400, innerHeight: 900 };
global.document = {
  getElementById(id) {
    if (id === "journey-data") return { textContent: dataM ? dataM[1].replace(/<\\\//g, "</") : "null" };
    return nodes[id] || mk("#", id);
  },
  createElementNS(ns, tag) { return mk(tag); },
  createElement(tag) { return mk(tag); },
  querySelector() { return null; },
  querySelectorAll() { return []; },
};
const out = { ok: false, sections: {}, hidden: [], bad_attrs: bad, error: null };
try {
  if (!dataM || !codeM) throw new Error("模板结构缺失(journey-data 或主脚本)");
  JSON.parse(dataM[1].replace(/<\\\//g, "</"));
  eval(codeM[1]);
  for (const id of ["h-kpis", "c-ladder", "c-share", "c-evo", "c-path", "c-forest", "c-fals", "c-final-table",
                    "c-cap", "c-dec", "c-gen", "c-donut", "c-events", "c-decisions", "c-ins", "toc-nav"]) {
    const n = nodes[id]; out.sections[id] = n ? (n.children.length || (n._html ? 1 : 0)) : 0;
  }
  for (const id of ["s-ladder", "s-evo", "s-path", "s-forest", "s-fals", "s-final", "s-process", "s-events", "s-insights"])
    if (nodes[id] && nodes[id].style.display === "none") out.hidden.push(id);
  out.ok = bad.length === 0;
} catch (e) { out.error = String(e && e.stack || e); }
console.log(JSON.stringify(out));
