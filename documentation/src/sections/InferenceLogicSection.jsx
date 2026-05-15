import React, { useState, useEffect, useRef } from 'react';
import {
  ChevronRight, ChevronLeft, Play, Pause, Shuffle, ArrowDown,
  ArrowRight, Database, Cpu, Zap, Activity, Target, BarChart2,
  RefreshCw, CheckCircle, Info, GitMerge, Box, Layers
} from 'lucide-react';

// ─────────────────────────────────────────────────────────────
// SHARED HELPERS
// ─────────────────────────────────────────────────────────────

const Pill = ({ children, color = 'blue' }) => {
  const colors = {
    blue:   'bg-blue-900/60 text-blue-200 border-blue-500/60',
    green:  'bg-emerald-900/60 text-emerald-200 border-emerald-500/60',
    purple: 'bg-purple-900/60 text-purple-200 border-purple-500/60',
    amber:  'bg-amber-900/60 text-amber-200 border-amber-500/60',
    red:    'bg-red-900/60 text-red-200 border-red-500/60',
    slate:  'bg-slate-700/60 text-slate-200 border-slate-500/60',
    indigo: 'bg-indigo-900/60 text-indigo-200 border-indigo-500/60',
  };
  return (
    <span className={`text-[11px] font-mono px-2 py-0.5 rounded border ${colors[color] || colors.blue}`}>
      {children}
    </span>
  );
};

const NavBtn = ({ onClick, disabled, children, primary }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`flex items-center gap-1.5 text-xs px-4 py-2 rounded-lg font-bold transition-all shadow active:scale-95
      disabled:opacity-30 disabled:cursor-not-allowed disabled:active:scale-100
      ${primary
        ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/50'
        : 'bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600'}`}
  >
    {children}
  </button>
);

const CodeLine = ({ children }) => (
  <div className="font-mono text-xs leading-relaxed text-slate-200">{children}</div>
);
const Kw = ({ c }) => <span className="text-rose-400 font-bold">{c}</span>;
const Var = ({ c }) => <span className="text-blue-300 italic">{c}</span>;
const Fn = ({ c }) => <span className="text-amber-300">{c}</span>;
const Str = ({ c }) => <span className="text-indigo-300">{c}</span>;
const Num = ({ c }) => <span className="text-purple-300">{c}</span>;
const Comment = ({ c }) => <span className="text-slate-500 italic">{c}</span>;

// ─────────────────────────────────────────────────────────────
// STEP 0  — The Starting Point: test.parquet
// ─────────────────────────────────────────────────────────────
const Step0_TestParquet = () => {
  const [highlight, setHighlight] = useState(null);
  const cols = [
    { group: 'x (condition)', cols: ['COMP_Suction_Pressure', 'COMP_Suction_Drum_Temp', 'KPI_Fuel_Gas_LHV'], color: 'blue' },
    { group: 'u (bypassed)', cols: ['Turbine_SHAFT_SPEED', 'UK_14PDCV-504_H-SEL', 'SEAL_GAS_SUP_DE'], color: 'slate' },
    { group: 'θ (ground truth)', cols: Array.from({ length: 12 }, (_, i) => `PCA_Coefficient_${i + 1}`), color: 'amber' },
  ];
  return (
    <div className="flex flex-col gap-6">
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-5">
        <div className="flex items-center gap-2 mb-4">
          <Database size={18} className="text-blue-400" />
          <span className="text-sm font-bold text-slate-200">test.parquet — One row = one 4-hour scenario</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono border-collapse">
            <thead>
              <tr>
                {cols.map(g => (
                  <th key={g.group} colSpan={g.cols.length}
                    className={`px-2 py-1.5 text-center border border-slate-700 cursor-pointer
                      ${highlight === g.group ? 'bg-blue-700/40 text-blue-200' : 'bg-slate-800 text-slate-400'}`}
                    onClick={() => setHighlight(highlight === g.group ? null : g.group)}>
                    {g.group} <span className="opacity-60">({g.cols.length} cols)</span>
                  </th>
                ))}
              </tr>
              <tr>
                {cols.flatMap(g => g.cols.map(c => (
                  <th key={c}
                    className={`px-2 py-1 border border-slate-700 font-normal text-center whitespace-nowrap
                      ${highlight === g.group ? 'bg-blue-900/40 text-blue-300' : 'text-slate-500'}`}>
                    {c.length > 16 ? c.slice(0, 14) + '…' : c}
                  </th>
                )))}
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2].map(row => (
                <tr key={row}>
                  {cols.flatMap(g => g.cols.map(c => (
                    <td key={c}
                      className={`px-2 py-1 border border-slate-800 text-center
                        ${highlight === g.group ? 'bg-blue-900/20 text-blue-200' : 'text-slate-400'}`}>
                      {(Math.random() * 2 - 1).toFixed(3)}
                    </td>
                  )))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-slate-500 mt-3">
          * All values already <strong className="text-slate-300">StandardScaler-normalized</strong> from preprocessing.py — no further scaling needed at inference.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'x — the condition', icon: Zap, color: 'text-blue-400', bg: 'bg-blue-900/30 border-blue-700', body: '3 measured-now SCADA readings. This is the ONLY thing we send into the MLP during inference. It answers: "What is the compressor doing RIGHT NOW?"', pill: '3 dims' },
          { label: 'u — the control', icon: RefreshCw, color: 'text-slate-400', bg: 'bg-slate-800 border-slate-600', body: 'Shaft speed, valve positions, seal gas. These BYPASS the flow entirely. They were used in earlier project phases but are not wired into Phase II coupling layers.', pill: '3 dims — ignored' },
          { label: 'θ — ground truth PCA', icon: Target, color: 'text-amber-400', bg: 'bg-amber-900/30 border-amber-700', body: '12 PCA coefficients encoding the entire 4-hour future trajectory. Used ONLY to evaluate/compare after inference. Never fed to the model during generation.', pill: '12 dims' },
        ].map(({ label, icon: Icon, color, bg, body, pill }) => (
          <div key={label} className={`rounded-xl border p-4 ${bg}`}>
            <div className="flex items-center gap-2 mb-2">
              <Icon size={16} className={color} />
              <span className="text-sm font-bold text-slate-200">{label}</span>
              <Pill color={label.includes('x') ? 'blue' : label.includes('u') ? 'slate' : 'amber'}>{pill}</Pill>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{body}</p>
          </div>
        ))}
      </div>

      <div className="bg-amber-900/20 border border-amber-700 rounded-xl p-4">
        <div className="flex items-start gap-2">
          <Info size={16} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-xs text-amber-200 leading-relaxed">
            <strong>Key takeaway:</strong> During inference we only READ <code className="bg-amber-900/60 px-1 rounded">x</code> from the test row.
            The 12 θ columns are kept aside as the "answer key" to score the model after generation.
            The model never sees θ_true when producing predictions.
          </p>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// STEP 1  — Sample z ~ N(0, I₁₂)
// ─────────────────────────────────────────────────────────────
const Step1_SampleZ = () => {
  const [samples, setSamples] = useState(() => Array.from({ length: 12 }, () => (Math.random() * 6 - 3).toFixed(3)));
  const [nDraws, setNDraws] = useState(1);
  const [animating, setAnimating] = useState(false);

  const resample = () => {
    setAnimating(true);
    setNDraws(n => n + 1);
    setSamples(Array.from({ length: 12 }, () => (Math.random() * 6 - 3).toFixed(3)));
    setTimeout(() => setAnimating(false), 400);
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Shuffle size={18} className="text-purple-400" />
            <span className="text-sm font-bold text-slate-200">Draw z ~ N(0, I₁₂)  — 12-dimensional standard normal</span>
          </div>
          <button onClick={resample}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-purple-700 hover:bg-purple-600 text-white rounded-lg font-bold transition-all active:scale-95">
            <RefreshCw size={12} /> New Draw #{nDraws}
          </button>
        </div>

        {/* Bell curve SVG */}
        <div className="flex items-center gap-6">
          <div className="flex-1">
            <svg viewBox="0 0 300 100" className="w-full h-28">
              <defs>
                <linearGradient id="bellGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#7c3aed" stopOpacity="0.1" />
                  <stop offset="50%" stopColor="#7c3aed" stopOpacity="0.6" />
                  <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.1" />
                </linearGradient>
              </defs>
              {/* Gaussian curve */}
              <path
                d="M 10,90 C 30,88 50,80 70,60 C 90,40 110,10 150,5 C 190,10 210,40 230,60 C 250,80 270,88 290,90 Z"
                fill="url(#bellGrad)" stroke="#7c3aed" strokeWidth="1.5"
              />
              <text x="150" y="97" textAnchor="middle" fill="#94a3b8" fontSize="8" fontFamily="monospace">μ=0, σ=1</text>
              <line x1="150" y1="5" x2="150" y2="90" stroke="#a78bfa" strokeWidth="1" strokeDasharray="3,2" />
              {/* 1σ markers */}
              <line x1="110" y1="88" x2="110" y2="55" stroke="#7c3aed" strokeWidth="0.8" strokeDasharray="2,2" />
              <line x1="190" y1="88" x2="190" y2="55" stroke="#7c3aed" strokeWidth="0.8" strokeDasharray="2,2" />
              <text x="110" y="97" textAnchor="middle" fill="#7c3aed" fontSize="7" fontFamily="monospace">-1σ</text>
              <text x="190" y="97" textAnchor="middle" fill="#7c3aed" fontSize="7" fontFamily="monospace">+1σ</text>
              {/* Sample dots */}
              {samples.slice(0, 12).map((v, i) => {
                const x = 150 + parseFloat(v) * 40;
                return (
                  <circle key={i} cx={Math.max(15, Math.min(285, x))} cy={75} r={3}
                    fill={`hsl(${i * 30}, 70%, 65%)`}
                    className={animating ? 'opacity-0' : 'opacity-100'}
                    style={{ transition: 'opacity 0.3s' }} />
                );
              })}
            </svg>
          </div>

          {/* The 12-dim vector */}
          <div className="flex-shrink-0">
            <div className="text-[10px] text-slate-500 mb-1 font-mono text-center">z vector (12 dims)</div>
            <div className="grid grid-cols-3 gap-1">
              {samples.map((v, i) => (
                <div key={i}
                  className={`text-[11px] font-mono px-2 py-1 rounded border text-center transition-all duration-300
                    ${animating ? 'opacity-0 scale-95' : 'opacity-100 scale-100'}
                    ${Math.abs(parseFloat(v)) < 1 ? 'bg-purple-900/50 border-purple-600 text-purple-200' :
                      Math.abs(parseFloat(v)) < 2 ? 'bg-indigo-900/50 border-indigo-600 text-indigo-200' :
                        'bg-red-900/40 border-red-700 text-red-300'}`}>
                  z<sub>{i + 1}</sub>: {v}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
          <div className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1.5"><Cpu size={14} className="text-purple-400" /> Code</div>
          <div className="bg-slate-900 rounded p-3 space-y-1">
            <CodeLine><Comment c="# In flow_model.py — sample()" /></CodeLine>
            <CodeLine><Var c="blueprint" /> = <Fn c="self.get_blueprint" />()</CodeLine>
            <CodeLine><Comment c="# Returns Independent(Normal(0,1), 12 dims)" /></CodeLine>
            <CodeLine><Var c="z" /> = <Var c="blueprint" />.<Fn c="sample" />(<Fn c="(batch_size,)" />)</CodeLine>
            <CodeLine><Comment c="# z.shape = (batch_size, 12)" /></CodeLine>
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
          <div className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1.5"><Info size={14} className="text-blue-400" /> Why N(0,1)?</div>
          <p className="text-xs text-slate-400 leading-relaxed">
            The flow was <strong className="text-slate-200">trained to push θ → z</strong> such that z lands in N(0,1).
            Inference reverses this: we start from N(0,1) and walk backward through the splines.
            Any point you sample is a <em className="text-purple-300">valid physically plausible scenario</em> for that operating condition.
          </p>
          <div className="mt-2 flex gap-2 flex-wrap">
            <Pill color="purple">z.shape = (N, 12)</Pill>
            <Pill color="purple">each dim ~ N(0,1)</Pill>
            <Pill color="blue">N = num_samples</Pill>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// STEP 2  — The Reverse Loop
// ─────────────────────────────────────────────────────────────
const Step2_ReverseLoop = () => {
  const [hovered, setHovered] = useState(null);
  const layers = [3, 2, 1, 0];

  return (
    <div className="flex flex-col gap-6">
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-5">
        <div className="flex items-center gap-2 mb-1">
          <RefreshCw size={18} className="text-green-400" />
          <span className="text-sm font-bold text-slate-200">Reversed Layer Loop — walk backward through 4 coupling layers</span>
        </div>
        <p className="text-xs text-slate-400 mb-4">
          Training goes <strong className="text-blue-300">θ → z</strong> (forward).
          Inference goes <strong className="text-emerald-300">z → θ</strong> (backward, reversed order, inverse spline).
        </p>

        {/* Horizontal pipeline diagram */}
        <div className="overflow-x-auto">
          <div className="flex items-center gap-2 min-w-max py-2 px-1">
            {/* z input */}
            <div className="flex flex-col items-center gap-1">
              <div className="w-16 h-16 rounded-xl bg-purple-900/60 border-2 border-purple-500 flex flex-col items-center justify-center shadow-lg shadow-purple-900/30">
                <span className="text-[10px] text-purple-300 font-mono font-bold">z</span>
                <span className="text-[9px] text-purple-400">~N(0,1)</span>
                <span className="text-[9px] text-purple-400">12 dims</span>
              </div>
              <span className="text-[9px] text-slate-500">INPUT</span>
            </div>

            {layers.map((li, idx) => (
              <React.Fragment key={li}>
                <div className="flex flex-col items-center">
                  {/* Flip arrow */}
                  <div className="flex items-center gap-1 mb-1">
                    <ArrowRight size={10} className="text-slate-600" />
                    <span className="text-[8px] text-slate-500 font-mono">flip + .contiguous()</span>
                    <ArrowRight size={10} className="text-slate-600" />
                  </div>
                  {/* Layer box */}
                  <div
                    className={`w-28 border-2 rounded-xl p-2 cursor-pointer transition-all shadow-lg
                      ${hovered === li
                        ? 'bg-blue-900/60 border-blue-400 scale-105 shadow-blue-900/40'
                        : 'bg-slate-800 border-slate-600 hover:border-slate-400'}`}
                    onMouseEnter={() => setHovered(li)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <div className="text-[10px] font-bold text-slate-200 text-center mb-1">Layer {li}</div>
                    <div className="text-[9px] text-slate-400 text-center font-mono">.inverse(z, x)</div>
                    <div className="flex justify-center gap-1 mt-1.5">
                      <span className="text-[8px] bg-blue-900/50 text-blue-300 px-1 rounded border border-blue-700">z₁ pass</span>
                      <span className="text-[8px] bg-emerald-900/50 text-emerald-300 px-1 rounded border border-emerald-700">z₂ warp</span>
                    </div>
                  </div>
                </div>
                {idx < layers.length - 1 && (
                  <ArrowRight size={16} className="text-slate-600 flex-shrink-0" />
                )}
              </React.Fragment>
            ))}

            <ArrowRight size={16} className="text-slate-600 flex-shrink-0" />

            {/* θ output */}
            <div className="flex flex-col items-center gap-1">
              <div className="w-16 h-16 rounded-xl bg-amber-900/60 border-2 border-amber-500 flex flex-col items-center justify-center shadow-lg shadow-amber-900/30">
                <span className="text-[10px] text-amber-300 font-mono font-bold">θ̂</span>
                <span className="text-[9px] text-amber-400">predicted</span>
                <span className="text-[9px] text-amber-400">12 dims</span>
              </div>
              <span className="text-[9px] text-slate-500">OUTPUT</span>
            </div>
          </div>
        </div>

        {hovered !== null && (
          <div className="mt-3 bg-blue-900/30 border border-blue-700 rounded-lg p-3 text-xs text-blue-200">
            <strong>Layer {hovered} (inverse):</strong> flip z → split [z₁ | z₂] →
            brain_input = [z₁, x] → MLP → spline params for z₂ →
            inverse RQ-spline on z₂ → output = [z₁ | θ₂_hat]
          </div>
        )}
      </div>

      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        <div className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1.5"><Cpu size={14} className="text-green-400" /> Exact code path — flow_model.py <code>sample()</code></div>
        <div className="bg-slate-950 rounded p-3 space-y-0.5 overflow-x-auto">
          <CodeLine><Kw c="for" /> <Var c="layer" /> <Kw c="in" /> <Fn c="reversed" />(<Var c="self.layers" />):</CodeLine>
          <CodeLine>    <Comment c="# 1. Undo the flip that was applied during training" /></CodeLine>
          <CodeLine>    <Var c="z" /> = <Fn c="torch.flip" />(<Var c="z" />, dims=[<Num c="-1" />]).<Fn c="contiguous" />()</CodeLine>
          <CodeLine>    <Comment c="    ↑ .contiguous() required on MPS — flip() returns non-contiguous tensor" /></CodeLine>
          <CodeLine>    <Comment c="      torch.searchsorted (inside the spline) produces NaN without this" /></CodeLine>
          <CodeLine>&nbsp;</CodeLine>
          <CodeLine>    <Comment c="# 2. Run the algebraic inverse through this coupling layer" /></CodeLine>
          <CodeLine>    <Var c="z" /> = <Var c="layer" />.<Fn c="inverse" />(<Var c="z" />, <Var c="condition" />)</CodeLine>
          <CodeLine>&nbsp;</CodeLine>
          <CodeLine><Kw c="return" /> <Var c="z" />  <Comment c="# Now θ_predicted — shape: (batch_size, 12)" /></CodeLine>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// STEP 3  — Inside ONE inverse coupling layer
// ─────────────────────────────────────────────────────────────
const Step3_InsideLayer = () => {
  const [subStep, setSubStep] = useState(0);
  const SUB_STEPS = [
    { label: '① Flip', color: 'slate' },
    { label: '② Split', color: 'blue' },
    { label: '③ Brain input', color: 'indigo' },
    { label: '④ MLP → spline params', color: 'purple' },
    { label: '⑤ Inverse spline on z₂', color: 'green' },
    { label: '⑥ Concat output', color: 'amber' },
  ];

  const DIMS = 12;
  const HALF = 6;

  const content = [
    // ① Flip
    <div key={0} className="flex flex-col gap-4">
      <p className="text-xs text-slate-400 leading-relaxed">
        Before entering the layer we undo the flip that was applied when this layer ran in the <em>forward</em> direction during training.
        This puts z back in the same orientation the layer expects for its inverse pass.
      </p>
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        <div className="flex items-center gap-4 justify-center flex-wrap">
          <div>
            <div className="text-[10px] text-slate-500 mb-1 text-center font-mono">z (before flip)</div>
            <div className="flex gap-1">
              {Array.from({ length: DIMS }, (_, i) => (
                <div key={i} className="w-8 h-8 rounded border border-slate-600 bg-slate-800 flex items-center justify-center text-[9px] font-mono text-slate-300">
                  {i + 1}
                </div>
              ))}
            </div>
          </div>
          <div className="flex flex-col items-center gap-1">
            <RefreshCw size={20} className="text-slate-500" />
            <span className="text-[9px] text-slate-500 font-mono">torch.flip</span>
          </div>
          <div>
            <div className="text-[10px] text-slate-500 mb-1 text-center font-mono">z (after flip)</div>
            <div className="flex gap-1">
              {Array.from({ length: DIMS }, (_, i) => (
                <div key={i} className="w-8 h-8 rounded border border-blue-700/60 bg-blue-900/30 flex items-center justify-center text-[9px] font-mono text-blue-300">
                  {DIMS - i}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="bg-slate-950 rounded mt-3 p-2">
          <CodeLine><Var c="z" /> = <Fn c="torch.flip" />(<Var c="z" />, dims=[<Num c="-1" />]).<Fn c="contiguous" />()</CodeLine>
        </div>
      </div>
    </div>,

    // ② Split
    <div key={1} className="flex flex-col gap-4">
      <p className="text-xs text-slate-400 leading-relaxed">
        z is split cleanly in half: <strong className="text-blue-300">z₁</strong> (dims 0–5) and <strong className="text-emerald-300">z₂</strong> (dims 6–11).
        z₁ <em>never changes</em> — it passes straight through. Only z₂ gets transformed.
      </p>
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        <div className="flex gap-1 justify-center mb-3">
          {Array.from({ length: DIMS }, (_, i) => (
            <div key={i} className={`w-9 h-9 rounded border flex items-center justify-center text-[9px] font-mono
              ${i < HALF
                ? 'border-blue-600 bg-blue-900/50 text-blue-200'
                : 'border-emerald-600 bg-emerald-900/50 text-emerald-200'}`}>
              z{i + 1}
            </div>
          ))}
        </div>
        <div className="flex gap-4 justify-center">
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              {Array.from({ length: HALF }, (_, i) => (
                <div key={i} className="w-9 h-8 rounded border border-blue-600 bg-blue-900/60 flex items-center justify-center text-[9px] font-mono text-blue-200">z{i + 1}</div>
              ))}
            </div>
            <Pill color="blue">z₁  →  conditions MLP</Pill>
          </div>
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              {Array.from({ length: HALF }, (_, i) => (
                <div key={i} className="w-9 h-8 rounded border border-emerald-600 bg-emerald-900/60 flex items-center justify-center text-[9px] font-mono text-emerald-200">z{HALF + i + 1}</div>
              ))}
            </div>
            <Pill color="green">z₂  →  gets inverse-splined</Pill>
          </div>
        </div>
        <div className="bg-slate-950 rounded mt-3 p-2">
          <CodeLine><Var c="z_1" />, <Var c="z_2" /> = <Var c="z" />[:, :<Num c="6" />],  <Var c="z" />[:, <Num c="6" />:]</CodeLine>
        </div>
      </div>
    </div>,

    // ③ Brain input
    <div key={2} className="flex flex-col gap-4">
      <p className="text-xs text-slate-400 leading-relaxed">
        The MLP sees <strong className="text-blue-300">z₁</strong> (6 dims) <em>concatenated</em> with <strong className="text-amber-300">x</strong> (3 dims) = <strong className="text-white">9 total inputs</strong>.
        x is the measured SCADA state — it <em>shapes the spline boxes</em> to match the current operating condition.
      </p>
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        <div className="flex items-center justify-center gap-3 flex-wrap">
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              {Array.from({ length: HALF }, (_, i) => (
                <div key={i} className="w-8 h-8 rounded border border-blue-600 bg-blue-900/60 flex items-center justify-center text-[8px] font-mono text-blue-200">z{i + 1}</div>
              ))}
            </div>
            <Pill color="blue">z₁  (6)</Pill>
          </div>
          <span className="text-slate-400 font-bold text-lg">⊕</span>
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              {['Pₛ', 'Tₛ', 'LHV'].map(l => (
                <div key={l} className="w-10 h-8 rounded border border-amber-600 bg-amber-900/60 flex items-center justify-center text-[9px] font-mono text-amber-200">{l}</div>
              ))}
            </div>
            <Pill color="amber">x  (3)</Pill>
          </div>
          <ArrowRight size={20} className="text-slate-500" />
          <div className="flex flex-col items-center gap-1">
            <div className="w-20 h-12 rounded-xl border-2 border-indigo-500 bg-indigo-900/40 flex items-center justify-center">
              <span className="text-[10px] font-mono font-bold text-indigo-200">MLP</span>
            </div>
            <Pill color="indigo">input dim = 9</Pill>
          </div>
        </div>
        <div className="bg-slate-950 rounded mt-3 p-2">
          <CodeLine><Var c="brain_input" /> = <Fn c="torch.cat" />([<Var c="z_1" />, <Var c="condition" />], dim=<Num c="-1" />)</CodeLine>
          <CodeLine><Comment c="# brain_input.shape = (batch, 9)" /></CodeLine>
        </div>
      </div>
      <div className="bg-indigo-900/20 border border-indigo-700 rounded-lg p-3 text-xs text-indigo-200">
        <strong>This is identical to training.</strong> The MLP was always called with exactly this shape (6 + 3 = 9).
        During training it was [θ₁, x]. During inference it is [z₁, x] — structurally the same tensor shape.
      </div>
    </div>,

    // ④ MLP → spline params
    <div key={3} className="flex flex-col gap-4">
      <p className="text-xs text-slate-400 leading-relaxed">
        The MLP outputs <strong className="text-purple-300">(3K − 1) × 6</strong> raw numbers — the raw widths, heights and slopes
        of the spline boxes that will be applied to z₂.
        With K=12 bins this is <strong className="text-white">35 × 6 = 210 numbers</strong> per sample.
      </p>
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        <div className="flex items-center justify-center gap-4 flex-wrap">
          <div className="w-20 h-16 rounded-xl border-2 border-indigo-500 bg-indigo-900/40 flex flex-col items-center justify-center gap-1 shadow-lg">
            <span className="text-[9px] text-indigo-300 font-mono">MLP</span>
            <span className="text-[8px] text-slate-400">9 → 128 → 210</span>
          </div>
          <ArrowRight size={16} className="text-slate-500" />
          <div className="flex flex-col gap-1">
            {[
              { label: 'unnorm_widths', dims: 'K=12 per dim × 6', color: 'bg-purple-900/50 border-purple-600 text-purple-200' },
              { label: 'unnorm_heights', dims: 'K=12 per dim × 6', color: 'bg-fuchsia-900/50 border-fuchsia-600 text-fuchsia-200' },
              { label: 'unnorm_derivs', dims: 'K-1=11 per dim × 6', color: 'bg-pink-900/50 border-pink-600 text-pink-200' },
            ].map(p => (
              <div key={p.label} className={`flex items-center gap-2 text-[9px] font-mono rounded border px-2 py-1 ${p.color}`}>
                <span>{p.label}</span>
                <Pill color="slate">{p.dims}</Pill>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-slate-950 rounded mt-3 p-2 space-y-0.5">
          <CodeLine><Var c="raw_params" /> = <Var c="self.brain" />(<Var c="brain_input" />)</CodeLine>
          <CodeLine><Var c="raw_params" /> = <Var c="raw_params" />.<Fn c="reshape" />(-<Num c="1" />, <Num c="6" />, <Num c="35" />)  <Comment c="# (batch, 6 dims, 3K-1)" /></CodeLine>
          <CodeLine><Var c="W" /> = <Var c="raw_params" />[..., :<Num c="12" />]      <Comment c="# unnormalized widths" /></CodeLine>
          <CodeLine><Var c="H" /> = <Var c="raw_params" />[..., <Num c="12" />:<Num c="24" />]  <Comment c="# unnormalized heights" /></CodeLine>
          <CodeLine><Var c="D" /> = <Var c="raw_params" />[..., <Num c="24" />:]      <Comment c="# unnormalized derivatives" /></CodeLine>
        </div>
      </div>
    </div>,

    // ⑤ Inverse spline
    <div key={4} className="flex flex-col gap-4">
      <p className="text-xs text-slate-400 leading-relaxed">
        The <strong className="text-emerald-300">inverse rational-quadratic spline</strong> maps each z₂ value back through the spline box
        using the quadratic formula. This is the algebraic "time-reversal" of what happened during training.
      </p>
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        {/* Spline visual */}
        <svg viewBox="0 0 320 140" className="w-full h-36">
          <defs>
            <linearGradient id="splineGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.05" />
            </linearGradient>
          </defs>
          {/* Grid */}
          {[1, 2, 3, 4].map(i => (
            <line key={i} x1={60 * i} y1="10" x2={60 * i} y2="120" stroke="#334155" strokeWidth="0.5" />
          ))}
          {[1, 2, 3].map(i => (
            <line key={i} x1="10" y1={30 * i + 10} x2="310" y2={30 * i + 10} stroke="#334155" strokeWidth="0.5" />
          ))}
          {/* Identity line */}
          <line x1="10" y1="120" x2="310" y2="10" stroke="#475569" strokeWidth="1" strokeDasharray="4,3" />
          <text x="270" y="22" fill="#64748b" fontSize="8" fontFamily="monospace">identity</text>
          {/* Spline curve (forward — θ→z) */}
          <path d="M 10,120 C 60,115 90,100 130,75 C 165,52 195,30 230,20 C 260,12 285,10 310,10"
            fill="none" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="5,3" />
          <text x="250" y="45" fill="#818cf8" fontSize="8" fontFamily="monospace">forward (θ→z)</text>
          {/* Inverse curve (z→θ) */}
          <path d="M 10,120 C 30,108 55,90 90,72 C 140,48 200,22 260,12 C 285,8 300,9 310,10"
            fill="url(#splineGrad)" stroke="#10b981" strokeWidth="2" />
          <text x="110" y="108" fill="#10b981" fontSize="8" fontFamily="monospace">inverse (z→θ)</text>
          {/* Example z₂ point and its mapping */}
          <circle cx="170" cy="55" r="4" fill="#f59e0b" />
          <line x1="170" y1="55" x2="170" y2="130" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3,2" />
          <text x="163" y="136" fill="#f59e0b" fontSize="7" fontFamily="monospace">z₂</text>
          <line x1="10" y1="55" x2="170" y2="55" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3,2" />
          <text x="12" y="52" fill="#f59e0b" fontSize="7" fontFamily="monospace">θ₂</text>
          {/* Axes */}
          <line x1="10" y1="10" x2="10" y2="130" stroke="#64748b" strokeWidth="1" />
          <line x1="10" y1="130" x2="315" y2="130" stroke="#64748b" strokeWidth="1" />
          <text x="155" y="138" fill="#64748b" fontSize="8" textAnchor="middle" fontFamily="monospace">z₂ (latent)</text>
          <text x="4" y="70" fill="#64748b" fontSize="8" textAnchor="middle" fontFamily="monospace" transform="rotate(-90,4,70)">θ₂ (PCA space)</text>
        </svg>
        <div className="bg-slate-950 rounded mt-2 p-2 space-y-0.5">
          <CodeLine><Comment c="# Quadratic formula inverse inside rational_quadratic_spline(inverse=True)" /></CodeLine>
          <CodeLine><Var c="a" /> = H_k*(S - D_0) + y_shifted*(D_1 + D_0 - <Num c="2" />*S)</CodeLine>
          <CodeLine><Var c="b" /> = H_k*D_0 - y_shifted*(D_1 + D_0 - <Num c="2" />*S)</CodeLine>
          <CodeLine><Var c="c" /> = -S * y_shifted</CodeLine>
          <CodeLine><Var c="xi" /> = (<Num c="2" />*c) / (-b - <Fn c="torch.sqrt" />(b**<Num c="2" /> - <Num c="4" />*a*c))</CodeLine>
          <CodeLine><Var c="outputs" /> = start_x + xi * W_k  <Comment c="# θ₂ recovered" /></CodeLine>
        </div>
      </div>
    </div>,

    // ⑥ Concat output
    <div key={5} className="flex flex-col gap-4">
      <p className="text-xs text-slate-400 leading-relaxed">
        z₁ (unchanged) is glued back to θ₂_hat (freshly inverse-splined z₂). This becomes the input to the next layer.
        After 4 layers, all 12 dimensions have been progressively warped from latent space back into PCA coefficient space.
      </p>
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        <div className="flex items-center justify-center gap-3 flex-wrap">
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              {Array.from({ length: HALF }, (_, i) => (
                <div key={i} className="w-9 h-9 rounded border border-blue-600 bg-blue-900/60 flex items-center justify-center text-[9px] font-mono text-blue-200">z{i + 1}</div>
              ))}
            </div>
            <Pill color="blue">z₁ — unchanged</Pill>
          </div>
          <span className="text-slate-400 font-bold text-xl">⊕</span>
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              {Array.from({ length: HALF }, (_, i) => (
                <div key={i} className="w-9 h-9 rounded border border-amber-600 bg-amber-900/60 flex items-center justify-center text-[8px] font-mono text-amber-200">θ̂{HALF + i + 1}</div>
              ))}
            </div>
            <Pill color="amber">θ₂_hat — inverse-splined</Pill>
          </div>
          <ArrowRight size={20} className="text-slate-500" />
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1 flex-wrap max-w-[250px]">
              {Array.from({ length: DIMS }, (_, i) => (
                <div key={i} className={`w-9 h-9 rounded border flex items-center justify-center text-[8px] font-mono
                  ${i < HALF ? 'border-blue-600 bg-blue-900/40 text-blue-200' : 'border-amber-600 bg-amber-900/40 text-amber-200'}`}>
                  {i < HALF ? `z${i + 1}` : `θ̂${i + 1}`}
                </div>
              ))}
            </div>
            <Pill color="slate">→ next layer's input</Pill>
          </div>
        </div>
        <div className="bg-slate-950 rounded mt-3 p-2">
          <CodeLine><Kw c="return" /> <Fn c="torch.cat" />([<Var c="z_1" />, <Var c="outputs" />], dim=<Num c="-1" />)</CodeLine>
          <CodeLine><Comment c="# shape: (batch, 12) — ready for the next reversed layer" /></CodeLine>
        </div>
      </div>
    </div>,
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-2 flex-wrap">
        {SUB_STEPS.map((s, i) => (
          <button key={i} onClick={() => setSubStep(i)}
            className={`text-[11px] px-3 py-1.5 rounded-lg font-bold border transition-all
              ${subStep === i ? 'bg-blue-600 border-blue-400 text-white' : 'bg-slate-800 border-slate-600 text-slate-300 hover:border-slate-400'}`}>
            {s.label}
          </button>
        ))}
      </div>
      <div className="min-h-[280px]">
        {content[subStep]}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// STEP 4  — Output: 12 PCA coefficients
// ─────────────────────────────────────────────────────────────
const Step4_PCOutput = () => {
  const [nSamples, setNSamples] = useState(3);
  const samples = Array.from({ length: nSamples }, (_, si) =>
    Array.from({ length: 12 }, () => (Math.random() * 4 - 2).toFixed(3))
  );

  return (
    <div className="flex flex-col gap-6">
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <BarChart2 size={18} className="text-amber-400" />
            <span className="text-sm font-bold text-slate-200">Output: N × 12 PCA coefficient vectors</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Samples:</span>
            {[1, 3, 5].map(n => (
              <button key={n} onClick={() => setNSamples(n)}
                className={`text-xs px-2.5 py-1 rounded font-bold border transition-all
                  ${nSamples === n ? 'bg-amber-600 border-amber-400 text-white' : 'bg-slate-800 border-slate-600 text-slate-300'}`}>
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="text-[11px] font-mono border-collapse w-full">
            <thead>
              <tr>
                <th className="px-2 py-1 bg-slate-800 border border-slate-700 text-slate-400 text-left">draw #</th>
                {Array.from({ length: 12 }, (_, i) => (
                  <th key={i} className="px-2 py-1 bg-slate-800 border border-slate-700 text-amber-400 text-center">PC{String(i + 1).padStart(2, '0')}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {samples.map((row, si) => (
                <tr key={si}>
                  <td className="px-2 py-1 border border-slate-800 text-slate-400">draw {si + 1}</td>
                  {row.map((v, ci) => (
                    <td key={ci} className={`px-2 py-1 border border-slate-800 text-center
                      ${Math.abs(parseFloat(v)) < 1 ? 'text-emerald-300' :
                        Math.abs(parseFloat(v)) < 2 ? 'text-amber-300' : 'text-red-300'}`}>
                      {v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-3 flex gap-3 flex-wrap">
          <Pill color="green">|v| &lt; 1.0 → high prob region</Pill>
          <Pill color="amber">|v| 1–2 → medium prob</Pill>
          <Pill color="red">|v| &gt; 2 → rare / outlier</Pill>
        </div>
      </div>

      <div className="bg-amber-900/20 border border-amber-700 rounded-xl p-4">
        <div className="flex items-start gap-2">
          <Info size={16} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <div className="text-xs text-amber-200 leading-relaxed space-y-1">
            <p><strong>These 12 numbers ARE the prediction.</strong> Each is a coordinate along a learned "shape direction" in the 4-hour trajectory space.</p>
            <p>PC01 typically encodes the dominant trend (e.g., gradual pressure rise).
               PC02 encodes the main variation orthogonal to that, and so on.
               Together, 12 components reconstruct &gt;90% of the variance in the original 8 sensor signals over 4 hours.</p>
            <p>Each draw from a <em>different</em> z gives different coefficients — sampling the <strong>conditional distribution</strong> p(θ | x).</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// STEP 5  — Decode back to physical sensor values
// ─────────────────────────────────────────────────────────────
const Step5_Decode = () => {
  const [activeStage, setActiveStage] = useState(0);
  const stages = [
    { label: 'StandardScaler⁻¹', short: 'Unscale PCA', color: 'blue' },
    { label: '× PCA components', short: 'Unproject', color: 'purple' },
    { label: 'Reshape (240,8)', short: 'Reshape', color: 'indigo' },
    { label: 'RobustScaler⁻¹', short: 'Physical units', color: 'green' },
  ];

  const SENSORS = ['Seal Gas Filter DP', 'Lube Oil Level', 'Thermal Cycle Eff', 'Isentropic Eff', 'Discharge Pressure', 'Discharge Temp', 'Exhaust Temp Spread', 'Turbine Heat Rate'];

  const StageDesc = [
    <div key={0} className="space-y-2">
      <p className="text-xs text-slate-400 leading-relaxed">
        The 12 PCA coefficients were <strong className="text-blue-300">StandardScaler-normalized</strong> during preprocessing so they fit inside the flow's bound=9 window.
        We must reverse this first.
      </p>
      <div className="bg-slate-950 rounded p-2 space-y-0.5">
        <CodeLine><Comment c="# In evaluate.py — panel_sampled_trajectories()" /></CodeLine>
        <CodeLine><Kw c="import" /> joblib</CodeLine>
        <CodeLine><Var c="coeff_scaler" /> = <Fn c="joblib.load" />(<Str c="'outputs/checkpoints/pca_coeff_scaler.pkl'" />)</CodeLine>
        <CodeLine><Var c="theta_unscaled" /> = <Var c="coeff_scaler" />.<Fn c="inverse_transform" />(<Var c="sampled_pca" />)</CodeLine>
        <CodeLine><Comment c="# sampled_pca: (100, 12) → theta_unscaled: (100, 12)" /></CodeLine>
      </div>
    </div>,
    <div key={1} className="space-y-2">
      <p className="text-xs text-slate-400 leading-relaxed">
        PCA is a linear projection: θ = z_pca · V<sup>T</sup> + μ, where V are the principal components and μ is the training mean.
        Matrix multiply to go from 12 coefficients back to 1920 flattened sensor values.
      </p>
      <div className="bg-slate-950 rounded p-2 space-y-0.5">
        <CodeLine><Var c="pca" /> = <Fn c="joblib.load" />(<Str c="'outputs/checkpoints/trajectory_pca_model.pkl'" />)</CodeLine>
        <CodeLine><Comment c="# pca.components_: (12, 1920)  pca.mean_: (1920,)" /></CodeLine>
        <CodeLine><Var c="flat" /> = <Var c="theta_unscaled" /> @ <Var c="pca.components_" /> + <Var c="pca.mean_" /></CodeLine>
        <CodeLine><Comment c="# flat: (100, 1920)  =  100 draws × (240 timesteps × 8 sensors) flattened" /></CodeLine>
        <CodeLine><Comment c="# This is mathematically: PCA.inverse_transform() but using @ for Apple Silicon safety" /></CodeLine>
      </div>
    </div>,
    <div key={2} className="space-y-2">
      <p className="text-xs text-slate-400 leading-relaxed">
        The 1920 flat values are 240 timesteps × 8 sensors (WINDOW_SIZE÷DOWNSAMPLE = 14400÷60).
        Reshape to get a proper time-series array.
      </p>
      <div className="bg-slate-950 rounded p-2 space-y-0.5">
        <CodeLine><Var c="trajectories" /> = <Var c="flat" />.<Fn c="reshape" />(<Num c="100" />, <Num c="240" />, <Num c="8" />)</CodeLine>
        <CodeLine><Comment c="# trajectories: (100 draws, 240 minutes, 8 sensors)" /></CodeLine>
        <CodeLine><Comment c="# axis 0 = different scenario draws" /></CodeLine>
        <CodeLine><Comment c="# axis 1 = time (1-minute intervals over 4 hours)" /></CodeLine>
        <CodeLine><Comment c="# axis 2 = sensor channels" /></CodeLine>
      </div>
    </div>,
    <div key={3} className="space-y-2">
      <p className="text-xs text-slate-400 leading-relaxed">
        The theta values were <strong className="text-green-300">RobustScaler-normalized</strong> during preprocessing. The final inverse_transform
        returns values in the <em>original physical engineering units</em> of each sensor.
      </p>
      <div className="bg-slate-950 rounded p-2 space-y-0.5">
        <CodeLine><Var c="base_scaler" /> = <Fn c="joblib.load" />(<Str c="'outputs/checkpoints/theta_base_scaler.pkl'" />)</CodeLine>
        <CodeLine><Kw c="for" /> <Var c="t" /> <Kw c="in" /> <Fn c="range" />(<Num c="240" />):</CodeLine>
        <CodeLine>    <Var c="trajectories" />[:, t, :] = <Var c="base_scaler" />.<Fn c="inverse_transform" />(<Var c="trajectories" />[:, t, :])</CodeLine>
        <CodeLine><Comment c="# Now trajectories has real-world units: bar, °C, %, etc." /></CodeLine>
      </div>
    </div>,
  ];

  return (
    <div className="flex flex-col gap-4">
      {/* Pipeline steps */}
      <div className="flex items-center gap-0 overflow-x-auto pb-2">
        {stages.map((s, i) => (
          <React.Fragment key={i}>
            <button onClick={() => setActiveStage(i)}
              className={`flex-shrink-0 flex flex-col items-center gap-1 px-3 py-2 rounded-lg border transition-all text-center
                ${activeStage === i ? 'bg-blue-700/60 border-blue-400' : 'bg-slate-800 border-slate-600 hover:border-slate-400'}`}>
              <span className="text-[10px] font-bold text-slate-200 whitespace-nowrap">{s.label}</span>
              <span className="text-[9px] text-slate-400">{s.short}</span>
            </button>
            {i < stages.length - 1 && <ArrowRight size={14} className="text-slate-600 flex-shrink-0 mx-1" />}
          </React.Fragment>
        ))}
      </div>

      {/* Shape tracker */}
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        <div className="flex items-center gap-3 flex-wrap text-xs font-mono">
          {[
            { label: 'sampled_pca', shape: '(N, 12)', active: activeStage >= 0, color: 'amber' },
            { label: 'theta_unscaled', shape: '(N, 12)', active: activeStage >= 1, color: 'blue' },
            { label: 'flat', shape: '(N, 1920)', active: activeStage >= 2, color: 'purple' },
            { label: 'trajectories', shape: '(N, 240, 8)', active: activeStage >= 3, color: 'green' },
            { label: 'physical', shape: '(N, 240, 8) — real units', active: activeStage >= 3, color: 'green' },
          ].map((item, i) => (
            <React.Fragment key={i}>
              <div className={`flex flex-col items-center transition-opacity ${item.active ? 'opacity-100' : 'opacity-30'}`}>
                <Pill color={item.color}>{item.label}</Pill>
                <span className="text-[9px] text-slate-500 mt-0.5">{item.shape}</span>
              </div>
              {i < 4 && <ArrowRight size={12} className={`text-slate-600 transition-opacity ${item.active ? 'opacity-100' : 'opacity-20'}`} />}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
        {StageDesc[activeStage]}
      </div>

      {/* Sensor output preview */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
        <div className="text-xs font-bold text-slate-300 mb-2">Final output — 8 physical sensor trajectories (240 timesteps = 4 hours)</div>
        <div className="grid grid-cols-4 gap-2">
          {SENSORS.map((s, i) => (
            <div key={i} className="bg-slate-900 rounded-lg border border-slate-700 p-2">
              <div className="text-[9px] text-slate-400 mb-1 font-mono">{s}</div>
              {/* Tiny sparkline */}
              <svg viewBox="0 0 60 20" className="w-full h-5">
                {Array.from({ length: 3 }, (_, draw) => {
                  const pts = Array.from({ length: 20 }, (_, t) => {
                    const x = t * 3;
                    const y = 10 + (Math.sin(t * 0.4 + draw + i) * 4) + (Math.random() - 0.5) * 2;
                    return `${x},${y}`;
                  }).join(' ');
                  return (
                    <polyline key={draw} points={pts} fill="none"
                      stroke={`hsl(${i * 40 + draw * 30}, 60%, 55%)`} strokeWidth="0.8" strokeOpacity="0.5" />
                  );
                })}
                {/* True value line */}
                <polyline
                  points={Array.from({ length: 20 }, (_, t) => `${t * 3},${10 + Math.sin(t * 0.4 + i) * 4}`).join(' ')}
                  fill="none" stroke="white" strokeWidth="1.2" strokeDasharray="2,1" />
              </svg>
              <div className="text-[8px] text-slate-500 mt-0.5 font-mono">
                <span className="text-slate-300">—</span> predicted  <span className="text-white">- -</span> true
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// STEP 6  — Compare & Score
// ─────────────────────────────────────────────────────────────
const Step6_Compare = () => {
  const [metric, setMetric] = useState('nll');

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-900 rounded-xl border border-amber-700/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Target size={16} className="text-amber-400" />
            <span className="text-sm font-bold text-slate-200">θ_predicted (from flow)</span>
            <Pill color="amber">never seen during training</Pill>
          </div>
          <div className="grid grid-cols-3 gap-1">
            {Array.from({ length: 12 }, (_, i) => (
              <div key={i} className="text-[10px] font-mono bg-amber-900/30 border border-amber-700/50 rounded px-1.5 py-1 text-amber-200 text-center">
                PC{String(i + 1).padStart(2, '0')}: {(Math.random() * 3 - 1.5).toFixed(2)}
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900 rounded-xl border border-blue-700/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Database size={16} className="text-blue-400" />
            <span className="text-sm font-bold text-slate-200">θ_true (from test.parquet)</span>
            <Pill color="blue">the answer key</Pill>
          </div>
          <div className="grid grid-cols-3 gap-1">
            {Array.from({ length: 12 }, (_, i) => (
              <div key={i} className="text-[10px] font-mono bg-blue-900/30 border border-blue-700/50 rounded px-1.5 py-1 text-blue-200 text-center">
                PC{String(i + 1).padStart(2, '0')}: {(Math.random() * 3 - 1.5).toFixed(2)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-slate-900 rounded-xl border border-slate-700 p-5">
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm font-bold text-slate-200">Evaluation Metrics</span>
          <div className="flex gap-2">
            {[
              { id: 'nll', label: 'NLL Score' },
              { id: 'mae', label: 'MAE per PC' },
              { id: 'coverage', label: 'Coverage' },
            ].map(m => (
              <button key={m.id} onClick={() => setMetric(m.id)}
                className={`text-xs px-2.5 py-1 rounded font-bold border transition-all
                  ${metric === m.id ? 'bg-blue-600 border-blue-400 text-white' : 'bg-slate-800 border-slate-600 text-slate-300'}`}>
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {metric === 'nll' && (
          <div className="space-y-3">
            <p className="text-xs text-slate-400 leading-relaxed">
              The <strong className="text-white">NLL (Negative Log-Likelihood)</strong> measures how much probability the model
              assigns to the true θ. Lower = model says "this true value is very plausible".
              Positive NLL means the model thinks the true value is unlikely.
            </p>
            <div className="bg-slate-950 rounded p-3 space-y-0.5">
              <CodeLine><Comment c="# During evaluate.py — compute_nll_and_z()" /></CodeLine>
              <CodeLine><Var c="z_final" />, <Var c="log_det" /> = <Var c="model" />.<Fn c="forward" />(<Var c="theta_true" />, <Var c="condition" />)</CodeLine>
              <CodeLine><Comment c="# Push theta_true through the flow (forward) to get its latent z" /></CodeLine>
              <CodeLine><Var c="blueprint_score" /> = <Var c="blueprint" />.<Fn c="log_prob" />(<Var c="z_final" />)</CodeLine>
              <CodeLine><Var c="nll" /> = -(<Var c="blueprint_score" /> + <Var c="log_det" />)</CodeLine>
              <CodeLine><Comment c="# nll per sample — lower = model more confident" /></CodeLine>
            </div>
          </div>
        )}

        {metric === 'mae' && (
          <div className="space-y-3">
            <p className="text-xs text-slate-400 leading-relaxed">
              For each PC, we sample many z's, run inference, take the mean prediction, and compare to the true value.
              This gives per-component reconstruction quality.
            </p>
            <div className="flex gap-2 items-end h-20">
              {Array.from({ length: 12 }, (_, i) => {
                const h = Math.random() * 60 + 10;
                return (
                  <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
                    <div className="w-full rounded-t" style={{ height: h, background: `hsl(${240 - h * 2}, 70%, 55%)` }} />
                    <span className="text-[7px] text-slate-500 font-mono">{i + 1}</span>
                  </div>
                );
              })}
            </div>
            <div className="text-[10px] text-slate-500 font-mono text-center">PC index — bar height = MAE (lower = better)</div>
          </div>
        )}

        {metric === 'coverage' && (
          <div className="space-y-3">
            <p className="text-xs text-slate-400 leading-relaxed">
              For k = 1, 2, 3 standard deviations: what fraction of true θ values fall inside the predicted interval?
              A well-calibrated model should hit ~68%, ~95%, ~99.7%.
            </p>
            <div className="space-y-2">
              {[
                { k: 1, nominal: 68.3, empirical: 64 },
                { k: 2, nominal: 95.4, empirical: 91 },
                { k: 3, nominal: 99.7, empirical: 96 },
              ].map(({ k, nominal, empirical }) => (
                <div key={k} className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 w-14 font-mono flex-shrink-0">±{k}σ band</span>
                  <div className="flex-1 bg-slate-800 rounded-full h-5 relative overflow-hidden">
                    <div className="absolute inset-y-0 left-0 bg-blue-700/40 rounded-full" style={{ width: `${nominal}%` }} />
                    <div className="absolute inset-y-0 left-0 bg-emerald-500 rounded-full transition-all" style={{ width: `${empirical}%` }} />
                  </div>
                  <span className="text-xs font-mono w-20 flex-shrink-0">
                    <span className="text-emerald-300">{empirical}%</span>
                    <span className="text-slate-500"> / {nominal}%</span>
                  </span>
                </div>
              ))}
            </div>
            <div className="flex gap-3 text-[10px]">
              <span className="flex items-center gap-1"><span className="w-3 h-2 bg-emerald-500 rounded inline-block" /> empirical</span>
              <span className="flex items-center gap-1"><span className="w-3 h-2 bg-blue-700/40 rounded inline-block" /> nominal (ideal)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// STEP 7  — Full End-to-End Summary
// ─────────────────────────────────────────────────────────────
const Step7_Summary = () => {
  const steps = [
    { icon: Database, color: 'text-blue-400', bg: 'bg-blue-900/30 border-blue-700', title: 'Load test row', body: 'Read x (3 dims) from test.parquet. Keep θ_true (12 dims) aside as the answer key. Ignore u.' },
    { icon: Shuffle, color: 'text-purple-400', bg: 'bg-purple-900/30 border-purple-700', title: 'Draw z ~ N(0,1)', body: 'Sample 12 independent standard normals. Repeat N times for N different plausible futures.' },
    { icon: RefreshCw, color: 'text-green-400', bg: 'bg-green-900/30 border-green-700', title: 'Reversed layer loop', body: 'For layers [3,2,1,0]: flip z → split [z₁|z₂] → brain([z₁,x]) → MLP → inverse-spline z₂ → reassemble.' },
    { icon: BarChart2, color: 'text-amber-400', bg: 'bg-amber-900/30 border-amber-700', title: '12 PCA coefficients out', body: 'After 4 layers, z has been fully warped into θ̂ — a 12-dim coordinate in PCA coefficient space.' },
    { icon: Layers, color: 'text-indigo-400', bg: 'bg-indigo-900/30 border-indigo-700', title: 'Decode to physical space', body: 'StandardScaler⁻¹ → @ PCA.components_ + mean_ → reshape (240,8) → RobustScaler⁻¹ → real sensor units.' },
    { icon: CheckCircle, color: 'text-emerald-400', bg: 'bg-emerald-900/30 border-emerald-700', title: 'Compare & evaluate', body: 'NLL score, MAE per PC, coverage calibration, trajectory band (5–95%) vs true black dashed line.' },
  ];

  return (
    <div className="flex flex-col gap-6">
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-5">
        <div className="text-xs font-bold text-slate-300 mb-4 flex items-center gap-2">
          <Activity size={16} className="text-blue-400" /> Complete Inference Pipeline at a Glance
        </div>
        <div className="flex flex-col gap-0">
          {steps.map(({ icon: Icon, color, bg, title, body }, i) => (
            <div key={i} className="flex gap-3">
              <div className="flex flex-col items-center">
                <div className={`w-8 h-8 rounded-full border flex items-center justify-center flex-shrink-0 ${bg}`}>
                  <Icon size={14} className={color} />
                </div>
                {i < steps.length - 1 && <div className="w-0.5 bg-slate-700 flex-1 my-1" style={{ minHeight: 16 }} />}
              </div>
              <div className="pb-4 pt-1">
                <div className="text-xs font-bold text-slate-200 mb-0.5">
                  <span className="text-slate-500 mr-1">{i + 1}.</span>{title}
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">{body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-emerald-900/20 border border-emerald-700 rounded-xl p-4">
          <div className="text-xs font-bold text-emerald-300 mb-2">✅ What x does</div>
          <p className="text-xs text-emerald-200/80 leading-relaxed">
            x shapes the spline boxes inside every MLP call. It <em>conditions</em> the warp so that 
            the same z drawn from N(0,1) produces a physically plausible trajectory 
            for <em>that specific</em> operating point.
          </p>
        </div>
        <div className="bg-red-900/20 border border-red-700 rounded-xl p-4">
          <div className="text-xs font-bold text-red-300 mb-2">❌ What x does NOT do</div>
          <p className="text-xs text-red-200/80 leading-relaxed">
            x is never a source of "information about the future" on its own.
            It cannot predict θ_true by itself. The diversity in predictions 
            comes entirely from sampling different z values.
          </p>
        </div>
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
        <div className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1.5"><GitMerge size={14} className="text-purple-400" /> Key contrast: Training vs Inference</div>
        <div className="grid grid-cols-2 gap-4">
          {[
            {
              label: 'Training (θ → z)',
              color: 'border-blue-700 bg-blue-900/20',
              header: 'text-blue-300',
              items: [
                'Input: [θ_true, x] from train.parquet',
                'Direction: FORWARD through layers 0→1→2→3',
                'Flip applied AFTER each layer',
                'Learns: spline params that push θ to N(0,1)',
                'Loss: NLL = -log p(θ|x)',
              ]
            },
            {
              label: 'Inference (z → θ)',
              color: 'border-emerald-700 bg-emerald-900/20',
              header: 'text-emerald-300',
              items: [
                'Input: [z~N(0,1), x] from test.parquet',
                'Direction: BACKWARD through layers 3→2→1→0',
                'Flip undone BEFORE each layer',
                'Reuses: EXACT SAME spline params (frozen)',
                'Output: θ̂ in PCA coefficient space',
              ]
            }
          ].map(({ label, color, header, items }) => (
            <div key={label} className={`rounded-lg border p-3 ${color}`}>
              <div className={`text-xs font-bold mb-2 ${header}`}>{label}</div>
              <ul className="space-y-1">
                {items.map((it, i) => (
                  <li key={i} className="text-[11px] text-slate-400 flex items-start gap-1.5">
                    <span className="mt-0.5 flex-shrink-0">•</span>{it}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────

const SLIDES = [
  {
    id: 0,
    title: 'The Starting Point',
    subtitle: 'What we load from test.parquet',
    icon: Database,
    accentColor: 'blue',
    component: Step0_TestParquet,
  },
  {
    id: 1,
    title: 'Sample z ~ N(0,1)',
    subtitle: 'The source of all predictions',
    icon: Shuffle,
    accentColor: 'purple',
    component: Step1_SampleZ,
  },
  {
    id: 2,
    title: 'The Reversed Loop',
    subtitle: '4 coupling layers, backward',
    icon: RefreshCw,
    accentColor: 'green',
    component: Step2_ReverseLoop,
  },
  {
    id: 3,
    title: 'Inside ONE Layer',
    subtitle: 'flip → split → brain → spline⁻¹ → concat',
    icon: Cpu,
    accentColor: 'indigo',
    component: Step3_InsideLayer,
  },
  {
    id: 4,
    title: '12 PCA Coefficients Out',
    subtitle: 'θ̂ — the predicted scenario coordinates',
    icon: BarChart2,
    accentColor: 'amber',
    component: Step4_PCOutput,
  },
  {
    id: 5,
    title: 'Decode to Physical Space',
    subtitle: 'StandardScaler⁻¹ → PCA⁻¹ → RobustScaler⁻¹',
    icon: Layers,
    accentColor: 'pink',
    component: Step5_Decode,
  },
  {
    id: 6,
    title: 'Compare & Score',
    subtitle: 'NLL, MAE, Coverage Calibration',
    icon: Target,
    accentColor: 'emerald',
    component: Step6_Compare,
  },
  {
    id: 7,
    title: 'Full Pipeline Summary',
    subtitle: 'Everything in one view',
    icon: Activity,
    accentColor: 'slate',
    component: Step7_Summary,
  },
];

const ACCENT_STYLES = {
  blue:    { active: 'bg-blue-600 border-blue-400 text-white', dot: 'bg-blue-400', ring: 'ring-blue-500', header: 'from-blue-900/60 border-blue-700' },
  purple:  { active: 'bg-purple-600 border-purple-400 text-white', dot: 'bg-purple-400', ring: 'ring-purple-500', header: 'from-purple-900/60 border-purple-700' },
  green:   { active: 'bg-green-700 border-green-400 text-white', dot: 'bg-green-400', ring: 'ring-green-500', header: 'from-green-900/60 border-green-700' },
  indigo:  { active: 'bg-indigo-600 border-indigo-400 text-white', dot: 'bg-indigo-400', ring: 'ring-indigo-500', header: 'from-indigo-900/60 border-indigo-700' },
  amber:   { active: 'bg-amber-600 border-amber-400 text-white', dot: 'bg-amber-400', ring: 'ring-amber-500', header: 'from-amber-900/60 border-amber-700' },
  pink:    { active: 'bg-pink-700 border-pink-400 text-white', dot: 'bg-pink-400', ring: 'ring-pink-500', header: 'from-pink-900/60 border-pink-700' },
  emerald: { active: 'bg-emerald-700 border-emerald-400 text-white', dot: 'bg-emerald-400', ring: 'ring-emerald-500', header: 'from-emerald-900/60 border-emerald-700' },
  slate:   { active: 'bg-slate-600 border-slate-400 text-white', dot: 'bg-slate-400', ring: 'ring-slate-400', header: 'from-slate-800 border-slate-600' },
};

export default function InferenceLogicSection() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [animDir, setAnimDir] = useState('right');
  const autoRef = useRef(null);

  const goTo = (idx, dir) => {
    setAnimDir(dir ?? (idx > currentSlide ? 'right' : 'left'));
    setCurrentSlide(idx);
  };

  const prev = () => { if (currentSlide > 0) goTo(currentSlide - 1, 'left'); };
  const next = () => { if (currentSlide < SLIDES.length - 1) goTo(currentSlide + 1, 'right'); };

  useEffect(() => {
    if (autoPlay) {
      autoRef.current = setInterval(() => {
        setCurrentSlide(s => {
          if (s >= SLIDES.length - 1) { setAutoPlay(false); return s; }
          setAnimDir('right');
          return s + 1;
        });
      }, 4000);
    }
    return () => clearInterval(autoRef.current);
  }, [autoPlay]);

  const slide = SLIDES[currentSlide];
  const accent = ACCENT_STYLES[slide.accentColor] || ACCENT_STYLES.blue;
  const SlideComponent = slide.component;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold text-slate-800">Inference Logic</h2>
        <p className="text-slate-600 leading-relaxed mt-1">
          Step-by-step breakdown of how the trained normalizing flow generates 4-hour SCADA trajectories
          at test time — from sampling z to physical sensor units.
        </p>
      </div>

      {/* Slide selector dots + autoplay */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex gap-1.5">
          {SLIDES.map((s, i) => (
            <button key={i} onClick={() => goTo(i)}
              title={s.title}
              className={`transition-all rounded-full border ${i === currentSlide
                ? `w-6 h-3 ${ACCENT_STYLES[s.accentColor].dot} border-transparent`
                : 'w-3 h-3 bg-slate-300 border-slate-400 hover:bg-slate-400'}`} />
          ))}
        </div>
        <button onClick={() => setAutoPlay(a => !a)}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-bold border transition-all
            ${autoPlay ? 'bg-red-700 border-red-500 text-white' : 'bg-slate-700 border-slate-500 text-slate-200 hover:bg-slate-600'}`}>
          {autoPlay ? <><Pause size={12} /> Stop</> : <><Play size={12} /> Auto-play</>}
        </button>
        <span className="text-xs text-slate-500">{currentSlide + 1} / {SLIDES.length}</span>
      </div>

      {/* Slide thumbnail strip */}
      <div className="flex gap-2 overflow-x-auto pb-1">
        {SLIDES.map((s, i) => {
          const Icon = s.icon;
          const a = ACCENT_STYLES[s.accentColor];
          return (
            <button key={i} onClick={() => goTo(i)}
              className={`flex-shrink-0 flex flex-col items-center gap-1 px-3 py-2 rounded-xl border text-center transition-all w-28
                ${i === currentSlide ? `${a.active} shadow-lg` : 'bg-slate-800 border-slate-600 text-slate-400 hover:border-slate-400'}`}>
              <Icon size={16} />
              <span className="text-[9px] font-bold leading-tight">{s.title}</span>
            </button>
          );
        })}
      </div>

      {/* Main slide card */}
      <div className="bg-slate-900 rounded-2xl border border-slate-700 overflow-hidden shadow-xl">
        {/* Slide header */}
        <div className={`bg-gradient-to-r ${accent.header} border-b px-6 py-4 flex items-center justify-between`}>
          <div className="flex items-center gap-3">
            <div className={`w-9 h-9 rounded-xl flex items-center justify-center bg-slate-900/50 border border-slate-600`}>
              <slide.icon size={18} className="text-slate-200" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-slate-400 font-mono">Step {currentSlide + 1}/{SLIDES.length}</span>
                <span className={`text-[10px] px-2 py-0.5 rounded font-bold border ${accent.active}`}>
                  {slide.accentColor.toUpperCase()}
                </span>
              </div>
              <h3 className="text-lg font-bold text-white leading-tight">{slide.title}</h3>
              <p className="text-xs text-slate-400">{slide.subtitle}</p>
            </div>
          </div>
          {/* Progress bar */}
          <div className="hidden md:flex flex-col items-end gap-1">
            <div className="w-32 bg-slate-700 rounded-full h-1.5 overflow-hidden">
              <div className={`h-full rounded-full transition-all duration-500 ${accent.dot}`}
                style={{ width: `${((currentSlide + 1) / SLIDES.length) * 100}%` }} />
            </div>
            <span className="text-[10px] text-slate-500">{Math.round(((currentSlide + 1) / SLIDES.length) * 100)}% complete</span>
          </div>
        </div>

        {/* Slide body */}
        <div className="p-6 min-h-[420px]">
          <SlideComponent />
        </div>

        {/* Navigation footer */}
        <div className="border-t border-slate-700 px-6 py-3 flex items-center justify-between bg-slate-900/50">
          <NavBtn onClick={prev} disabled={currentSlide === 0}>
            <ChevronLeft size={14} /> Previous
          </NavBtn>

          <div className="flex gap-1">
            {SLIDES.map((_, i) => (
              <button key={i} onClick={() => goTo(i)}
                className={`w-2 h-2 rounded-full transition-all ${i === currentSlide ? `${accent.dot} scale-125` : 'bg-slate-600 hover:bg-slate-400'}`} />
            ))}
          </div>

          <NavBtn onClick={next} disabled={currentSlide === SLIDES.length - 1} primary>
            Next <ChevronRight size={14} />
          </NavBtn>
        </div>
      </div>

      {/* Quick reference callout */}
      <div className="bg-slate-100 border border-slate-300 rounded-xl p-4 text-sm text-slate-700 leading-relaxed">
        <strong>TL;DR:</strong> Inference = draw z from N(0,1) → run it backward through 4 coupling layers, 
        each conditioned on x (your current sensor reading) → get 12 PCA coefficients → 
        decode to 240 × 8 physical sensor trajectory.
        <code className="bg-slate-200 px-1 rounded ml-1 text-xs">θ_true</code> is only ever used to check accuracy — 
        it is <em>never</em> seen by the model during generation.
      </div>
    </div>
  );
}
