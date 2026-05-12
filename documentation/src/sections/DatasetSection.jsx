import React, { useState, useEffect } from 'react';
import {
  Database,
  ChevronRight,
  ChevronLeft,
  Activity,
  Terminal,
  Play,
  Pause,
  Layers,
  GitMerge,
  Box,
  ArrowRight,
  CheckCircle,
  Cpu,
  RefreshCw,
} from 'lucide-react';

// ─── Syntax highlighter ───────────────────────────────────────────────────────

const highlightCode = (code) => {
  if (!code) return '';
  let html = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const tokens = [];
  const push = (m, cls) => { tokens.push(`<span class="${cls}">${m}</span>`); return `TOKENz${tokens.length - 1}z`; };
  html = html.replace(/(#[^\n]*)/g, m => push(m, 'text-slate-500 italic'));
  html = html.replace(/('.*?'|".*?")/g, m => push(m, 'text-emerald-300'));
  html = html.replace(/\b(\d+\.\d+|\d+)\b/g, m => push(m, 'text-purple-300'));
  html = html.replace(/\b(pd\.read_parquet|os\.path\.join|torch\.tensor|torch\.cat|torch\.float32|Dataset|DataLoader)\b/g, m => push(m, 'text-amber-300'));
  html = html.replace(/\b(def|class|if|else|elif|for|return|import|from|as|not|in|True|False|self|super)\b/g, m => push(m, 'text-rose-400 font-bold'));
  html = html.replace(/\b(split|data_path|df|x_tensor|u_tensor|theta_tensor|condition_tensor|x_cols|u_cols|theta_cols|idx|split_path|batch|theta|condition|device)\b/g, m => push(m, 'text-blue-300 italic'));
  for (let i = tokens.length - 1; i >= 0; i--) html = html.replace(`TOKENz${i}z`, tokens[i]);
  return html;
};

// ─── Shared button ────────────────────────────────────────────────────────────

const VisualButton = ({ onClick, disabled, children, active }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-bold transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed z-10 ${
      active
        ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/50'
        : 'bg-slate-700 hover:bg-slate-600 text-slate-200 shadow-slate-900/50 border border-slate-600'
    }`}
  >
    {children}
  </button>
);

// ─── Visual 1 · Column Extraction ────────────────────────────────────────────

const AnimatedColumnExtraction = () => {
  const [phase, setPhase] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX = 3;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setPhase(p => { if (p >= MAX) { setIsPlaying(false); return p; } return p + 1; }), 1600);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  const allCols = [
    { name: 'COMP_Suction_Pressure',      group: 'x', color: 'cyan'   },
    { name: 'COMP_Suction_Drum_Temp',     group: 'x', color: 'cyan'   },
    { name: 'KPI_Fuel_Gas_LHV',           group: 'x', color: 'cyan'   },
    { name: 'Turbine_SHAFT_SPEED',        group: 'u', color: 'sky'    },
    { name: '14PDCV-504_Position',        group: 'u', color: 'sky'    },
    { name: 'SEAL_GAS_FLOW',              group: 'u', color: 'sky'    },
    { name: 'PCA_Coefficient_1',          group: 'θ', color: 'violet' },
    { name: 'PCA_Coefficient_2',          group: 'θ', color: 'violet' },
    { name: '... PCA_Coefficient_12',     group: 'θ', color: 'violet' },
  ];

  const colorCls = {
    cyan:   { bg: 'bg-cyan-900/40',   border: 'border-cyan-500/60',   text: 'text-cyan-300'   },
    sky:    { bg: 'bg-sky-900/40',    border: 'border-sky-500/60',    text: 'text-sky-300'    },
    violet: { bg: 'bg-violet-900/40', border: 'border-violet-500/60', text: 'text-violet-300' },
  };

  const isHighlighted = (col) => {
    if (phase === 0) return false;
    if (phase === 1) return col.group === 'x';
    if (phase === 2) return col.group === 'x' || col.group === 'u';
    return true;
  };

  const phaseDesc = [
    'The parquet file contains all sensor columns side by side. dataset.py reads the whole file then selects specific column groups.',
    'x_cols (3): The instantaneous process state — suction pressure, temperature, fuel LHV. These describe what the compressor is currently experiencing.',
    'u_cols (3): Control inputs — shaft speed, valve position, seal gas flow. Together with x_cols they form the 6-dimensional condition vector.',
    'theta_cols (12): The 12 PCA coefficients written by preprocessing.py. This is the 4-hour trajectory the flow model learns to generate.',
  ];

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">Column Groups — What dataset.py Extracts</div>
        <div className="w-full max-w-lg mx-auto flex flex-col gap-1.5">
          {allCols.map((col, i) => {
            const active = isHighlighted(col);
            const c = colorCls[col.color];
            return (
              <div
                key={i}
                className={`flex items-center gap-3 px-3 py-1.5 rounded-lg border transition-all duration-400 ${active ? `${c.bg} ${c.border}` : 'bg-slate-800 border-slate-700'}`}
                style={{ transitionDelay: `${i * 40}ms` }}
              >
                <span className={`text-[9px] font-bold font-mono w-4 ${active ? c.text : 'text-slate-600'}`}>{col.group}</span>
                <span className={`font-mono text-[10px] transition-colors ${active ? c.text : 'text-slate-500'}`}>{col.name}</span>
                {active && <CheckCircle size={10} className={`ml-auto flex-shrink-0 ${c.text}`} />}
              </div>
            );
          })}
        </div>
        {phase > 0 && (
          <div className="flex gap-2 justify-center flex-wrap">
            {phase >= 1 && <div className="flex items-center gap-1.5 bg-cyan-900/30 border border-cyan-500/40 rounded-lg px-3 py-1"><span className="text-[10px] font-bold text-cyan-300 font-mono">x_cols[3]</span><span className="text-[9px] text-slate-400">process state</span></div>}
            {phase >= 2 && <div className="flex items-center gap-1.5 bg-sky-900/30 border border-sky-500/40 rounded-lg px-3 py-1"><span className="text-[10px] font-bold text-sky-300 font-mono">u_cols[3]</span><span className="text-[9px] text-slate-400">control inputs</span></div>}
            {phase >= 3 && <div className="flex items-center gap-1.5 bg-violet-900/30 border border-violet-500/40 rounded-lg px-3 py-1"><span className="text-[10px] font-bold text-violet-300 font-mono">theta_cols[12]</span><span className="text-[9px] text-slate-400">PCA trajectory</span></div>}
          </div>
        )}
        <p className="text-[11px] text-slate-300 leading-relaxed text-center px-2 max-w-lg mx-auto bg-slate-800/50 border border-slate-700 rounded-xl p-3">
          {phaseDesc[phase]}
        </p>
      </div>
      <div className="flex-shrink-0 flex justify-center gap-3 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setPhase(p => Math.max(0, p - 1))} disabled={phase === 0 || isPlaying}><ChevronLeft size={14} /> Prev</VisualButton>
        <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying}>{isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play'}</VisualButton>
        <VisualButton onClick={() => setPhase(p => Math.min(MAX, p + 1))} disabled={phase === MAX || isPlaying}>Next <ChevronRight size={14} /></VisualButton>
      </div>
    </div>
  );
};

// ─── Visual 2 · Tensor Assembly ───────────────────────────────────────────────

const AnimatedTensorAssembly = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX = 5;

  useEffect(() => {
    let int;
    if (isPlaying) int = setInterval(() => setStep(s => { if (s >= MAX) { setIsPlaying(false); return s; } return s + 1; }), 1500);
    return () => clearInterval(int);
  }, [isPlaying]);

  const show = (n) => step >= n;

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-3 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">Tensor Construction — __init__ to __getitem__</div>
        <div className="w-full max-w-lg mx-auto flex flex-col gap-3">

          <div className={`transition-all duration-500 ${show(0) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            <div className="flex items-center gap-3 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3">
              <Database size={16} className="text-slate-400 flex-shrink-0" />
              <div className="flex flex-col">
                <code className="text-[11px] font-bold text-slate-300 font-mono">pd.read_parquet(<span className="text-emerald-300">"data/processed/train.parquet"</span>)</code>
                <span className="text-[9px] text-slate-500 mt-0.5">DataFrame — N rows × 18 columns</span>
              </div>
            </div>
          </div>

          <div className={`flex justify-center transition-all duration-300 ${show(1) ? 'opacity-100' : 'opacity-0'}`}><div className="w-px h-5 bg-slate-600"></div></div>

          <div className={`transition-all duration-500 ${show(1) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: 'x_cols[3]',      color: 'cyan',   items: ['Suction_P','Suction_T','Fuel_LHV'] },
                { label: 'u_cols[3]',      color: 'sky',    items: ['Shaft_Speed','Valve_504','Seal_Gas'] },
                { label: 'theta_cols[12]', color: 'violet', items: ['PCA_1','PCA_2','...PCA_12'] },
              ].map(g => (
                <div key={g.label} className={`rounded-lg border p-2 bg-${g.color}-900/30 border-${g.color}-500/40`}>
                  <div className={`text-[9px] font-bold font-mono text-${g.color}-300 mb-1.5`}>{g.label}</div>
                  {g.items.map(it => <div key={it} className={`text-[8px] font-mono text-${g.color}-400/80 truncate`}>{it}</div>)}
                </div>
              ))}
            </div>
          </div>

          <div className={`flex justify-center transition-all duration-300 ${show(2) ? 'opacity-100' : 'opacity-0'}`}><div className="w-px h-5 bg-slate-600"></div></div>

          <div className={`transition-all duration-500 ${show(2) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 flex flex-col gap-1.5">
              <div className="text-[9px] text-slate-500 uppercase font-bold mb-1">Convert to float32 tensors in RAM</div>
              {[
                { name: 'x_tensor',     shape: '[N,  3]', color: 'cyan'   },
                { name: 'u_tensor',     shape: '[N,  3]', color: 'sky'    },
                { name: 'theta_tensor', shape: '[N, 12]', color: 'violet' },
              ].map(t => (
                <div key={t.name} className={`flex items-center gap-2 px-2 py-1 rounded-lg bg-${t.color}-900/20 border border-${t.color}-500/30`}>
                  <code className={`font-mono text-[10px] font-bold text-${t.color}-300`}>{t.name}</code>
                  <span className="text-slate-500 text-[9px]">= torch.tensor(df[cols].values, float32)</span>
                  <span className={`text-[9px] text-${t.color}-400 font-mono ml-auto`}>{t.shape}</span>
                </div>
              ))}
            </div>
          </div>

          <div className={`flex justify-center transition-all duration-300 ${show(3) ? 'opacity-100' : 'opacity-0'}`}><div className="w-px h-5 bg-slate-600"></div></div>

          <div className={`transition-all duration-500 ${show(3) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            <div className="flex items-center gap-2 bg-slate-800 border border-slate-700 rounded-xl p-3">
              <div className="flex flex-col gap-1 flex-1">
                <div className="h-7 bg-cyan-900/30 border border-cyan-500/40 rounded text-[10px] font-mono text-cyan-300 flex items-center px-2 font-bold">x_tensor [N, 3]</div>
                <div className="h-7 bg-sky-900/30 border border-sky-500/40 rounded text-[10px] font-mono text-sky-300 flex items-center px-2 font-bold">u_tensor [N, 3]</div>
              </div>
              <div className="text-slate-400 font-bold text-sm">cat()</div>
              <ArrowRight size={14} className="text-slate-500" />
              <div className="flex-1 h-16 bg-indigo-900/30 border-2 border-indigo-500/60 rounded-xl flex flex-col items-center justify-center gap-1 shadow-[0_0_12px_rgba(99,102,241,0.2)]">
                <code className="font-mono text-[11px] font-extrabold text-indigo-300">condition_tensor</code>
                <span className="text-[9px] text-indigo-400 font-mono">[N, 6]</span>
              </div>
            </div>
          </div>

          <div className={`flex justify-center transition-all duration-300 ${show(4) ? 'opacity-100' : 'opacity-0'}`}><div className="w-px h-5 bg-slate-600"></div></div>

          <div className={`transition-all duration-500 ${show(4) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            <div className="bg-indigo-900/10 border-2 border-indigo-500/40 rounded-xl p-3">
              <div className="text-[9px] text-slate-500 font-bold uppercase mb-2">__getitem__(idx) returns per sample:</div>
              <div className="flex gap-3">
                <div className="flex-1 bg-violet-900/20 border border-violet-500/40 rounded-lg p-2.5 text-center">
                  <code className="font-mono text-sm font-extrabold text-violet-300 block mb-1">theta</code>
                  <span className="text-[9px] font-mono text-violet-400">[12]</span>
                  <p className="text-[8px] text-slate-500 mt-1 leading-tight">PCA coefficients<br/>= "what will happen"</p>
                </div>
                <div className="flex-1 bg-indigo-900/20 border border-indigo-500/40 rounded-lg p-2.5 text-center">
                  <code className="font-mono text-sm font-extrabold text-indigo-300 block mb-1">condition</code>
                  <span className="text-[9px] font-mono text-indigo-400">[6]</span>
                  <p className="text-[8px] text-slate-500 mt-1 leading-tight">x + u concatenated<br/>= "current state"</p>
                </div>
              </div>
            </div>
          </div>

          <div className={`transition-all duration-500 ${show(5) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            <div className="flex items-center gap-2 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2">
              <GitMerge size={13} className="text-amber-400 flex-shrink-0" />
              <code className="text-[10px] font-mono text-amber-300">DataLoader(dataset, batch_size=<span className="text-purple-300">1024</span>, num_workers=<span className="text-purple-300">8</span>)</code>
              <ArrowRight size={12} className="text-slate-500 ml-auto" />
              <span className="text-[9px] text-amber-400 font-mono flex-shrink-0">batch[1024, ·]</span>
            </div>
          </div>

        </div>
      </div>
      <div className="flex-shrink-0 flex justify-center gap-3 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setStep(p => Math.max(0, p - 1))} disabled={step === 0 || isPlaying}><ChevronLeft size={14} /> Prev</VisualButton>
        <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying}>{isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play'}</VisualButton>
        <VisualButton onClick={() => setStep(p => Math.min(MAX, p + 1))} disabled={step === MAX || isPlaying}>Next <ChevronRight size={14} /></VisualButton>
      </div>
    </div>
  );
};

// ─── Visual 3 · In-Memory Layout ──────────────────────────────────────────────

const AnimatedMemoryLayout = () => {
  const [showSplit, setShowSplit] = useState(false);
  const [hoveredSplit, setHoveredSplit] = useState(null);

  const splits = [
    { name: 'train', pct: 70, color: 'blue',  rows: '~N×0.70 rows', note: 'Largest block — model learns from this' },
    { name: 'val',   pct: 15, color: 'amber', rows: '~N×0.15 rows', note: 'Validation: no weight updates, checks overfit' },
    { name: 'test',  pct: 15, color: 'rose',  rows: '~N×0.15 rows', note: 'Held out entirely until final evaluation' },
  ];

  const tensors = [
    { label: 'theta_tensor',     shape: '[N, 12]', dtype: 'float32', bytes: 'N × 48 bytes', color: 'violet', cols: 12 },
    { label: 'condition_tensor', shape: '[N,  6]', dtype: 'float32', bytes: 'N × 24 bytes', color: 'indigo', cols: 6  },
  ];

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-5 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">In-Memory Tensor Layout per Split</div>

        <div className="w-full max-w-lg mx-auto flex flex-col gap-3">
          <button onClick={() => setShowSplit(v => !v)} className="text-[10px] text-blue-400 font-bold underline underline-offset-2 self-center">
            {showSplit ? 'Collapse splits' : 'Show all 3 dataset splits'}
          </button>

          <div className={`flex gap-2 transition-all duration-500 ${showSplit ? 'opacity-100' : 'opacity-0 h-0 overflow-hidden'}`}>
            {splits.map(s => (
              <div
                key={s.name}
                onMouseEnter={() => setHoveredSplit(s.name)}
                onMouseLeave={() => setHoveredSplit(null)}
                className={`flex-1 rounded-xl border-2 p-3 text-center transition-all duration-200 ${hoveredSplit === s.name ? `bg-${s.color}-900/40 border-${s.color}-500/60` : 'bg-slate-800 border-slate-700'}`}
              >
                <div className={`font-mono text-sm font-extrabold text-${s.color}-300`}>{s.name}</div>
                <div className={`text-2xl font-black text-${s.color}-400 my-0.5`}>{s.pct}%</div>
                <div className="text-[8px] text-slate-500">{s.rows}</div>
                {hoveredSplit === s.name && <div className={`text-[8px] text-${s.color}-300/80 mt-1 leading-tight animate-in fade-in duration-150`}>{s.note}</div>}
              </div>
            ))}
          </div>

          <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
            <div className="text-[9px] font-bold text-slate-500 uppercase mb-3">Tensors held in unified RAM (per dataset instance)</div>
            <div className="flex flex-col gap-2">
              {tensors.map(t => (
                <div key={t.label} className={`rounded-lg border bg-${t.color}-900/20 border-${t.color}-500/30 px-3 py-2`}>
                  <div className="flex items-center justify-between mb-1">
                    <code className={`font-mono text-[11px] font-bold text-${t.color}-300`}>{t.label}</code>
                    <div className="flex gap-2">
                      <span className={`text-[9px] font-mono text-${t.color}-400 bg-${t.color}-900/40 border border-${t.color}-500/30 px-1.5 py-0.5 rounded`}>{t.shape}</span>
                      <span className="text-[9px] text-slate-500 font-mono">{t.dtype}</span>
                    </div>
                  </div>
                  <div className="text-[9px] text-slate-500">{t.bytes} per split</div>
                  <div className="mt-2 flex gap-0.5">
                    {Array.from({ length: t.cols }).map((_, i) => (
                      <div key={i} className="flex-1 rounded-[2px] h-3" style={{ background: `hsl(${240 + i * 15}, 60%, 40%)`, opacity: 0.7 }} />
                    ))}
                  </div>
                  <div className="text-[8px] text-slate-600 mt-0.5">← {t.cols} feature columns →</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-800/60 border border-slate-600 rounded-xl p-3 flex items-start gap-2">
            <Cpu size={14} className="text-emerald-400 flex-shrink-0 mt-0.5" />
            <div>
              <div className="text-[9px] font-bold text-emerald-400 uppercase mb-1">M4 Max — 128 GB Unified Memory</div>
              <p className="text-[9px] text-slate-400 leading-relaxed">
                Loading all parquet rows into float32 tensors once at startup eliminates disk I/O from the training hot path. On the M4 Max with 128 GB unified RAM, every <code className="text-blue-300">__getitem__</code> call is a single tensor slice — no file reads, no re-parsing, no CPU↔GPU copy overhead (MPS unified memory removes that cost entirely).
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── Visual 4 · Animated Walkthrough ─────────────────────────────────────────

const AnimatedWalkthrough = () => {
  const [lineIdx, setLineIdx] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);

  const lines = [
    { code: 'class SCADAPipelineDataset(Dataset):',                          note: 'Inherits from torch.utils.data.Dataset — gives DataLoader compatibility for free.', color: 'sky' },
    { code: '    def __init__(self, data_path, split):',                     note: 'Receives the path to data/processed/ and the split name ("train", "val", or "test").', color: 'sky' },
    { code: '        split_path = os.path.join(data_path, f"{split}.parquet")', note: 'Builds the exact file path: data/processed/train.parquet etc. Fixed from earlier os.path.dirname bug.', color: 'emerald' },
    { code: '        df = pd.read_parquet(split_path)',                      note: 'Reads the entire parquet file into a pandas DataFrame — one-time cost at dataset construction.', color: 'blue' },
    { code: '        x_t = torch.tensor(df[x_cols].values, dtype=torch.float32)', note: 'Converts the 3 state sensor columns to a float32 tensor and holds it in unified RAM.', color: 'cyan' },
    { code: '        u_t = torch.tensor(df[u_cols].values, dtype=torch.float32)', note: 'Same for the 3 control input columns.', color: 'sky' },
    { code: '        self.theta_tensor     = torch.tensor(df[theta_cols].values, dtype=torch.float32)', note: 'Stores all 12 PCA coefficient columns as the theta tensor — shape [N, 12].', color: 'violet' },
    { code: '        self.condition_tensor = torch.cat([x_t, u_t], dim=-1)', note: 'Concatenates x [N,3] and u [N,3] → condition [N,6]. Stored once, never recomputed.', color: 'indigo' },
    { code: '    def __len__(self):',                                        note: 'Required by Dataset protocol. Used by DataLoader to know total sample count.', color: 'amber' },
    { code: '        return len(self.theta_tensor)',                         note: '', color: 'amber' },
    { code: '    def __getitem__(self, idx):',                               note: 'Called once per sample index. DataLoader calls this 1024× per batch with shuffled indices.', color: 'rose' },
    { code: '        return {',                                              note: '', color: 'rose' },
    { code: '            "theta":     self.theta_tensor[idx],',             note: '"theta" — shape [12]. The target: what 4-hour trajectory looks like from this state.', color: 'violet' },
    { code: '            "condition": self.condition_tensor[idx],',         note: '"condition" — shape [6]. The input: what the compressor looks like right now.', color: 'indigo' },
    { code: '        }',                                                     note: '', color: 'rose' },
  ];

  const MAX = lines.length - 1;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setLineIdx(i => { if (i >= MAX) { setIsPlaying(false); return i; } return i + 1; }), 900);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  const borderCls = {
    sky:    'border-l-sky-500    bg-sky-900/10',
    emerald:'border-l-emerald-500 bg-emerald-900/10',
    blue:   'border-l-blue-500   bg-blue-900/10',
    cyan:   'border-l-cyan-500   bg-cyan-900/10',
    violet: 'border-l-violet-500 bg-violet-900/10',
    indigo: 'border-l-indigo-500 bg-indigo-900/10',
    amber:  'border-l-amber-500  bg-amber-900/10',
    rose:   'border-l-rose-500   bg-rose-900/10',
  };
  const noteCls = {
    sky:    'text-sky-300',    emerald:'text-emerald-300',
    blue:   'text-blue-300',  cyan:   'text-cyan-300',
    violet: 'text-violet-300',indigo: 'text-indigo-300',
    amber:  'text-amber-300', rose:   'text-rose-300',
  };

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-3 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">Animated Code Walkthrough — dataset.py</div>

        <div className="w-full max-w-2xl mx-auto bg-[#0d1117] rounded-xl border border-slate-700 overflow-hidden">
          {lines.map((ln, i) => (
            <div
              key={i}
              className={`flex items-start border-l-4 transition-all duration-300 ${
                lineIdx === i ? `${borderCls[ln.color]} opacity-100`
                : lineIdx > i ? 'border-l-slate-700 bg-slate-900/50 opacity-60'
                : 'border-l-transparent bg-transparent opacity-30'
              }`}
            >
              <span className="text-[9px] font-mono text-slate-600 w-7 flex-shrink-0 pt-2 pl-2 select-none">{i + 1}</span>
              <pre className="text-[10px] font-mono text-slate-300 py-1.5 px-2 flex-1 whitespace-pre-wrap leading-relaxed">{ln.code}</pre>
            </div>
          ))}
        </div>

        {lineIdx >= 0 && lines[lineIdx].note ? (
          <div key={lineIdx} className="w-full max-w-2xl mx-auto bg-slate-800 border border-slate-600 rounded-xl p-3 animate-in fade-in slide-in-from-bottom-2 duration-200">
            <span className={`text-[11px] leading-relaxed ${noteCls[lines[lineIdx].color]}`}>{lines[lineIdx].note}</span>
          </div>
        ) : lineIdx === -1 ? (
          <p className="text-center text-[11px] text-slate-500">Press Play or Step to walk through every line.</p>
        ) : null}
      </div>

      <div className="flex-shrink-0 flex justify-center gap-3 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => { setLineIdx(-1); setIsPlaying(false); }}><RefreshCw size={14} /> Reset</VisualButton>
        <VisualButton onClick={() => setLineIdx(p => Math.max(-1, p - 1))} disabled={lineIdx <= -1 || isPlaying}><ChevronLeft size={14} /> Back</VisualButton>
        <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying} disabled={lineIdx >= MAX}>{isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play All'}</VisualButton>
        <VisualButton onClick={() => setLineIdx(p => Math.min(MAX, p + 1))} disabled={lineIdx >= MAX || isPlaying}>Step <ChevronRight size={14} /></VisualButton>
      </div>
    </div>
  );
};

// ─── Step definitions ─────────────────────────────────────────────────────────

const steps = [
  {
    id: 'columns',
    title: '1. Column Selection',
    icon: Layers,
    description:
      'dataset.py reads a pre-split parquet file and extracts exactly three column groups: x_cols (3 process state sensors), u_cols (3 control inputs), and theta_cols (12 PCA trajectory coefficients written by preprocessing.py). Nothing else is read.',
    pytorch:
      'Why these three groups? x and u together describe the current operating point of the compressor — the "conditions" the future trajectory depends on. theta is the compressed 4-hour future. Keeping them as separate named groups makes the downstream concatenation (condition = cat(x, u)) explicit and easy to reconfigure without touching the model.',
    code: `# Column definitions at top of dataset.py

x_cols = [
    'COMP_Suction_Pressure',         # Process state
    'COMP_Suction_Drum_Temperature',
    'KPI_Fuel_Gas_LHV',
]

u_cols = [
    'Turbine_SHAFT_SPEED',           # Control inputs
    '14PDCV-504_Position',
    'SEAL_GAS_FLOW',
]

theta_cols = [
    f'PCA_Coefficient_{i}'
    for i in range(1, 13)            # 12 PCA components
]  # Written by preprocessing.py`,
    Visual: AnimatedColumnExtraction,
  },
  {
    id: 'tensors',
    title: '2. Tensor Assembly',
    icon: Database,
    description:
      'During __init__, the full parquet split is loaded into pandas then immediately converted to float32 PyTorch tensors in unified RAM. The condition tensor is built once by concatenating x and u along the feature axis — giving a permanent [N, 6] tensor for the lifetime of the dataset object.',
    pytorch:
      'Why concatenate x and u at dataset time rather than inside the model? It keeps the model completely agnostic to which sensors are "state" vs "control" — the model sees only one condition vector. Adding a 4th control input means changing dataset.py alone, with zero changes to the model architecture.',
    code: `class SCADAPipelineDataset(Dataset):
    def __init__(self, data_path, split):
        path = os.path.join(data_path, f"{split}.parquet")
        df   = pd.read_parquet(path)

        # Convert each group to float32 tensors
        x_t = torch.tensor(
            df[x_cols].values, dtype=torch.float32)   # [N, 3]
        u_t = torch.tensor(
            df[u_cols].values, dtype=torch.float32)   # [N, 3]

        self.theta_tensor = torch.tensor(
            df[theta_cols].values, dtype=torch.float32)  # [N, 12]

        # Fuse state + control → single condition vector
        self.condition_tensor = torch.cat(
            [x_t, u_t], dim=-1                        # [N, 6]
        )`,
    Visual: AnimatedTensorAssembly,
  },
  {
    id: 'memory',
    title: '3. In-Memory Layout',
    icon: Cpu,
    description:
      'Three separate SCADAPipelineDataset instances are constructed — one per chronological split. Each holds its two tensors in RAM for the full training run. The DataLoader wraps each with batch_size=1024 and num_workers=8 for fast parallel prefetching on the M4 Max.',
    pytorch:
      "Why load everything into RAM upfront? The M4 Max has 128 GB of unified memory shared between the CPU and the MPS GPU. Loading all windows into float32 tensors at startup means every __getitem__ call is a microsecond tensor slice — no disk I/O, no parquet parsing, and no CPU→GPU copy overhead (MPS unified memory eliminates that cost entirely).",
    code: `# In train.py — three dataset instances

train_dataset = SCADAPipelineDataset(DATA_PATH, "train")
val_dataset   = SCADAPipelineDataset(DATA_PATH, "val")
test_dataset  = SCADAPipelineDataset(DATA_PATH, "test")

# Memory per split (float32 = 4 bytes):
# theta_tensor:     N × 12 × 4 = N × 48  bytes
# condition_tensor: N ×  6 × 4 = N × 24  bytes
# Total per split:  N × 72 bytes
# 128 GB M4 Max → trivially holds all three

dataloader_kwargs = dict(
    batch_size  = BATCH_SIZE,   # 1024
    num_workers = 8,            # parallel prefetch workers
    pin_memory  = False,        # not needed: MPS unified memory
    drop_last   = True,
)`,
    Visual: AnimatedMemoryLayout,
  },
  {
    id: 'getitem',
    title: '4. Animated Walkthrough',
    icon: Activity,
    description:
      "The full dataset.py class annotated line by line — from __init__ reading the parquet, building tensors and fusing the condition vector, through __len__ and __getitem__ which returns the per-sample dict that DataLoader auto-collates into model-ready batches.",
    pytorch:
      "Why return a dict instead of a tuple? DataLoader auto-collates dicts of tensors into batched dicts — batch[\"theta\"] and batch[\"condition\"]. This makes trainer.py code self-documenting: you always know which tensor you are operating on by name, not by positional index. It also makes it trivial to add a third output (e.g. a timestamp) without breaking any existing code.",
    code: `    def __len__(self):
        return len(self.theta_tensor)

    def __getitem__(self, idx):
        # Called once per sample by DataLoader.
        # DataLoader collates 1024 calls into:
        #   batch["theta"]     → [1024, 12]
        #   batch["condition"] → [1024,  6]
        return {
            "theta":     self.theta_tensor[idx],
            "condition": self.condition_tensor[idx],
        }

# In trainer.py the batch is consumed as:
for batch in train_dataloader:
    theta     = batch["theta"].to(device)      # [B, 12]
    condition = batch["condition"].to(device)  # [B,  6]
    loss = model.compute_loss(theta, condition)`,
    Visual: AnimatedWalkthrough,
  },
];

// ─── Main Component ───────────────────────────────────────────────────────────

export default function DatasetSection() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">

      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
        <div className="p-3 bg-blue-100 text-blue-700 rounded-xl shadow-sm border border-blue-200">
          <Database className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800 tracking-tight">Dataset Interface</h2>
          <p className="text-sm text-slate-500 font-medium">
            dataset.py — Bridges processed parquet splits to PyTorch DataLoaders
          </p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full flex gap-1.5 mb-6">
        {steps.map((s, idx) => (
          <div
            key={s.id}
            onClick={() => setCurrentStep(idx)}
            title={s.title}
            className={`h-2.5 flex-1 rounded-full cursor-pointer transition-all duration-300 ${
              idx === currentStep
                ? 'bg-blue-500 scale-y-110 shadow-sm'
                : idx < currentStep
                ? 'bg-blue-300'
                : 'bg-slate-100 hover:bg-slate-200'
            }`}
          />
        ))}
      </div>

      {/* Main 2-column layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[500px]">

        {/* Left: Visual + Description */}
        <div className="lg:col-span-7 flex flex-col gap-6 min-w-0">
          <div className="w-full h-[380px] rounded-2xl shadow-xl overflow-hidden border-4 border-slate-900/5 bg-[#0d1117]">
            <step.Visual />
          </div>
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex-1 flex flex-col">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2.5 bg-blue-100 rounded-xl text-blue-600 shadow-sm border border-blue-200">
                <step.icon size={20} />
              </div>
              <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
            </div>
            <p className="text-slate-600 leading-relaxed text-[15px] mb-5">{step.description}</p>
            <div className="bg-amber-50/80 border border-amber-200 p-5 rounded-xl shadow-sm relative mt-auto">
              <div className="absolute top-0 left-0 w-1.5 h-full bg-amber-400 rounded-l-xl" />
              <h4 className="text-sm font-bold text-amber-900 mb-2 flex items-center gap-2 uppercase tracking-wide">
                <Activity size={16} className="text-amber-600" /> PyTorch Design Logic
              </h4>
              <p className="text-sm text-amber-800/90 leading-relaxed">{step.pytorch}</p>
            </div>
          </div>
        </div>

        {/* Right: Code panel */}
        <div className="lg:col-span-5 flex flex-col min-w-0 h-full">
          <div className="flex-1 bg-[#0f172a] rounded-2xl shadow-xl flex flex-col overflow-hidden border border-slate-700">
            <div className="bg-slate-800 px-4 py-3 border-b border-slate-700 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Terminal size={14} className="text-blue-400" />
                <span className="text-xs font-bold font-mono text-slate-300">dataset.py</span>
              </div>
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-rose-500/80" />
                <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
              </div>
            </div>
            <div className="p-5 overflow-auto flex-1 text-[13px] font-mono leading-relaxed text-slate-300 dataset-scroll">
              <style dangerouslySetInnerHTML={{__html:`
                .dataset-scroll::-webkit-scrollbar{width:6px;height:6px;}
                .dataset-scroll::-webkit-scrollbar-track{background:transparent;}
                .dataset-scroll::-webkit-scrollbar-thumb{background:#334155;border-radius:4px;}
              `}}/>
              <pre
                className="whitespace-pre-wrap text-[12px] leading-relaxed"
                dangerouslySetInnerHTML={{ __html: highlightCode(step.code) }}
              />
            </div>
          </div>
        </div>

      </div>

      {/* Bottom Nav */}
      <div className="flex items-center justify-between mt-8 pt-6 border-t border-slate-200">
        <button
          onClick={() => setCurrentStep(p => Math.max(0, p - 1))}
          disabled={currentStep === 0}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
        >
          <ChevronLeft size={18} /> Previous
        </button>
        <span className="text-xs text-slate-400 font-mono">{currentStep + 1} / {steps.length}</span>
        <button
          onClick={() => setCurrentStep(p => Math.min(steps.length - 1, p + 1))}
          disabled={currentStep === steps.length - 1}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-white bg-blue-500 hover:bg-blue-600 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:translate-y-0"
        >
          Next <ChevronRight size={18} />
        </button>
      </div>

    </div>
  );
}