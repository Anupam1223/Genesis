import React, { useState, useEffect, useRef } from 'react';
import {
  ChevronRight,
  ChevronLeft,
  Play,
  Pause,
  Database,
  Cpu,
  Layers,
  TrendingDown,
  Save,
  Zap,
  ArrowRight,
  ArrowDown,
  Box,
  Activity,
  Terminal,
  GitMerge,
  CheckCircle,
  RefreshCw,
  BarChart2,
  Shield,
} from 'lucide-react';

// ─── Tiny helpers ─────────────────────────────────────────────────────────────

const Tag = ({ children, color = 'slate' }) => {
  const map = {
    cyan:    'bg-cyan-900/40 text-cyan-300 border-cyan-500/50',
    violet:  'bg-violet-900/40 text-violet-300 border-violet-500/50',
    amber:   'bg-amber-900/40 text-amber-300 border-amber-500/50',
    emerald: 'bg-emerald-900/40 text-emerald-300 border-emerald-500/50',
    rose:    'bg-rose-900/40 text-rose-300 border-rose-500/50',
    blue:    'bg-blue-900/40 text-blue-300 border-blue-500/50',
    indigo:  'bg-indigo-900/40 text-indigo-300 border-indigo-500/50',
    slate:   'bg-slate-800 text-slate-300 border-slate-600',
    sky:     'bg-sky-900/40 text-sky-300 border-sky-500/50',
    fuchsia: 'bg-fuchsia-900/40 text-fuchsia-300 border-fuchsia-500/50',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[9px] font-bold font-mono border ${map[color]}`}>
      {children}
    </span>
  );
};

const VisualButton = ({ onClick, disabled, children, active }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-bold transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed z-10 ${
      active
        ? 'bg-cyan-600 hover:bg-cyan-500 text-white'
        : 'bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600'
    }`}
  >
    {children}
  </button>
);

// ─── Step 1 · Full Pipeline Map ───────────────────────────────────────────────

const pipelineNodes = [
  {
    id: 'raw',
    file: 'DataAllParts.xlsx / .parquet',
    role: 'Raw Input',
    color: 'slate',
    icon: <Database size={18} />,
    outputs: ['14 raw SCADA sensor columns', '~months of 1-second readings'],
    accent: '#64748b',
  },
  {
    id: 'preprocessing',
    file: 'preprocessing.py',
    role: 'Data Preparation',
    color: 'cyan',
    icon: <Activity size={18} />,
    outputs: ['Sliding windows 14,400 steps', 'RobustScaler on theta', 'StandardScaler on X & U', 'PCA → 12 coefficients', '→ train / val / test .parquet'],
    accent: '#06b6d4',
  },
  {
    id: 'dataset',
    file: 'dataset.py',
    role: 'PyTorch Interface',
    color: 'blue',
    icon: <Layers size={18} />,
    outputs: ['theta tensor [12]', 'condition tensor [6]', 'SCADAPipelineDataset'],
    accent: '#3b82f6',
  },
  {
    id: 'components',
    file: 'components.py',
    role: 'Model Building Blocks',
    color: 'violet',
    icon: <Box size={18} />,
    outputs: ['ResidualMLP (GELU + residuals)', 'rational_quadratic_spline()', 'NeuralSplineCouplingLayer'],
    accent: '#8b5cf6',
  },
  {
    id: 'flow_model',
    file: 'flow_model.py',
    role: 'Normalizing Flow',
    color: 'fuchsia',
    icon: <Zap size={18} />,
    outputs: ['PipelineConditionalFlow', '6 stacked coupling layers', 'compute_loss() → NLL', 'sample() → generate futures'],
    accent: '#d946ef',
  },
  {
    id: 'train',
    file: 'train.py',
    role: 'Assembly & Launch',
    color: 'amber',
    icon: <GitMerge size={18} />,
    outputs: ['Connects dataset → model → trainer', 'DataLoader (8 workers)', 'Dynamic dim_theta / dim_condition'],
    accent: '#f59e0b',
  },
  {
    id: 'trainer',
    file: 'trainer.py',
    role: 'Training Engine',
    color: 'rose',
    icon: <TrendingDown size={18} />,
    outputs: ['50 epochs × all batches', 'AdamW + ReduceLROnPlateau', 'Gradient clip 1.0', 'W&B logging every 50 steps'],
    accent: '#f43f5e',
  },
  {
    id: 'output',
    file: 'model_best.pt',
    role: 'Final Artefact',
    color: 'emerald',
    icon: <Save size={18} />,
    outputs: ['Trained spline flow weights', 'Can generate SCADA futures', 'Conditioned on sensor state'],
    accent: '#10b981',
  },
];

const colorMap = {
  slate:   { bg: 'bg-slate-800',      border: 'border-slate-600',      text: 'text-slate-300',   dot: 'bg-slate-400'   },
  cyan:    { bg: 'bg-cyan-900/30',    border: 'border-cyan-500/60',    text: 'text-cyan-300',    dot: 'bg-cyan-400'    },
  blue:    { bg: 'bg-blue-900/30',    border: 'border-blue-500/60',    text: 'text-blue-300',    dot: 'bg-blue-400'    },
  violet:  { bg: 'bg-violet-900/30',  border: 'border-violet-500/60',  text: 'text-violet-300',  dot: 'bg-violet-400'  },
  fuchsia: { bg: 'bg-fuchsia-900/30', border: 'border-fuchsia-500/60', text: 'text-fuchsia-300', dot: 'bg-fuchsia-400' },
  amber:   { bg: 'bg-amber-900/30',   border: 'border-amber-500/60',   text: 'text-amber-300',   dot: 'bg-amber-400'   },
  rose:    { bg: 'bg-rose-900/30',    border: 'border-rose-500/60',    text: 'text-rose-300',    dot: 'bg-rose-400'    },
  emerald: { bg: 'bg-emerald-900/30', border: 'border-emerald-500/60', text: 'text-emerald-300', dot: 'bg-emerald-400' },
};

const AnimatedPipelineMap = () => {
  const [active, setActive] = useState(null);
  const [flowing, setFlowing] = useState(false);
  const [flowIdx, setFlowIdx] = useState(-1);

  useEffect(() => {
    let int;
    if (flowing) {
      int = setInterval(() => {
        setFlowIdx(i => {
          if (i >= pipelineNodes.length - 1) { setFlowing(false); return i; }
          return i + 1;
        });
      }, 600);
    }
    return () => clearInterval(int);
  }, [flowing]);

  const handleFlow = () => {
    setFlowIdx(0);
    setActive(null);
    setFlowing(true);
  };

  const activeNode = active != null ? pipelineNodes[active] : flowIdx >= 0 ? pipelineNodes[flowIdx] : null;

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">Complete ML Pipeline — Click Any Node</div>

        {/* Pipeline chain */}
        <div className="w-full max-w-2xl mx-auto">
          {/* Row 1: raw → preprocessing → dataset */}
          <div className="flex items-center gap-1 justify-center flex-wrap mb-2">
            {pipelineNodes.slice(0, 3).map((node, i) => {
              const c = colorMap[node.color];
              const isActive = active === i || flowIdx === i;
              return (
                <React.Fragment key={node.id}>
                  <button
                    onClick={() => setActive(active === i ? null : i)}
                    className={`flex flex-col items-center gap-1 px-3 py-2.5 rounded-xl border-2 transition-all duration-300 cursor-pointer min-w-[90px] ${
                      isActive ? `${c.bg} ${c.border} scale-105 shadow-lg` : 'bg-slate-800 border-slate-700 hover:border-slate-500'
                    }`}
                  >
                    <span className={`transition-colors ${isActive ? c.text : 'text-slate-500'}`}>{node.icon}</span>
                    <span className={`text-[9px] font-bold font-mono text-center leading-tight ${isActive ? c.text : 'text-slate-400'}`}>{node.file}</span>
                    <span className="text-[8px] text-slate-500">{node.role}</span>
                  </button>
                  {i < 2 && <ArrowRight size={14} className={`flex-shrink-0 transition-colors duration-300 ${flowIdx > i ? 'text-cyan-400' : 'text-slate-700'}`} />}
                </React.Fragment>
              );
            })}
          </div>

          {/* Down arrow */}
          <div className="flex justify-center mb-2">
            <ArrowDown size={14} className={`transition-colors duration-300 ${flowIdx >= 3 ? 'text-cyan-400' : 'text-slate-700'}`} />
          </div>

          {/* Row 2: components → flow_model */}
          <div className="flex items-center gap-1 justify-center flex-wrap mb-2">
            {pipelineNodes.slice(3, 5).map((node, ii) => {
              const i = ii + 3;
              const c = colorMap[node.color];
              const isActive = active === i || flowIdx === i;
              return (
                <React.Fragment key={node.id}>
                  <button
                    onClick={() => setActive(active === i ? null : i)}
                    className={`flex flex-col items-center gap-1 px-3 py-2.5 rounded-xl border-2 transition-all duration-300 cursor-pointer min-w-[100px] ${
                      isActive ? `${c.bg} ${c.border} scale-105 shadow-lg` : 'bg-slate-800 border-slate-700 hover:border-slate-500'
                    }`}
                  >
                    <span className={`transition-colors ${isActive ? c.text : 'text-slate-500'}`}>{node.icon}</span>
                    <span className={`text-[9px] font-bold font-mono text-center leading-tight ${isActive ? c.text : 'text-slate-400'}`}>{node.file}</span>
                    <span className="text-[8px] text-slate-500">{node.role}</span>
                  </button>
                  {ii < 1 && <ArrowRight size={14} className={`flex-shrink-0 transition-colors duration-300 ${flowIdx >= 4 ? 'text-cyan-400' : 'text-slate-700'}`} />}
                </React.Fragment>
              );
            })}
          </div>

          {/* Down arrow */}
          <div className="flex justify-center mb-2">
            <ArrowDown size={14} className={`transition-colors duration-300 ${flowIdx >= 5 ? 'text-cyan-400' : 'text-slate-700'}`} />
          </div>

          {/* Row 3: train → trainer → output */}
          <div className="flex items-center gap-1 justify-center flex-wrap">
            {pipelineNodes.slice(5).map((node, ii) => {
              const i = ii + 5;
              const c = colorMap[node.color];
              const isActive = active === i || flowIdx === i;
              return (
                <React.Fragment key={node.id}>
                  <button
                    onClick={() => setActive(active === i ? null : i)}
                    className={`flex flex-col items-center gap-1 px-3 py-2.5 rounded-xl border-2 transition-all duration-300 cursor-pointer min-w-[90px] ${
                      isActive ? `${c.bg} ${c.border} scale-105 shadow-lg` : 'bg-slate-800 border-slate-700 hover:border-slate-500'
                    }`}
                  >
                    <span className={`transition-colors ${isActive ? c.text : 'text-slate-500'}`}>{node.icon}</span>
                    <span className={`text-[9px] font-bold font-mono text-center leading-tight ${isActive ? c.text : 'text-slate-400'}`}>{node.file}</span>
                    <span className="text-[8px] text-slate-500">{node.role}</span>
                  </button>
                  {ii < 2 && <ArrowRight size={14} className={`flex-shrink-0 transition-colors duration-300 ${flowIdx > i ? 'text-cyan-400' : 'text-slate-700'}`} />}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Detail panel */}
        {activeNode && (
          <div
            key={activeNode.id}
            className={`w-full max-w-2xl mx-auto rounded-xl border-2 p-4 transition-all duration-300 animate-in fade-in slide-in-from-bottom-2 ${colorMap[activeNode.color].bg} ${colorMap[activeNode.color].border}`}
          >
            <div className="flex items-center gap-2 mb-3">
              <span className={colorMap[activeNode.color].text}>{activeNode.icon}</span>
              <span className={`font-mono font-bold text-sm ${colorMap[activeNode.color].text}`}>{activeNode.file}</span>
              <span className="text-[9px] text-slate-500 ml-auto uppercase tracking-wider">{activeNode.role}</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {activeNode.outputs.map((o, i) => (
                <span key={i} className={`text-[10px] px-2 py-0.5 rounded-lg border ${colorMap[activeNode.color].bg} ${colorMap[activeNode.color].border} ${colorMap[activeNode.color].text}`}>
                  {o}
                </span>
              ))}
            </div>
          </div>
        )}

        {!activeNode && (
          <p className="text-center text-[11px] text-slate-400 max-w-lg mx-auto">
            Click any node to see what it produces. Press "Animate Flow" to watch data travel through the pipeline step by step.
          </p>
        )}
      </div>

      <div className="flex-shrink-0 flex justify-center gap-3 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={handleFlow} active={flowing} disabled={flowing}>
          {flowing ? <><Pause size={14} /> Flowing…</> : <><Play size={14} /> Animate Flow</>}
        </VisualButton>
        <VisualButton onClick={() => { setActive(null); setFlowIdx(-1); setFlowing(false); }}>
          <RefreshCw size={14} /> Reset
        </VisualButton>
      </div>
    </div>
  );
};

// ─── Step 2 · Preprocessing Deep-Dive ────────────────────────────────────────

const AnimatedPreprocessing = () => {
  const [phase, setPhase] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX = 5;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setPhase(p => { if (p >= MAX) { setIsPlaying(false); return p; } return p + 1; }), 1800);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  const stages = [
    {
      icon: '📦',
      label: 'Load Raw Data',
      color: 'slate',
      shape: (
        <div className="flex flex-col gap-1 w-full">
          <div className="text-[9px] text-slate-500 mb-1">DataAllParts.parquet</div>
          {['COMP_Suction_Pressure', 'COMP_Suction_Drum_Temperature', 'KPI_Fuel_Gas_LHV', 'Turbine_SHAFT_SPEED', 'COMP_Discharge_Pressure', '...8 theta cols...'].map((c, i) => (
            <div key={i} className="w-full h-3 bg-slate-700 rounded text-[8px] text-slate-400 flex items-center px-2">{c}</div>
          ))}
          <div className="text-[9px] text-slate-500 mt-1">→ millions of 1-second rows</div>
        </div>
      ),
      desc: 'Raw SCADA data loaded from DataAllParts.parquet — 14 sensor columns, millions of 1-second timestep rows. NaNs, Infs, and flatlined sensors are cleaned and micro-noise is injected to protect the spline math engine.',
    },
    {
      icon: '⚖️',
      label: 'Scale X & U (StandardScaler)',
      color: 'cyan',
      shape: (
        <div className="flex gap-3 w-full">
          <div className="flex-1 flex flex-col gap-1">
            <div className="text-[9px] text-cyan-400 font-bold mb-1">X Conditions (3)</div>
            {['Suction_Pressure', 'Suction_Temp', 'Fuel_Gas_LHV'].map((c, i) => (
              <div key={i} className="h-3 bg-cyan-900/40 border border-cyan-500/40 rounded text-[7px] text-cyan-300 flex items-center px-1">{c}</div>
            ))}
          </div>
          <div className="flex-1 flex flex-col gap-1">
            <div className="text-[9px] text-sky-400 font-bold mb-1">U Controls (3)</div>
            {['SHAFT_SPEED', '14PDCV-504', 'SEAL_GAS'].map((c, i) => (
              <div key={i} className="h-3 bg-sky-900/40 border border-sky-500/40 rounded text-[7px] text-sky-300 flex items-center px-1">{c}</div>
            ))}
          </div>
          <div className="flex items-center text-slate-500 font-bold">→ μ=0, σ=1</div>
        </div>
      ),
      desc: 'Condition variables (x_cols + u_cols) are StandardScaler-normalized to mean=0, std=1. The scaler is fitted on training data only — val and test rows are transformed but never seen during fit, preventing data leakage.',
    },
    {
      icon: '🦾',
      label: 'Scale θ (RobustScaler)',
      color: 'violet',
      shape: (
        <div className="flex flex-col gap-1 w-full">
          <div className="text-[9px] text-violet-400 font-bold mb-1">8 Theta Columns → RobustScaler</div>
          {['SEAL_GAS_FLTR_DP', 'LUBE_OIL_LVL', 'Turbine_Thermal_Efficiency', 'Gas_COMP_Efficiency', 'Discharge_Pressure', 'Discharge_Temp', 'Exhaust_Spread', 'Turbine_Heat_Rate'].map((c, i) => (
            <div key={i} className="h-3 bg-violet-900/40 border border-violet-500/40 rounded text-[7px] text-violet-300 flex items-center px-1">{c}</div>
          ))}
          <div className="text-[9px] text-slate-500 mt-1">Uses IQR (median/quartile) — robust to SCADA outlier spikes</div>
        </div>
      ),
      desc: 'Theta variables (the outputs to be modelled) use RobustScaler — it uses the interquartile range instead of mean/std. SCADA data has extreme outlier spikes from sensor faults; RobustScaler prevents those from corrupting the scaling. Output is clipped to [-20, 20].',
    },
    {
      icon: '🪟',
      label: 'Sliding Window → PCA',
      color: 'amber',
      shape: (
        <div className="flex flex-col gap-2 w-full">
          <div className="flex items-center gap-2">
            <div className="flex flex-col gap-0.5">
              {[0,1,2,3].map(i => (
                <div key={i} className="flex gap-0.5">
                  {Array.from({length: 8}).map((_, j) => (
                    <div key={j} className={`w-2 h-2 rounded-[2px] ${i===1&&j<6?'bg-amber-500':i===2&&j>=1&&j<7?'bg-amber-400/70':i===3&&j>=2?'bg-amber-300/50':'bg-slate-700'}`} />
                  ))}
                </div>
              ))}
            </div>
            <ArrowRight size={14} className="text-amber-400" />
            <div className="flex flex-col gap-1">
              <div className="text-[9px] text-amber-400 font-bold">14,400 steps/window</div>
              <div className="text-[9px] text-amber-300">↓ 60× downsample</div>
              <div className="text-[9px] text-amber-200">240 points × 8 cols</div>
              <div className="text-[9px] text-slate-400">= 1,920 features</div>
            </div>
            <ArrowRight size={14} className="text-amber-400" />
            <div className="flex flex-col items-center gap-1">
              <div className="text-[9px] text-emerald-400 font-bold">PCA</div>
              <div className="w-8 h-16 bg-gradient-to-b from-emerald-500 to-emerald-900/30 rounded border border-emerald-500/50 flex items-end justify-center pb-1">
                <span className="text-[7px] text-emerald-300 font-bold">12</span>
              </div>
              <div className="text-[7px] text-slate-500">&gt;90% var</div>
            </div>
          </div>
        </div>
      ),
      desc: 'A sliding window of 14,400 timesteps (4 hours) is carved from the time-series. Each window is downsampled 60× to 240 points, flattened to 1,920 features, and projected through PCA to just 12 coefficients — these are theta. PCA is fitted on training data only, using np.einsum to bypass Apple Silicon BLAS precision bugs.',
    },
    {
      icon: '✂️',
      label: 'Chronological Split',
      color: 'emerald',
      shape: (
        <div className="flex flex-col gap-2 w-full">
          <div className="w-full h-5 flex rounded-lg overflow-hidden gap-0.5 border border-slate-600">
            <div className="flex-[7] bg-blue-600 flex items-center justify-center text-[8px] font-bold text-white">train 70%</div>
            <div className="flex-[15] bg-amber-500 flex items-center justify-center text-[8px] font-bold text-white">val 15%</div>
            <div className="flex-[15] bg-rose-500 flex items-center justify-center text-[8px] font-bold text-white">test 15%</div>
          </div>
          {['train.parquet', 'val.parquet', 'test.parquet', 'DataAllParts_PCA.parquet'].map((f, i) => (
            <div key={i} className={`flex items-center gap-2 px-2 py-1 rounded border ${i<3?'bg-emerald-900/20 border-emerald-500/40':'bg-slate-800 border-slate-600'}`}>
              <Save size={10} className={i<3?'text-emerald-400':'text-slate-500'} />
              <span className={`font-mono text-[9px] font-bold ${i<3?'text-emerald-300':'text-slate-500'}`}>data/processed/{f}</span>
            </div>
          ))}
        </div>
      ),
      desc: 'The data is split chronologically — never randomly — to respect time-series causality. train=70%, val=15%, test=15%. Each split is saved as its own .parquet so dataset.py never has to re-split anything. Scalers and the PCA model are saved as .pkl files for future inference runs.',
    },
    {
      icon: '✅',
      label: 'Artefacts Saved',
      color: 'emerald',
      shape: (
        <div className="flex flex-col gap-1.5 w-full">
          <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">outputs/checkpoints/</div>
          {[
            { f: 'x_scaler.pkl',               note: 'StandardScaler for X cols' },
            { f: 'u_scaler.pkl',               note: 'StandardScaler for U cols' },
            { f: 'theta_base_scaler.pkl',       note: 'RobustScaler for theta' },
            { f: 'trajectory_pca_model.pkl',    note: 'PCA (12 components)' },
          ].map(a => (
            <div key={a.f} className="flex items-center gap-2 px-2 py-1 rounded bg-emerald-900/20 border border-emerald-500/40">
              <CheckCircle size={10} className="text-emerald-400 flex-shrink-0" />
              <span className="font-mono text-[9px] text-emerald-300 font-bold flex-shrink-0">{a.f}</span>
              <span className="text-[8px] text-slate-500">{a.note}</span>
            </div>
          ))}
        </div>
      ),
      desc: 'Preprocessing saves 4 sklearn artefacts: the two StandardScalers for sensor conditions, the RobustScaler for theta, and the PCA model. These are needed at inference time to preprocess new sensor readings before passing them to the trained flow model.',
    },
  ];

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">preprocessing.py — 6 Transformation Stages</div>

        {/* Stage tabs */}
        <div className="flex flex-wrap gap-1.5 justify-center w-full max-w-lg mx-auto">
          {stages.map((s, i) => (
            <button
              key={i}
              onClick={() => setPhase(i)}
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[9px] font-bold font-mono transition-all border ${
                phase === i
                  ? `bg-${s.color}-900/40 border-${s.color}-500/60 text-${s.color}-300`
                  : i < phase
                    ? 'bg-slate-700/50 border-slate-600 text-slate-400'
                    : 'bg-slate-800 border-slate-700 text-slate-500 hover:border-slate-600'
              }`}
            >
              {i < phase && <CheckCircle size={9} className="text-emerald-400" />}
              <span>{s.icon}</span>
              <span className="hidden sm:inline">{s.label}</span>
              <span className="sm:hidden">{i + 1}</span>
            </button>
          ))}
        </div>

        {/* Shape visual */}
        <div className="w-full max-w-lg mx-auto bg-slate-800 border border-slate-700 rounded-xl p-4 min-h-[120px]">
          <div key={phase} className="transition-opacity duration-200">
            {stages[phase].shape}
          </div>
        </div>

        {/* Description */}
        <div className="w-full max-w-lg mx-auto bg-slate-800/50 border border-slate-700 rounded-xl p-3 flex-shrink-0">
          <p key={`desc-${phase}`} className="text-[11px] text-slate-300 leading-relaxed">
            {stages[phase].desc}
          </p>
        </div>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-3 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setPhase(p => Math.max(0, p-1))} disabled={phase === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play'}
        </VisualButton>
        <VisualButton onClick={() => setPhase(p => Math.min(MAX, p+1))} active disabled={phase === MAX || isPlaying}>
          Next <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

// ─── Step 3 · Data → Tensors → Model ─────────────────────────────────────────

const AnimatedDataToModel = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX = 4;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(p => { if (p >= MAX) { setIsPlaying(false); return p; } return p + 1; }), 1800);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">dataset.py + components.py → How Data Becomes Model Input</div>

        <div className="w-full max-w-lg mx-auto flex flex-col gap-3">

          {/* Parquet files */}
          <div className="flex gap-2">
            {['train.parquet', 'val.parquet', 'test.parquet'].map((f, i) => (
              <div key={f} className={`flex-1 bg-blue-900/20 border border-blue-500/40 rounded-lg p-2 text-center transition-all duration-500 ${step >= 0 ? 'opacity-100' : 'opacity-0'}`} style={{transitionDelay:`${i*80}ms`}}>
                <Database size={12} className="text-blue-400 mx-auto mb-1" />
                <span className="font-mono text-[8px] font-bold text-blue-300">{f}</span>
              </div>
            ))}
          </div>

          {/* Column extraction */}
          <div className={`transition-all duration-500 ${step >= 1 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-3">
              <div className="text-[9px] text-slate-500 font-bold uppercase mb-2">SCADAPipelineDataset extracts columns</div>
              <div className="flex gap-2">
                <div className="flex-1">
                  <div className="text-[8px] text-cyan-400 font-bold mb-1">x_cols [3]</div>
                  {['Suction_Pressure','Suction_Temp','Fuel_Gas_LHV'].map(c => <div key={c} className="h-2.5 bg-cyan-900/30 border border-cyan-500/30 rounded text-[6px] text-cyan-400 px-1 flex items-center mb-0.5">{c}</div>)}
                </div>
                <div className="flex-1">
                  <div className="text-[8px] text-sky-400 font-bold mb-1">u_cols [3]</div>
                  {['SHAFT_SPEED','14PDCV-504','SEAL_GAS'].map(c => <div key={c} className="h-2.5 bg-sky-900/30 border border-sky-500/30 rounded text-[6px] text-sky-400 px-1 flex items-center mb-0.5">{c}</div>)}
                </div>
                <div className="flex-1">
                  <div className="text-[8px] text-violet-400 font-bold mb-1">theta_cols [12]</div>
                  {['PCA_Coefficient_1','PCA_Coefficient_2','...12 total...'].map(c => <div key={c} className="h-2.5 bg-violet-900/30 border border-violet-500/30 rounded text-[6px] text-violet-400 px-1 flex items-center mb-0.5">{c}</div>)}
                </div>
              </div>
            </div>
          </div>

          {/* Tensor concat */}
          <div className={`transition-all duration-500 ${step >= 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
            <div className="flex items-center gap-2 bg-slate-800 border border-slate-700 rounded-xl p-3">
              <div className="flex flex-col gap-1 flex-1">
                <div className="h-5 bg-cyan-900/30 border border-cyan-500/40 rounded text-[9px] text-cyan-300 flex items-center px-2 font-mono">x [3]</div>
                <div className="h-5 bg-sky-900/30 border border-sky-500/40 rounded text-[9px] text-sky-300 flex items-center px-2 font-mono">u [3]</div>
              </div>
              <div className="text-slate-500 font-bold text-lg">cat()</div>
              <ArrowRight size={14} className="text-slate-500" />
              <div className="flex-1 h-12 bg-indigo-900/30 border-2 border-indigo-500/60 rounded-xl flex items-center justify-center">
                <span className="font-mono text-[11px] font-bold text-indigo-300">condition [6]</span>
              </div>
            </div>
          </div>

          {/* __getitem__ output */}
          <div className={`transition-all duration-500 ${step >= 3 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
            <div className="bg-indigo-900/10 border-2 border-indigo-500/40 rounded-xl p-3">
              <div className="text-[9px] text-slate-500 font-bold mb-2">__getitem__(idx) returns:</div>
              <div className="flex gap-3">
                <div className="flex-1 bg-violet-900/20 border border-violet-500/40 rounded-lg p-2 text-center">
                  <div className="font-mono text-xs font-extrabold text-violet-300 mb-1">theta [12]</div>
                  <div className="text-[8px] text-slate-500">PCA coefficients<br/>of the future trajectory</div>
                </div>
                <div className="flex-1 bg-indigo-900/20 border border-indigo-500/40 rounded-lg p-2 text-center">
                  <div className="font-mono text-xs font-extrabold text-indigo-300 mb-1">condition [6]</div>
                  <div className="text-[8px] text-slate-500">Current sensor state<br/>x + u concatenated</div>
                </div>
              </div>
            </div>
          </div>

          {/* Into coupling layer */}
          <div className={`transition-all duration-500 ${step >= 4 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-violet-900/20 border border-violet-500/40 rounded-lg p-2 text-center">
                <span className="font-mono text-[10px] text-violet-300 font-bold">theta [12]</span>
              </div>
              <div className="flex-1 bg-indigo-900/20 border border-indigo-500/40 rounded-lg p-2 text-center">
                <span className="font-mono text-[10px] text-indigo-300 font-bold">condition [6]</span>
              </div>
              <ArrowRight size={14} className="text-cyan-400 flex-shrink-0" />
              <div className="flex-1 bg-fuchsia-900/20 border-2 border-fuchsia-500/60 rounded-lg p-2 text-center shadow-[0_0_12px_rgba(217,70,239,0.2)]">
                <Zap size={12} className="text-fuchsia-400 mx-auto mb-0.5" />
                <span className="font-mono text-[10px] text-fuchsia-300 font-bold">6× Coupling Layer</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-3 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setStep(p => Math.max(0, p-1))} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play'}
        </VisualButton>
        <VisualButton onClick={() => setStep(p => Math.min(MAX, p+1))} active disabled={step === MAX || isPlaying}>
          Next <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

// ─── Step 4 · The Training Loop (epoch view) ──────────────────────────────────

const AnimatedEpochLoop = () => {
  const [epoch, setEpoch] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_EPOCH = 50;

  const seed = (e) => Math.sin(e * 127.3) * 0.5 + 0.5;
  const trainLoss = (e) => +(2.6 * Math.exp(-0.06 * e) + 0.28 + seed(e) * 0.08).toFixed(3);
  const valLoss   = (e) => +(2.7 * Math.exp(-0.055 * e) + 0.32 + seed(e + 5) * 0.06).toFixed(3);

  const [history, setHistory] = useState([]);

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => {
        setEpoch(e => {
          if (e >= MAX_EPOCH) { setIsPlaying(false); return e; }
          const next = e + 1;
          setHistory(prev => [...prev, { e: next, tl: trainLoss(next), vl: valLoss(next) }]);
          return next;
        });
      }, 120);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  const handleReset = () => { setEpoch(0); setHistory([]); setIsPlaying(false); };

  const chartH = 70;
  const chartW = 300;
  const maxL = 3.0;

  const trainPts = history.map((h, i) => `${(i/(MAX_EPOCH-1))*chartW},${chartH-(h.tl/maxL)*chartH}`).join(' ');
  const valPts   = history.map((h, i) => `${(i/(MAX_EPOCH-1))*chartW},${chartH-(h.vl/maxL)*chartH}`).join(' ');

  const best = history.reduce((b, h) => (!b || h.vl < b.vl) ? h : b, null);

  const phaseLabel = epoch === 0 ? 'Waiting…'
    : epoch <= 10 ? 'Early training — loss drops fast'
    : epoch <= 30 ? 'Mid training — ReduceLROnPlateau may fire'
    : epoch < MAX_EPOCH ? 'Late training — fine-tuning the spline knots'
    : '✅ Training complete — model_best.pt saved!';

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">Full 50-Epoch Training Run — Simulated</div>

        {/* Stats */}
        <div className="flex gap-2 justify-center w-full max-w-md mx-auto">
          <div className="flex flex-col items-center bg-slate-800 border border-slate-700 rounded-xl px-3 py-2">
            <span className="text-[8px] text-slate-500 uppercase mb-0.5">Epoch</span>
            <span className="text-2xl font-extrabold text-white font-mono">{String(epoch).padStart(2,'0')}</span>
            <span className="text-[8px] text-slate-500">/ {MAX_EPOCH}</span>
          </div>
          <div className="flex flex-col items-center bg-slate-800 border border-slate-700 rounded-xl px-3 py-2">
            <span className="text-[8px] text-slate-500 uppercase mb-0.5">Train NLL</span>
            <span className="text-xl font-extrabold text-blue-400 font-mono">
              {history.length > 0 ? history[history.length-1].tl : '—'}
            </span>
          </div>
          <div className="flex flex-col items-center bg-slate-800 border border-slate-700 rounded-xl px-3 py-2">
            <span className="text-[8px] text-slate-500 uppercase mb-0.5">Val NLL</span>
            <span className="text-xl font-extrabold text-rose-400 font-mono">
              {history.length > 0 ? history[history.length-1].vl : '—'}
            </span>
          </div>
          {best && (
            <div className="flex flex-col items-center bg-emerald-900/20 border border-emerald-500/40 rounded-xl px-3 py-2">
              <span className="text-[8px] text-slate-500 uppercase mb-0.5">Best @ e{best.e}</span>
              <span className="text-xl font-extrabold text-emerald-400 font-mono">{best.vl}</span>
              <span className="text-[7px] text-emerald-600">model_best.pt</span>
            </div>
          )}
        </div>

        {/* Dual loss chart */}
        <div className="w-full max-w-md mx-auto bg-slate-800/60 border border-slate-700 rounded-xl p-3">
          <div className="flex gap-3 text-[9px] mb-2">
            <div className="flex items-center gap-1"><div className="w-3 h-1 bg-blue-400 rounded"></div><span className="text-slate-400">train_loss</span></div>
            <div className="flex items-center gap-1"><div className="w-3 h-1 bg-rose-400 rounded"></div><span className="text-slate-400">val_loss</span></div>
          </div>
          <div className="overflow-hidden rounded">
            <svg width="100%" viewBox={`-2 -4 ${chartW + 4} ${chartH + 6}`}>
              {[0.25,0.5,0.75,1].map(f => (
                <line key={f} x1="0" y1={chartH*(1-f)} x2={chartW} y2={chartH*(1-f)} stroke="#334155" strokeWidth="0.5" strokeDasharray="3" />
              ))}
              {history.length > 1 && <polyline points={trainPts} fill="none" stroke="#60a5fa" strokeWidth="1.5" />}
              {history.length > 1 && <polyline points={valPts} fill="none" stroke="#fb7185" strokeWidth="1.5" />}
              {best && history.length > 1 && (() => {
                const x = ((best.e - 1) / (MAX_EPOCH - 1)) * chartW;
                return <line x1={x} y1={0} x2={x} y2={chartH} stroke="#34d399" strokeWidth="1" strokeDasharray="3" />;
              })()}
            </svg>
          </div>
        </div>

        {/* Epoch cycle breakdown */}
        <div className="w-full max-w-md mx-auto grid grid-cols-4 gap-1.5">
          {[
            { label: 'Train Batches', color: 'blue',    icon: <Activity size={10}/>,   desc: 'forward→backward→clip→step' },
            { label: 'Validate',      color: 'sky',     icon: <Shield size={10}/>,      desc: 'no_grad forward only' },
            { label: 'LR Schedule',   color: 'amber',   icon: <TrendingDown size={10}/>,desc: 'halve if stuck 3 epochs' },
            { label: 'Checkpoint',    color: 'emerald', icon: <Save size={10}/>,        desc: 'save if new best or %10' },
          ].map(({ label, color, icon, desc }) => (
            <div key={label} className={`bg-${color}-900/20 border border-${color}-500/30 rounded-lg p-2 flex flex-col items-center gap-1`}>
              <span className={`text-${color}-400`}>{icon}</span>
              <span className={`text-[8px] font-bold text-${color}-300 text-center`}>{label}</span>
              <span className="text-[7px] text-slate-500 text-center leading-tight">{desc}</span>
            </div>
          ))}
        </div>

        <p className="text-center text-[11px] text-slate-400 px-2 max-w-md mx-auto font-mono">{phaseLabel}</p>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-3 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={handleReset} disabled={isPlaying}>
          <RefreshCw size={14} /> Reset
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying} disabled={epoch >= MAX_EPOCH}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Run 50 Epochs'}
        </VisualButton>
      </div>
    </div>
  );
};

// ─── Step 5 · The Final Output ────────────────────────────────────────────────

// ─── Step 4b · One Batch Inside the Flow ─────────────────────────────────────

const AnimatedBatchInsideFlow = () => {
  const [epoch, setEpoch] = useState(1);
  const [batch, setBatch] = useState(0);
  const [flowStep, setFlowStep] = useState(-1); // -1=idle, 0-7=active
  const [isPlaying, setIsPlaying] = useState(false);
  const [logdets, setLogdets] = useState([null, null, null, null]);
  const [loss, setLoss] = useState(null);
  const [done, setDone] = useState(false);

  const TOTAL_BATCHES = 6; // visual only
  const NUM_LAYERS = 4;

  const seed = (e, b) => Math.sin(e * 17.3 + b * 5.7) * 0.5 + 0.5;

  const reset = () => {
    setEpoch(1); setBatch(0); setFlowStep(-1);
    setLogdets([null, null, null, null]); setLoss(null);
    setDone(false); setIsPlaying(false);
  };

  useEffect(() => {
    if (!isPlaying) return;
    // flowStep -1 → 0..9, then advance batch/epoch
    const t = setTimeout(() => {
      setFlowStep(fs => {
        const next = fs + 1;
        // reveal logdets as layers complete (steps 1-4)
        if (next >= 1 && next <= NUM_LAYERS) {
          setLogdets(prev => {
            const arr = [...prev];
            arr[next - 1] = +(-0.3 - seed(epoch, batch) * 0.4 - next * 0.05).toFixed(3);
            return arr;
          });
        }
        // compute loss at step 5
        if (next === 5) {
          setLoss(+(1.8 - epoch * 0.025 - batch * 0.01 + seed(epoch, batch) * 0.12).toFixed(3));
        }
        // backward + step at 6,7
        if (next === 8) {
          // advance batch or epoch
          setBatch(b => {
            if (b + 1 >= TOTAL_BATCHES) {
              setEpoch(e => {
                if (e >= 4) { setDone(true); setIsPlaying(false); return e; }
                return e + 1;
              });
              return 0;
            }
            return b + 1;
          });
          setLogdets([null, null, null, null]);
          setLoss(null);
          return -1; // restart for next batch
        }
        return next;
      });
    }, flowStep === -1 ? 400 : 700);
    return () => clearTimeout(t);
  }, [isPlaying, flowStep, epoch, batch]);

  // Layer definitions
  const layers = [
    { label: 'Layer 1 MLP', a: 'θ₁–₆ + x', b: 'θ₇–₁₂', out: 'z₇–₁₂', color: 'sky' },
    { label: 'Layer 2 MLP', a: 'θ₇–₁₂ + x', b: 'θ₁–₆', out: 'z₁–₆', color: 'violet' },
    { label: 'Layer 3 MLP', a: 'z₁–₆ + x', b: 'z₇–₁₂', out: 'z\'₇–₁₂', color: 'fuchsia' },
    { label: 'Layer 4 MLP', a: 'z₇–₁₂ + x', b: 'z₁–₆', out: 'z_final', color: 'rose' },
  ];

  const colorMap = {
    sky: { bg: 'bg-sky-900/30', border: 'border-sky-500/60', text: 'text-sky-300', dot: 'bg-sky-400' },
    violet: { bg: 'bg-violet-900/30', border: 'border-violet-500/60', text: 'text-violet-300', dot: 'bg-violet-400' },
    fuchsia: { bg: 'bg-fuchsia-900/30', border: 'border-fuchsia-500/60', text: 'text-fuchsia-300', dot: 'bg-fuchsia-400' },
    rose: { bg: 'bg-rose-900/30', border: 'border-rose-500/60', text: 'text-rose-300', dot: 'bg-rose-400' },
  };

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">One Batch Inside the Flow — All 4 Layers, One Loss, One Backprop</div>

        {/* Epoch + Batch progress */}
        <div className="flex gap-2 justify-center">
          <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 flex items-center gap-2">
            <span className="text-[9px] text-slate-500 uppercase">Epoch</span>
            <div className="flex gap-1">
              {[1,2,3,4].map(e => (
                <div key={e} className={`w-5 h-5 rounded text-[8px] font-bold flex items-center justify-center border transition-all ${
                  e < epoch ? 'bg-cyan-600 border-cyan-500 text-white' :
                  e === epoch ? 'bg-cyan-900/60 border-cyan-400 text-cyan-300 ring-1 ring-cyan-400' :
                  'bg-slate-700 border-slate-600 text-slate-500'
                }`}>{e}</div>
              ))}
            </div>
          </div>
          <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 flex items-center gap-2">
            <span className="text-[9px] text-slate-500 uppercase">Batch</span>
            <div className="flex gap-0.5">
              {Array.from({length: TOTAL_BATCHES}).map((_, b) => (
                <div key={b} className={`w-4 h-4 rounded-sm text-[7px] font-bold flex items-center justify-center transition-all ${
                  b < batch ? 'bg-emerald-600 text-white' :
                  b === batch && flowStep >= 0 ? 'bg-amber-500/80 text-black ring-1 ring-amber-400 animate-pulse' :
                  'bg-slate-700 text-slate-500'
                }`}>{b+1}</div>
              ))}
              <span className="text-[8px] text-slate-500 ml-1 self-center">…×N</span>
            </div>
          </div>
        </div>

        {/* 4 coupling layers */}
        <div className="flex flex-col gap-1.5">
          {layers.map((layer, i) => {
            const active = flowStep === i + 1;
            const done_layer = flowStep > i + 1 || (flowStep === -1 && logdets[i] !== null);
            const c = colorMap[layer.color];
            return (
              <div key={i}>
                <div className={`rounded-lg border p-2 transition-all duration-500 ${
                  active ? `${c.bg} ${c.border} shadow-lg scale-[1.01]` :
                  done_layer ? 'bg-slate-800/40 border-slate-600' :
                  'bg-slate-800/20 border-slate-700/50 opacity-40'
                }`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-1.5 h-1.5 rounded-full ${active ? `${c.dot} animate-pulse` : done_layer ? 'bg-emerald-500' : 'bg-slate-600'}`}/>
                      <span className={`text-[10px] font-bold ${active ? c.text : done_layer ? 'text-slate-300' : 'text-slate-600'}`}>{layer.label}</span>
                    </div>
                    {logdets[i] !== null && (
                      <span className="font-mono text-[9px] text-amber-300 bg-amber-900/30 px-1.5 py-0.5 rounded border border-amber-700/50">
                        logdet = {logdets[i]}
                      </span>
                    )}
                  </div>
                  <div className={`flex items-center gap-1 mt-1 text-[9px] font-mono ${active ? 'text-slate-300' : 'text-slate-600'}`}>
                    <span className={active ? c.text : 'text-slate-600'}>[{layer.a}]</span>
                    <span>→ MLP → W,H,D → spline({layer.b})</span>
                    <span>→</span>
                    <span className={active ? 'text-emerald-300' : 'text-slate-600'}>{layer.out}</span>
                  </div>
                </div>
                {i < NUM_LAYERS - 1 && (
                  <div className={`flex items-center gap-1 justify-center py-0.5 transition-all ${flowStep > i + 1 || (flowStep === -1 && logdets[i] !== null) ? 'opacity-100' : 'opacity-20'}`}>
                    <div className="w-12 h-px bg-slate-600"/>
                    <span className="text-[7px] text-slate-500 font-mono">flip(z)</span>
                    <div className="w-12 h-px bg-slate-600"/>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Loss + backward */}
        <div className={`rounded-xl border p-3 transition-all duration-500 ${flowStep >= 5 ? 'bg-slate-800/60 border-slate-600 opacity-100' : 'opacity-20 border-slate-700/30'}`}>
          <div className="flex flex-col gap-1.5">
            {/* loss formula */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[9px] text-slate-400 font-mono">loss =</span>
              <span className="text-[9px] font-mono text-violet-300">-log_prob(z_final)</span>
              <span className="text-[9px] text-slate-500">−</span>
              <span className="text-[9px] font-mono text-amber-300">
                ({logdets.filter(v=>v!==null).join(' + ') || 'logdet₁+…+logdet₄'})
              </span>
              {loss !== null && (
                <span className="font-mono text-[11px] font-bold text-white ml-1">= {loss}</span>
              )}
            </div>
            {/* backward */}
            <div className={`transition-all duration-500 ${flowStep >= 6 ? 'opacity-100' : 'opacity-20'}`}>
              <div className="flex gap-2 mt-1">
                <div className={`flex-1 rounded-lg border px-2 py-1.5 text-center text-[9px] font-bold transition-all ${flowStep === 6 ? 'bg-rose-900/40 border-rose-500/60 text-rose-300 animate-pulse' : flowStep > 6 ? 'bg-rose-900/20 border-rose-700/40 text-rose-400' : 'border-slate-700 text-slate-600'}`}>
                  loss.backward()<br/><span className="font-normal text-[8px]">grads flow through all 4 MLPs simultaneously</span>
                </div>
                <div className={`flex-1 rounded-lg border px-2 py-1.5 text-center text-[9px] font-bold transition-all ${flowStep === 7 ? 'bg-emerald-900/40 border-emerald-500/60 text-emerald-300 animate-pulse' : flowStep > 7 || (flowStep === -1 && batch > 0) ? 'bg-emerald-900/20 border-emerald-700/40 text-emerald-400' : 'border-slate-700 text-slate-600'}`}>
                  optimizer.step()<br/><span className="font-normal text-[8px]">all 4 MLP weights update at once</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {done && (
          <div className="text-center text-[11px] text-emerald-400 font-mono bg-emerald-900/20 border border-emerald-700/40 rounded-lg py-2">
            ✅ All epochs done — model_best.pt saved to disk
          </div>
        )}

        {/* Key insight */}
        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-2 text-[9px] text-slate-400 leading-relaxed">
          <span className="text-white font-bold">Key: </span>
          The 4 MLPs are <strong className="text-cyan-300">never trained one at a time</strong>. Every batch runs all 4 forward in sequence, computes <strong className="text-amber-300">one shared loss</strong>, then backprop sends gradients <strong className="text-rose-300">backward through all 4 simultaneously</strong>. One batch = one update for every weight in the entire model.
        </div>
      </div>

      <div className="flex-shrink-0 flex flex-col gap-2 py-3 px-4 border-t border-slate-700/60 bg-slate-900">
        {/* Step labels */}
        <div className="flex justify-center">
          <div className="text-[9px] font-mono text-slate-400 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-center">
            {flowStep === -1 && '⏸ Ready — press Next Step or Run Batches'}
            {flowStep === 0  && '→ Batch loaded into GPU memory'}
            {flowStep === 1  && '🔵 Layer 1: θ₁–₆ + x → MLP → W,H,D → transform θ₇–₁₂ → logdet₁'}
            {flowStep === 2  && '🟣 Layer 2: flip → θ₇–₁₂ + x → MLP → W,H,D → transform θ₁–₆ → logdet₂'}
            {flowStep === 3  && '🟤 Layer 3: flip → MLP → logdet₃'}
            {flowStep === 4  && '🔴 Layer 4: flip → MLP → z_final + logdet₄'}
            {flowStep === 5  && '📊 Loss = -log_prob(z_final) − (logdet₁+logdet₂+logdet₃+logdet₄)'}
            {flowStep === 6  && '⬅ loss.backward() — gradients flow back through ALL 4 MLPs at once'}
            {flowStep === 7  && '✅ optimizer.step() — all 4 MLP weights updated simultaneously'}
          </div>
        </div>
        {/* Buttons */}
        <div className="flex justify-center gap-2">
          <VisualButton onClick={reset} disabled={isPlaying}>
            <RefreshCw size={14}/> Reset
          </VisualButton>
          <VisualButton
            onClick={() => {
              if (isPlaying) return;
              setFlowStep(fs => {
                const next = fs + 1;
                if (next >= 1 && next <= NUM_LAYERS) {
                  setLogdets(prev => {
                    const arr = [...prev];
                    arr[next - 1] = +(-0.3 - seed(epoch, batch) * 0.4 - next * 0.05).toFixed(3);
                    return arr;
                  });
                }
                if (next === 5) {
                  setLoss(+(1.8 - epoch * 0.025 - batch * 0.01 + seed(epoch, batch) * 0.12).toFixed(3));
                }
                if (next === 8) {
                  setBatch(b => {
                    if (b + 1 >= TOTAL_BATCHES) {
                      setEpoch(e => { if (e >= 4) { setDone(true); return e; } return e + 1; });
                      return 0;
                    }
                    return b + 1;
                  });
                  setLogdets([null, null, null, null]);
                  setLoss(null);
                  return -1;
                }
                return next;
              });
            }}
            disabled={isPlaying || done}
            active={false}
          >
            <ChevronRight size={14}/> Next Step
          </VisualButton>
          <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying} disabled={done}>
            {isPlaying ? <Pause size={14}/> : <Play size={14}/>} {isPlaying ? 'Pause' : 'Auto Play'}
          </VisualButton>
        </div>
      </div>
    </div>
  );
};

const AnimatedOutput = () => {
  const [mode, setMode] = useState(null); // 'forward' | 'inverse'
  const [generating, setGenerating] = useState(false);
  const [samples, setSamples] = useState([]);

  const handleGenerate = () => {
    setGenerating(true);
    setSamples([]);
    let count = 0;
    const int = setInterval(() => {
      setSamples(prev => [...prev, {
        id: count,
        theta: Array.from({length: 12}, () => +(Math.random() * 2 - 1).toFixed(3)),
      }]);
      count++;
      if (count >= 5) { clearInterval(int); setGenerating(false); }
    }, 400);
  };

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">What We Get — model_best.pt Capabilities</div>

        {/* The checkpoint file */}
        <div className="w-full max-w-lg mx-auto bg-emerald-900/20 border-2 border-emerald-500/50 rounded-xl p-4 flex items-start gap-3">
          <Save size={28} className="text-emerald-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-mono text-sm font-bold text-emerald-300 mb-1">outputs/checkpoints/model_best.pt</div>
            <div className="text-[10px] text-slate-400 leading-relaxed">
              A PyTorch state dict containing the trained weights of all <strong className="text-slate-300">6 NeuralSplineCouplingLayers</strong>, each with its <strong className="text-slate-300">ResidualMLP (GELU + residuals)</strong> and <strong className="text-slate-300">8-bin Rational-Quadratic spline</strong> parameters. Also includes the AdamW optimizer state for resuming.
            </div>
          </div>
        </div>

        {/* Two modes */}
        <div className="w-full max-w-lg mx-auto flex gap-3">
          <button
            onClick={() => setMode(mode === 'forward' ? null : 'forward')}
            className={`flex-1 rounded-xl border-2 p-3 transition-all cursor-pointer text-left ${mode === 'forward' ? 'bg-blue-900/30 border-blue-500/60' : 'bg-slate-800 border-slate-700 hover:border-slate-600'}`}
          >
            <div className={`flex items-center gap-2 mb-2 ${mode === 'forward' ? 'text-blue-300' : 'text-slate-400'}`}>
              <Activity size={14} />
              <span className="font-bold text-[11px]">compute_loss() — Training Mode</span>
            </div>
            {mode === 'forward' && (
              <div className="text-[9px] text-slate-400 leading-relaxed animate-in fade-in duration-200">
                <code className="text-blue-300">theta [12]</code> + <code className="text-indigo-300">condition [6]</code>
                <br/>→ Forward through 6 coupling layers (with flip)
                <br/>→ z_final lands on Standard Normal
                <br/>→ NLL loss = −(log_prob + log_det_jacobian)
                <br/><span className="text-slate-500">Used during training to update weights</span>
              </div>
            )}
          </button>

          <button
            onClick={() => setMode(mode === 'inverse' ? null : 'inverse')}
            className={`flex-1 rounded-xl border-2 p-3 transition-all cursor-pointer text-left ${mode === 'inverse' ? 'bg-fuchsia-900/30 border-fuchsia-500/60' : 'bg-slate-800 border-slate-700 hover:border-slate-600'}`}
          >
            <div className={`flex items-center gap-2 mb-2 ${mode === 'inverse' ? 'text-fuchsia-300' : 'text-slate-400'}`}>
              <Zap size={14} />
              <span className="font-bold text-[11px]">sample() — Inference Mode</span>
            </div>
            {mode === 'inverse' && (
              <div className="text-[9px] text-slate-400 leading-relaxed animate-in fade-in duration-200">
                <code className="text-fuchsia-300">condition [6]</code> (current sensor state)
                <br/>→ Sample z ~ N(0, I) (random noise)
                <br/>→ Backward through 6 layers (undo flips)
                <br/>→ Quadratic formula inverse algebra
                <br/>→ <code className="text-emerald-300">theta [12]</code> = simulated future trajectory
              </div>
            )}
          </button>
        </div>

        {/* Live generation demo */}
        <div className="w-full max-w-lg mx-auto bg-slate-800 border border-slate-700 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[10px] font-bold text-fuchsia-300 uppercase tracking-wider">Live Generation Demo</span>
            <Tag color="fuchsia">sample(condition=[0.2, -0.1, 0.5, 1.1, -0.3, 0.8])</Tag>
          </div>
          {samples.length > 0 && (
            <div className="flex flex-col gap-1.5 mb-3">
              {samples.map(s => (
                <div key={s.id} className="flex items-center gap-2 animate-in fade-in slide-in-from-left-2 duration-300">
                  <span className="text-[8px] text-slate-500 w-16 flex-shrink-0 font-mono">future #{s.id + 1}</span>
                  <div className="flex gap-0.5 flex-1">
                    {s.theta.map((v, i) => (
                      <div
                        key={i}
                        className="flex-1 rounded-sm"
                        style={{
                          height: `${Math.abs(v) * 14 + 6}px`,
                          background: `hsl(${280 + i * 8}, 70%, ${40 + Math.abs(v) * 20}%)`,
                          opacity: 0.8,
                        }}
                      />
                    ))}
                  </div>
                  <span className="text-[8px] text-fuchsia-400 font-mono w-8">✓</span>
                </div>
              ))}
            </div>
          )}
          {samples.length === 0 && !generating && (
            <p className="text-[10px] text-slate-500 text-center mb-3">Press Generate to simulate 5 possible SCADA futures</p>
          )}
          {generating && <p className="text-[10px] text-fuchsia-400 text-center mb-3 animate-pulse">Sampling from learned distribution…</p>}
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="w-full py-2 bg-fuchsia-600 hover:bg-fuchsia-500 disabled:opacity-50 text-white text-xs font-bold rounded-lg transition-all"
          >
            {generating ? 'Generating…' : '⚡ Generate 5 SCADA Futures'}
          </button>
        </div>

        <p className="text-center text-[11px] text-slate-400 px-2 max-w-lg mx-auto leading-relaxed">
          The trained model captures the <strong className="text-slate-300">conditional distribution</strong> of future compressor behaviour given current sensor readings. Each call to <code className="text-fuchsia-300 bg-slate-800 px-1 rounded">sample()</code> generates a statistically plausible 4-hour SCADA trajectory in milliseconds — ready for edge deployment.
        </p>
      </div>
    </div>
  );
};

// ─── Step 6 · Full Walkthrough (script-by-script summary) ────────────────────

const AnimatedScriptSummary = () => {
  const [active, setActive] = useState(0);

  const scripts = [
    {
      file: 'preprocessing.py',
      color: 'cyan',
      role: 'Run once, offline',
      icon: <Activity size={16} />,
      input: 'data/raw/DataAllParts.parquet',
      output: 'data/processed/{train,val,test}.parquet + 4 scalers/PCA .pkl',
      what: [
        'Cleans NaN, Inf, flatlined sensors',
        'StandardScaler on X & U (fit on train only)',
        'RobustScaler on theta (IQR-based, outlier-robust)',
        'Sliding window 14,400 steps → downsample 60× → flatten',
        'PCA: 1,920 features → 12 coefficients (>90% variance)',
        'Uses np.einsum to bypass Apple Silicon BLAS bug',
        'Chronological split: train 70% / val 15% / test 15%',
      ],
      keyDecision: 'Why PCA? Normalizing flows need a fixed-dimension latent target. Compressing 4-hour trajectories into 12 principal components gives the model a tractable, low-noise theta to learn.',
    },
    {
      file: 'dataset.py',
      color: 'blue',
      role: 'Called by train.py',
      icon: <Database size={16} />,
      input: 'data/processed/{split}.parquet',
      output: 'PyTorch Dataset yielding {theta[12], condition[6]}',
      what: [
        'Reads the pre-split parquet directly (no re-splitting)',
        'Extracts x_cols[3] + u_cols[3] + theta_cols[12]',
        'Converts to float32 tensors in-memory (128GB M4 Max)',
        'condition = cat(x_tensor, u_tensor) along dim=-1',
        '__getitem__ returns {theta, condition} dict',
      ],
      keyDecision: 'Why in-memory? The M4 Max has 128GB unified memory. Loading all parquet rows into RAM eliminates disk I/O from the training hot path — each batch access is a tensor slice, not a file read.',
    },
    {
      file: 'components.py',
      color: 'violet',
      role: 'Model building blocks',
      icon: <Box size={16} />,
      input: 'None (defines nn.Module classes)',
      output: 'ResidualMLP, rational_quadratic_spline(), NeuralSplineCouplingLayer',
      what: [
        'ResidualMLP: input→hidden (GELU)→N residual blocks→output',
        'Initialized to zeros so flow starts as identity',
        'RQ spline: softmax widths/heights, softplus derivatives',
        'Bounding box [-5, 5]: points outside pass through unchanged',
        'Forward: compute y + log|dy/dx| (log Jacobian)',
        'Inverse: quadratic formula trick for exact algebra reversal',
        'NeuralSplineCouplingLayer: split θ → brain(θ₁+c) → spline(θ₂)',
      ],
      keyDecision: 'Why zero-initialize the MLP output? At the start of training, every coupling layer is an exact identity function — no transformation at all. This prevents catastrophic NaN loss on epoch 1 when the spline knots are random.',
    },
    {
      file: 'flow_model.py',
      color: 'fuchsia',
      role: 'Orchestrates the full flow',
      icon: <Zap size={16} />,
      input: 'dim_theta, dim_condition, num_layers, hidden_dim, num_bins, bound',
      output: 'PipelineConditionalFlow with compute_loss() and sample()',
      what: [
        'Stacks num_layers NeuralSplineCouplingLayers in ModuleList',
        'Forward: θ → z through all layers with flip between each',
        'Flip alternates which half gets transformed each layer',
        'compute_loss: NLL = −(blueprint_log_prob + sum_log_det)',
        'Registers Standard Normal as buffer (moves with .to(device))',
        'sample(): z ~ N(0,I) → backward through layers → θ',
      ],
      keyDecision: 'Why flip between layers? Each coupling layer only transforms half the dimensions. By flipping the tensor, the next layer transforms the other half. After 6 layers, every dimension has been through the spline math multiple times.',
    },
    {
      file: 'train.py',
      color: 'amber',
      role: 'Assembly script',
      icon: <GitMerge size={16} />,
      input: 'data/processed/ (parquets)',
      output: 'Calls trainer.train() → writes model_best.pt',
      what: [
        'Sets all hyperparameters as uppercase constants (one place)',
        'MPS → CUDA → CPU device detection chain',
        'Creates 3 SCADAPipelineDatasets (train/val/test)',
        'Shared dataloader_kwargs (batch=1024, workers=8)',
        'Peeks train_dataset[0] → reads dim_theta & dim_condition',
        'Instantiates PipelineConditionalFlow dynamically',
        'Passes everything to SMPCTrainer → trainer.train()',
      ],
      keyDecision: 'Why dynamic sizing? dim_theta and dim_condition are measured from the live dataset at runtime. Change N_COMPONENTS in preprocessing.py from 12 to 16 and train.py re-builds the correct model automatically — no code change needed.',
    },
    {
      file: 'trainer.py',
      color: 'rose',
      role: 'Training engine',
      icon: <TrendingDown size={16} />,
      input: 'model, dataloaders, hyperparameters',
      output: 'outputs/checkpoints/model_best.pt + model_epoch_N.pt',
      what: [
        'AdamW lr=5e-4, weight_decay=1e-4 (low: protects spline knots)',
        'ReduceLROnPlateau: halve LR after 3 stalled epochs',
        'No float16 autocast — RQ division overflows in float16',
        'Gradient clip max_norm=1.0 (tighter than affine 5.0)',
        'W&B logging: batch every 50 steps, epoch every epoch',
        'Saves model_best.pt when val_loss improves',
        'Saves model_epoch_N.pt every 10 epochs as safety net',
      ],
      keyDecision: 'Why no float16? The rational-quadratic spline formula divides bin widths and heights — numbers near zero. In float16 (min ~6e-5), these divisions easily underflow to zero, making the denominator vanish and producing NaN loss. float32 is non-negotiable for spline flows.',
    },
  ];

  const s = scripts[active];

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">Script-by-Script Reference — Key Decisions Explained</div>

        {/* Script selector */}
        <div className="flex flex-wrap gap-1.5 justify-center w-full max-w-2xl mx-auto">
          {scripts.map((sc, i) => {
            const c = colorMap[sc.color];
            return (
              <button
                key={sc.file}
                onClick={() => setActive(i)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[9px] font-bold font-mono transition-all border ${
                  active === i ? `${c.bg} ${c.border} ${c.text}` : 'bg-slate-800 border-slate-700 text-slate-500 hover:border-slate-600'
                }`}
              >
                <span className={active === i ? c.text : 'text-slate-600'}>{sc.icon}</span>
                {sc.file}
              </button>
            );
          })}
        </div>

        {/* Detail */}
        <div
          key={active}
          className={`w-full max-w-2xl mx-auto rounded-xl border-2 p-4 flex flex-col gap-3 animate-in fade-in duration-200 ${colorMap[s.color].bg} ${colorMap[s.color].border}`}
        >
          {/* Header */}
          <div className="flex items-center gap-3 flex-wrap">
            <span className={colorMap[s.color].text}>{s.icon}</span>
            <span className={`font-mono font-bold text-sm ${colorMap[s.color].text}`}>{s.file}</span>
            <Tag color={s.color}>{s.role}</Tag>
          </div>

          {/* IO */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className="bg-slate-900/50 rounded-lg p-2">
              <div className="text-[8px] text-slate-500 uppercase mb-1">Input</div>
              <code className="text-[9px] text-slate-300">{s.input}</code>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-2">
              <div className="text-[8px] text-slate-500 uppercase mb-1">Output</div>
              <code className="text-[9px] text-slate-300">{s.output}</code>
            </div>
          </div>

          {/* What it does */}
          <div className="flex flex-col gap-1">
            {s.what.map((w, i) => (
              <div key={i} className="flex items-start gap-2">
                <div className={`w-1.5 h-1.5 rounded-full mt-1 flex-shrink-0 ${colorMap[s.color].dot}`}></div>
                <span className="text-[10px] text-slate-300 leading-relaxed">{w}</span>
              </div>
            ))}
          </div>

          {/* Key decision */}
          <div className="bg-amber-950/40 border border-amber-500/30 rounded-lg p-3">
            <div className="text-[8px] font-bold text-amber-400 uppercase tracking-wider mb-1 flex items-center gap-1">
              <Activity size={10} /> Key Design Decision
            </div>
            <p className="text-[10px] text-amber-200/80 leading-relaxed">{s.keyDecision}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── STEP DEFINITIONS ─────────────────────────────────────────────────────────

const steps = [
  {
    id: 'map',
    title: '1. Full Pipeline Map',
    icon: GitMerge,
    description: 'Eight interconnected files form a single end-to-end ML pipeline. Raw SCADA sensor recordings enter on one end — a trained normalizing flow capable of generating realistic compressor futures exits on the other.',
    why: 'Every file has exactly one responsibility. preprocessing.py shapes data. dataset.py interfaces it to PyTorch. components.py defines the math. flow_model.py stacks it. train.py assembles it. trainer.py runs it. This separation means changing one layer never requires touching another.',
    codeSnippet: `# The data flow in one picture:\n\nDataAllParts.xlsx\n  └─ preprocessing.py  ──→  train.parquet\n                              val.parquet\n                              test.parquet\n                              trajectory_pca_model.pkl\n                              pca_coeff_scaler.pkl\n\ntrain.parquet ──→ dataset.py  ──→ SCADAPipelineDataset\n                                    theta [12]\n                                    condition [3]  # x only; u bypasses flow\n\ncomponents.py  ──→ flow_model.py  ──→ PipelineConditionalFlow\n  ResidualMLP                           10 coupling layers\n  RQ Spline                             compute_loss()\n  CouplingLayer                         sample()\n\ntrain.py  ──→  trainer.py  ──→  model_best.pt`,
    Visual: AnimatedPipelineMap,
  },
  {
    id: 'preprocessing',
    title: '2. Preprocessing Pipeline',
    icon: Activity,
    description: 'preprocessing.py is the most complex script — it transforms raw 1-second SCADA sensor readings into model-ready PCA coefficient vectors. It runs once offline and takes ~9 minutes. Everything it produces is cached to disk.',
    why: 'The chronological split happens here, not in the DataLoader. This is critical: if you split randomly, future data leaks into the training scaler fit and you get an optimistically biased model. Splitting time-series data by time index is the only correct approach.',
    codeSnippet: `# Key numbers:\nWINDOW_SIZE    = 14400  # 4 hours of 1-second data\nDOWNSAMPLE_RATE = 60   # → 240 points per window\nN_COMPONENTS   = 12    # PCA components kept\n\n# Scaling pipeline (all fitted on TRAIN only):\n# 1. StandardScaler  → x and u columns\n# 2. RobustScaler    → raw theta SCADA signals\n# 3. PCA projection  → 12 coefficients per window\n# 4. StandardScaler  → PCA coefficients (so they\n#    fall within BOUND=5.0 for the spline flow)\n\n# Why np.einsum instead of pca.transform()?\n# Apple Silicon BLAS has a precision bug with\n# large float64 matrix multiplications →\n# einsum bypasses Accelerate entirely\n\n# Split indices (chronological):\ntrain_end = int(M_total * 0.70)\nval_end   = int(M_total * 0.85)\n# test = 0.85 → end`,
    Visual: AnimatedPreprocessing,
  },
  {
    id: 'data_to_model',
    title: '3. Data → Tensors → Model',
    icon: Database,
    description: 'dataset.py bridges the processed parquet files to PyTorch. Each row in a parquet becomes a {theta, condition} dict of float32 tensors. The DataLoader then batches 1,024 of these per training step, pre-fetched by 8 parallel workers.',
    why: 'theta and condition are intentionally kept as separate tensors rather than concatenated. The coupling layer needs to split theta in half internally, and the MLP brain receives condition separately as a concat with theta₁. Mixing them earlier would require un-mixing them inside the model.',
    codeSnippet: `# dataset.py __getitem__ output shape:\n# theta     → torch.float32 [12]  ← what to model\n# condition → torch.float32 [3]   ← x only (measured-now)\n\n# CAUSAL PARTITION:\n# condition = x  (Suction_P, Suction_T, Fuel_LHV)\n# u (Shaft_Speed, Valve_504, Seal_Gas) BYPASSES flow\n# u is used only at the MPC decision layer\n\n# DataLoader in train.py:\n# batch['theta']     shape: [4096, 12]\n# batch['condition'] shape: [4096,  3]`,
    Visual: AnimatedDataToModel,
  },
  {
    id: 'training',
    title: '4. 50-Epoch Training Loop',
    icon: TrendingDown,
    description: 'trainer.py runs the full training loop. For each of 50 epochs it iterates every batch in train_dataloader, runs the 6-stage gradient update, validates on val_dataloader, steps the LR scheduler, and decides whether to save a checkpoint.',
    why: 'The epoch loop is entirely inside SMPCTrainer, not in train.py. This clean separation means train.py stays a simple assembly script and trainer.py is a reusable engine. You can swap the model, change the loss, or add a second validation dataset by only editing trainer.py.',
    codeSnippet: `# Per-epoch sequence inside trainer.train():\n\nfor epoch in 1..50:\n  # 1. TRAIN — update weights on every batch\n  model.train()\n  for batch in train_dataloader:\n    zero_grad → loss → backward\n    → clip(0.5) → AdamW.step()\n    # clip tightened to 0.5 for larger model\n    # (512 dim, 10 layers accumulate larger grads)\n\n  # 2. VALIDATE — no weight updates\n  avg_val_loss = evaluate()  # torch.no_grad()\n\n  # 3. SCHEDULE — maybe halve LR\n  scheduler.step(avg_val_loss)\n\n  # 4. CHECKPOINT — maybe save to disk\n  if avg_val_loss < best_val_loss:\n      save model_best.pt\n  elif epoch % 10 == 0:\n      save model_epoch_N.pt`,
    Visual: AnimatedEpochLoop,
  },
  {
    id: 'batch_inside_flow',
    title: '4b. One Batch Inside the Flow',
    icon: GitMerge,
    description: 'Zoom into a single batch: all 4 coupling layers run forward in sequence, each appending its logdet to a running total. One shared loss is computed. One loss.backward() sends gradients back through all 4 MLPs simultaneously. One optimizer.step() updates every weight at once.',
    why: 'All 4 MLPs are never trained one at a time. They share a single loss signal and update together every batch. Without this joint training, Layer 1 would optimize for the wrong target — it has no idea what Layers 2-4 will do with its output unless they all participate in the same loss.',
    codeSnippet: `# One batch — the full loop inside trainer.py:\n\nfor batch in train_dataloader:         # e.g. 4096 samples\n    optimizer.zero_grad()\n\n    # ── FORWARD: all 4 layers in sequence ──\n    z, logdet_1 = layer1(theta, x)     # θ₁–₆+x → W,H,D → transform θ₇–₁₂\n    z = flip(z)\n    z, logdet_2 = layer2(z,     x)     # θ₇–₁₂+x → W,H,D → transform θ₁–₆\n    z = flip(z)\n    z, logdet_3 = layer3(z,     x)\n    z = flip(z)\n    z, logdet_4 = layer4(z,     x)     # → z_final\n\n    total_logdet = logdet_1 + logdet_2 + logdet_3 + logdet_4\n\n    # ── ONE SHARED LOSS ──\n    loss = -( log_prob(z_final) + total_logdet )\n\n    # ── BACKWARD: grads flow through ALL 4 at once ──\n    loss.backward()\n    clip_grad_norm_(model.parameters(), 1.0)\n    optimizer.step()                   # all 4 MLPs update simultaneously`,
    Visual: AnimatedBatchInsideFlow,
  },
  {
    id: 'output',
    title: '5. End Result',
    icon: Save,
    description: 'After training, model_best.pt contains the weights of a conditional normalizing flow that has learned the distribution of 4-hour compressor futures given current sensor readings. At inference time, sample() generates new plausible futures in milliseconds.',
    why: "This is the core value proposition of the whole pipeline: instead of asking 'what will happen next?' with a single deterministic prediction, the model answers 'here is the full distribution of what could happen' — 1,000 possible futures in under a second, each statistically consistent with current conditions.",
    codeSnippet: `# Using the trained model at inference time:\nmodel = PipelineConditionalFlow(...)\nmodel.load_state_dict(\n    torch.load('outputs/checkpoints/model_best.pt')\n    ['model_state_dict']\n)\nmodel.eval()\n\n# Current measured-now sensor state (x only)\ncondition = torch.tensor(\n    [0.2, -0.1, 0.5]  # 3 x readings (StandardScaled)\n).unsqueeze(0)  # → [1, 3]\n# Note: u (controls) is NOT passed to the flow\n\n# Generate 1000 possible 4-hour futures instantly\nwith torch.no_grad():\n    theta_samples = model.sample(\n        num_samples=1000,\n        condition=condition\n    )  # → [1000, 12] PCA coefficients`,
    Visual: AnimatedOutput,
  },
  {
    id: 'reference',
    title: '6. Script Reference',
    icon: Terminal,
    description: 'A full script-by-script breakdown — what each file does, what it reads, what it writes, and the single most important design decision inside it.',
    why: '',
    codeSnippet: '',
    Visual: AnimatedScriptSummary,
  },
];

// ─── EXPORTED COMPONENT ───────────────────────────────────────────────────────

export default function PipelineOverviewSection() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">

      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
        <div className="p-3 bg-cyan-100 text-cyan-700 rounded-xl shadow-sm border border-cyan-200">
          <GitMerge className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800 tracking-tight">Full Pipeline Overview</h2>
          <p className="text-sm text-slate-500 font-medium">
            How all 6 scripts connect — from raw XLSX to a trained conditional flow
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
                ? 'bg-cyan-500 scale-y-110 shadow-sm'
                : idx < currentStep
                ? 'bg-cyan-300'
                : 'bg-slate-100 hover:bg-slate-200'
            }`}
          />
        ))}
      </div>

      {/* Main Content */}
      {step.id === 'reference' ? (
        <div className="flex-1 flex flex-col gap-4 min-w-0 min-h-[500px]">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2.5 bg-slate-900 rounded-xl shadow-md">
              <step.icon size={22} className="text-cyan-400" />
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
          </div>
          <div className="flex-1 w-full rounded-2xl shadow-xl overflow-hidden border-4 border-slate-900/5 bg-slate-900">
            <step.Visual />
          </div>
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[500px]">

          {/* Left: Visual + Explanation */}
          <div className="lg:col-span-7 flex flex-col gap-6 min-w-0">
            <div className="w-full h-[380px] rounded-2xl shadow-xl overflow-hidden border-4 border-slate-900/5 bg-[#0d1117]">
              <step.Visual />
            </div>

            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex-1 flex flex-col">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2.5 bg-cyan-100 rounded-xl text-cyan-600 shadow-sm border border-cyan-200">
                  <step.icon size={20} />
                </div>
                <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
              </div>
              <p className="text-slate-600 leading-relaxed text-[15px] mb-5">{step.description}</p>
              {step.why && (
                <div className="bg-amber-50/80 border border-amber-200 p-5 rounded-xl shadow-sm relative mt-auto">
                  <div className="absolute top-0 left-0 w-1.5 h-full bg-amber-400 rounded-l-xl"></div>
                  <h4 className="text-sm font-bold text-amber-900 mb-2 flex items-center gap-2 uppercase tracking-wide">
                    <Activity size={16} className="text-amber-600" /> Design Rationale
                  </h4>
                  <p className="text-sm text-amber-800/90 leading-relaxed">{step.why}</p>
                </div>
              )}
            </div>
          </div>

          {/* Right: Code */}
          <div className="lg:col-span-5 flex flex-col min-w-0 h-full">
            <div className="flex-1 bg-[#0f172a] rounded-2xl shadow-xl flex flex-col overflow-hidden border border-slate-700">
              <div className="bg-slate-800 px-4 py-3 border-b border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Terminal size={14} className="text-cyan-400" />
                  <span className="text-xs font-bold font-mono text-slate-300">pipeline overview</span>
                </div>
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-rose-500/80"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80"></div>
                </div>
              </div>
              <div className="p-5 overflow-auto flex-1 text-[13px] font-mono leading-relaxed text-slate-300 pipeline-scroll">
                <style dangerouslySetInnerHTML={{__html:`
                  .pipeline-scroll::-webkit-scrollbar{width:6px;height:6px;}
                  .pipeline-scroll::-webkit-scrollbar-track{background:transparent;}
                  .pipeline-scroll::-webkit-scrollbar-thumb{background:#334155;border-radius:4px;}
                `}}/>
                <pre className="whitespace-pre-wrap text-[12px] leading-relaxed">
                  <code className="text-slate-400">{step.codeSnippet}</code>
                </pre>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* Bottom Navigation */}
      <div className="flex items-center justify-between mt-8 pt-6 border-t border-slate-200">
        <button
          onClick={() => setCurrentStep(p => Math.max(0, p - 1))}
          disabled={currentStep === 0}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
        >
          <ChevronLeft size={18} /> Previous
        </button>
        <button
          onClick={() => setCurrentStep(p => Math.min(steps.length - 1, p + 1))}
          disabled={currentStep === steps.length - 1}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-white bg-cyan-500 hover:bg-cyan-600 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:translate-y-0"
        >
          {currentStep === steps.length - 1 ? 'Done' : 'Next'} <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}
