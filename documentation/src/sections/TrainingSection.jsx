import React, { useState, useEffect, useRef } from 'react';
import {
  Layers,
  ChevronRight,
  ChevronLeft,
  Activity,
  Terminal,
  Play,
  Pause,
  Database,
  Cpu,
  Settings,
  TrendingDown,
  BarChart2,
  GitMerge,
} from 'lucide-react';

// ==========================================
// UTILITIES & STYLING
// ==========================================

const highlightCode = (code) => {
  if (!code) return '';
  let html = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const tokens = [];
  const pushToken = (match, className) => {
    tokens.push(`<span class="${className}">${match}</span>`);
    return `TOKENz${tokens.length - 1}z`;
  };

  html = html.replace(/(#.*)/g, (m) => pushToken(m, 'text-slate-500 italic'));
  html = html.replace(/('.*?'|".*?")/g, (m) => pushToken(m, 'text-emerald-300'));
  html = html.replace(/\b(\d+\.\d+|\d+)\b/g, (m) => pushToken(m, 'text-purple-300'));

  const pytorchRegex =
    /\b(DataLoader|wandb\.init|wandb\.finish|SCADAPipelineDataset|PipelineConditionalFlow|SMPCTrainer|torch\.backends\.mps\.is_available|torch\.cuda\.is_available|trainer\.train)\b/g;
  html = html.replace(pytorchRegex, (m) => pushToken(m, 'text-amber-300'));

  const kwRegex =
    /\b(def|class|if|else|elif|for|return|import|from|as|not|in|True|False|print)\b/g;
  html = html.replace(kwRegex, (m) => pushToken(m, 'text-rose-400 font-bold'));

  const paramRegex =
    /\b(device|model|trainer|train_dataset|val_dataset|test_dataset|train_dataloader|val_dataloader|BATCH_SIZE|EPOCHS|LEARNING_RATE|NUM_LAYERS|HIDDEN_DIM|NUM_BINS|BOUND|LOG_WANDB|DATA_PATH|dim_theta|dim_condition|sample)\b/g;
  html = html.replace(paramRegex, (m) => pushToken(m, 'text-blue-300 italic'));

  for (let i = tokens.length - 1; i >= 0; i--) {
    html = html.replace(`TOKENz${i}z`, tokens[i]);
  }
  return html;
};

const VisualButton = ({ onClick, disabled, children, active }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-bold transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 z-10 ${
      active
        ? 'bg-orange-600 hover:bg-orange-500 text-white shadow-orange-900/50'
        : 'bg-slate-700 hover:bg-slate-600 text-slate-200 shadow-slate-900/50 border border-slate-600'
    }`}
  >
    {children}
  </button>
);

// ==========================================
// INTERACTIVE VISUAL COMPONENTS
// ==========================================

const AnimatedConfig = () => {
  const [revealed, setRevealed] = useState(false);

  const params = [
    { label: 'BATCH_SIZE', value: '1024', color: 'sky', note: 'Optimised for Apple Silicon MPS unified memory' },
    { label: 'EPOCHS', value: '50', color: 'violet', note: 'Full passes through the training dataset' },
    { label: 'LEARNING_RATE', value: '5e-4', color: 'rose', note: 'Tightened vs Affine flows — splines are sensitive' },
    { label: 'NUM_LAYERS', value: '6', color: 'amber', note: 'Coupling layer relay-race depth' },
    { label: 'HIDDEN_DIM', value: '128', color: 'emerald', note: 'Residual MLP neuron count per block' },
    { label: 'NUM_BINS', value: '8', color: 'fuchsia', note: 'Rational-Quadratic knots per spline curve' },
    { label: 'BOUND', value: '5.0', color: 'teal', note: 'Hard box boundary — matches RobustScaler output' },
  ];

  const colorMap = {
    sky: 'bg-sky-900/40 border-sky-500/60 text-sky-300',
    violet: 'bg-violet-900/40 border-violet-500/60 text-violet-300',
    rose: 'bg-rose-900/40 border-rose-500/60 text-rose-300',
    amber: 'bg-amber-900/40 border-amber-500/60 text-amber-300',
    emerald: 'bg-emerald-900/40 border-emerald-500/60 text-emerald-300',
    fuchsia: 'bg-fuchsia-900/40 border-fuchsia-500/60 text-fuchsia-300',
    teal: 'bg-teal-900/40 border-teal-500/60 text-teal-300',
  };

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">
          Hyperparameter Configuration Block
        </div>

        <div className="grid grid-cols-1 gap-2 w-full max-w-lg mx-auto">
          {params.map((p, i) => (
            <div
              key={p.label}
              className={`flex items-center gap-3 rounded-lg border px-3 py-2 transition-all duration-500 ${
                revealed ? colorMap[p.color] : 'bg-slate-800 border-slate-700 text-slate-500'
              }`}
              style={{ transitionDelay: revealed ? `${i * 60}ms` : '0ms' }}
            >
              <span className="font-mono text-[11px] font-bold w-32 flex-shrink-0">{p.label}</span>
              <span
                className={`font-mono text-[13px] font-extrabold w-14 text-center flex-shrink-0 transition-all duration-500 ${
                  revealed ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
                }`}
                style={{ transitionDelay: revealed ? `${i * 60 + 100}ms` : '0ms' }}
              >
                {p.value}
              </span>
              <span
                className={`text-[10px] leading-tight transition-all duration-500 ${
                  revealed ? 'opacity-70' : 'opacity-0'
                }`}
                style={{ transitionDelay: revealed ? `${i * 60 + 150}ms` : '0ms' }}
              >
                {p.note}
              </span>
            </div>
          ))}
        </div>

        <p className="text-center text-[11px] text-slate-400 leading-relaxed px-2 max-w-lg mx-auto">
          {revealed
            ? 'Every hyperparameter flows into both the DataLoader and model constructor — changing one value here cascades through the entire pipeline.'
            : 'These constants sit at the very top of train.py. They control every aspect of training — from memory batch size to the physical bounding box of the spline math.'}
        </p>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setRevealed(!revealed)} active={revealed}>
          <Activity size={14} /> {revealed ? 'Hide Values' : 'Reveal All Hyperparameters'}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedDevice = () => {
  const [selected, setSelected] = useState(null); // 'mps' | 'cuda' | 'cpu'

  const devices = [
    {
      id: 'mps',
      label: 'Apple Silicon (MPS)',
      icon: '🍎',
      color: 'emerald',
      available: true,
      note: 'torch.backends.mps.is_available() → True. Unified memory means no CPU↔GPU copy overhead — ideal for large SCADA batches.',
      speed: 95,
    },
    {
      id: 'cuda',
      label: 'NVIDIA GPU (CUDA)',
      icon: '⚡',
      color: 'amber',
      available: false,
      note: 'torch.cuda.is_available() checked second. Fastest for large clusters, but not available on this M4 Max machine.',
      speed: 85,
    },
    {
      id: 'cpu',
      label: 'CPU Fallback',
      icon: '🖥',
      color: 'rose',
      available: false,
      note: 'Guaranteed fallback. All PyTorch ops supported — but training a spline flow on CPU would take hours per epoch.',
      speed: 15,
    },
  ];

  const colorClass = {
    emerald: { border: 'border-emerald-500', bg: 'bg-emerald-900/30', text: 'text-emerald-300', bar: 'bg-emerald-500' },
    amber:   { border: 'border-amber-500',   bg: 'bg-amber-900/30',   text: 'text-amber-300',   bar: 'bg-amber-500'   },
    rose:    { border: 'border-rose-500',     bg: 'bg-rose-900/30',     text: 'text-rose-300',     bar: 'bg-rose-500'     },
  };

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">
          Device Auto-Detection Priority Chain
        </div>

        <div className="flex flex-col gap-3 w-full max-w-md mx-auto">
          {devices.map((d, i) => {
            const c = colorClass[d.color];
            const isSelected = selected === d.id;
            return (
              <button
                key={d.id}
                onClick={() => setSelected(isSelected ? null : d.id)}
                className={`w-full text-left rounded-xl border-2 p-3 transition-all duration-300 ${
                  isSelected ? `${c.border} ${c.bg} shadow-lg` : 'border-slate-700 bg-slate-800 hover:border-slate-500'
                }`}
              >
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-lg">{d.icon}</span>
                  <span className={`font-bold text-sm ${isSelected ? c.text : 'text-slate-300'}`}>{d.label}</span>
                  <span className={`ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full ${d.available ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-600' : 'bg-slate-700 text-slate-500 border border-slate-600'}`}>
                    {d.available ? '✓ AVAILABLE' : 'NOT FOUND'}
                  </span>
                </div>
                {/* Speed bar */}
                <div className="w-full bg-slate-700 rounded-full h-1.5 mb-2">
                  <div
                    className={`${c.bar} h-1.5 rounded-full transition-all duration-700`}
                    style={{ width: isSelected ? `${d.speed}%` : '0%' }}
                  ></div>
                </div>
                {isSelected && (
                  <p className="text-[10px] text-slate-400 leading-relaxed mt-1">{d.note}</p>
                )}
              </button>
            );
          })}
        </div>

        <p className="text-center text-[11px] text-slate-400 leading-relaxed px-2 max-w-md mx-auto">
          Click each device to inspect. The priority chain checks MPS → CUDA → CPU and assigns the first available one to the <code className="text-slate-300 bg-slate-800 px-1 rounded">device</code> string.
        </p>
      </div>
    </div>
  );
};

const AnimatedDataLoader = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_STEP = 3;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(s => { if (s + 1 > MAX_STEP) { setIsPlaying(false); return s; } return s + 1; }), 2200);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  const splits = [
    { label: 'train', color: 'bg-blue-500', flex: 'flex-[8]', text: '80%' },
    { label: 'val',   color: 'bg-amber-500', flex: 'flex-[1]', text: '10%' },
    { label: 'test',  color: 'bg-rose-500',  flex: 'flex-[1]', text: '10%' },
  ];

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">
          Dataset Splits → DataLoaders → GPU Batches
        </div>

        {/* Step 0: Raw Parquet */}
        <div className="w-full max-w-md mx-auto flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Pre-split by preprocessing.py</span>
            <div className="w-full h-6 flex rounded-lg overflow-hidden border border-slate-700 gap-0.5">
              {splits.map(s => (
                <div key={s.label} className={`${s.flex} ${s.color} flex items-center justify-center text-[9px] font-bold text-white`}>
                  {s.label}.parquet
                </div>
              ))}
            </div>
            <div className="flex gap-2 mt-1">
              {splits.map(s => (
                <div key={s.label} className="flex items-center gap-1">
                  <div className={`w-2 h-2 rounded-full ${s.color}`}></div>
                  <span className="text-[9px] text-slate-400 font-mono">{s.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Step 1: SCADAPipelineDataset objects */}
          <div className={`flex gap-2 transition-all duration-500 ${step >= 1 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            {splits.map(s => (
              <div key={s.label} className="flex-1 bg-slate-800 border border-slate-600 rounded-lg p-2 flex flex-col items-center gap-1">
                <Database size={14} className="text-slate-400" />
                <span className="text-[9px] font-bold text-slate-300 font-mono">{s.label}_dataset</span>
                <span className="text-[8px] text-slate-500">SCADAPipelineDataset</span>
              </div>
            ))}
          </div>

          {/* Step 2: DataLoader wrappers */}
          <div className={`flex gap-2 transition-all duration-500 ${step >= 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            {splits.map(s => (
              <div key={s.label} className="flex-1 bg-indigo-900/30 border border-indigo-500/50 rounded-lg p-2 flex flex-col items-center gap-1">
                <Layers size={14} className="text-indigo-400" />
                <span className="text-[9px] font-bold text-indigo-300 font-mono">{s.label}_loader</span>
                <span className="text-[8px] text-indigo-500">DataLoader</span>
              </div>
            ))}
          </div>

          {/* Step 3: Batch going to GPU */}
          <div className={`transition-all duration-500 ${step >= 3 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            <div className="bg-emerald-900/20 border border-emerald-500/50 rounded-xl p-3 flex items-center gap-3">
              <Cpu size={20} className="text-emerald-400 flex-shrink-0" />
              <div>
                <span className="text-[11px] font-bold text-emerald-300">Batch → MPS GPU</span>
                <p className="text-[9px] text-emerald-600 mt-0.5">
                  1024 rows · 8 workers · persistent_workers=True · prefetch_factor=2
                </p>
              </div>
            </div>
          </div>
        </div>

        <p className="text-center text-[11px] text-slate-400 leading-relaxed px-2 max-w-md mx-auto">
          {step === 0 && 'preprocessing.py already saved train.parquet, val.parquet, and test.parquet. SCADAPipelineDataset reads the right file directly — train.py never splits anything.'}
          {step === 1 && "Each SCADAPipelineDataset loads its own pre-split parquet, extracting the x_cols, u_cols, and PCA theta_cols into PyTorch tensors. The data is already scaled and windowed."}
          {step === 2 && 'Each dataset is wrapped in a DataLoader with identical kwargs — batch size, workers, and pin_memory are shared.'}
          {step === 3 && '8 background workers pre-load the next batch while the GPU trains the current one, eliminating I/O bottlenecks on the M4 Max.'}
        </p>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setStep(s => Math.max(0, s - 1))} active={false} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play'}
        </VisualButton>
        <VisualButton onClick={() => setStep(s => Math.min(MAX_STEP, s + 1))} active={true} disabled={step === MAX_STEP || isPlaying}>
          Next <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedModelBuild = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_STEP = 3;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(s => { if (s + 1 > MAX_STEP) { setIsPlaying(false); return s; } return s + 1; }), 2000);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">
          Dynamic Model Sizing from Live Data
        </div>

        <div className="w-full max-w-md mx-auto flex flex-col gap-3 font-mono text-[11px]">

          {/* Step 0: peek at dataset */}
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 flex flex-col gap-2">
            <span className="text-slate-400 text-[10px] font-bold uppercase tracking-wider">Peek first sample</span>
            <div className="flex gap-2">
              <span className="text-slate-500">sample = train_dataset[0]</span>
            </div>
            <div className={`flex gap-4 transition-all duration-500 ${step >= 1 ? 'opacity-100' : 'opacity-0'}`}>
              <div className="flex flex-col items-center gap-1">
                <span className="text-blue-400 font-bold">theta</span>
                <span className="bg-blue-900/40 border border-blue-500/50 text-blue-300 px-2 py-1 rounded">shape: [4]</span>
                <span className="text-slate-500 text-[9px]">4 PCA coefficients</span>
              </div>
              <div className="flex flex-col items-center gap-1">
                <span className="text-rose-400 font-bold">condition</span>
                <span className="bg-rose-900/40 border border-rose-500/50 text-rose-300 px-2 py-1 rounded">shape: [6]</span>
                <span className="text-slate-500 text-[9px]">3 X + 3 U sensors</span>
              </div>
            </div>
          </div>

          {/* Step 2: dim sizes */}
          <div className={`flex gap-3 transition-all duration-500 ${step >= 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
            <div className="flex-1 bg-blue-900/20 border border-blue-500/40 rounded-lg p-2 text-center">
              <div className="text-[9px] text-slate-400 mb-1">dim_theta</div>
              <div className="text-xl font-extrabold text-blue-300">4</div>
              <div className="text-[8px] text-slate-500">sample['theta'].shape[0]</div>
            </div>
            <div className="flex-1 bg-rose-900/20 border border-rose-500/40 rounded-lg p-2 text-center">
              <div className="text-[9px] text-slate-400 mb-1">dim_condition</div>
              <div className="text-xl font-extrabold text-rose-300">6</div>
              <div className="text-[8px] text-slate-500">sample['condition'].shape[0]</div>
            </div>
          </div>

          {/* Step 3: model instantiation */}
          <div className={`bg-indigo-900/20 border-2 border-indigo-500/60 rounded-xl p-3 flex flex-col gap-2 transition-all duration-500 shadow-[0_0_20px_rgba(99,102,241,0.15)] ${step >= 3 ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
            <div className="flex items-center gap-2 mb-1">
              <Cpu size={16} className="text-indigo-400" />
              <span className="text-indigo-300 font-bold text-[11px]">PipelineConditionalFlow</span>
            </div>
            {[
              ['dim_theta', '4', 'blue'],
              ['dim_condition', '6', 'rose'],
              ['num_layers', '6', 'amber'],
              ['hidden_dim', '128', 'emerald'],
              ['num_bins', '8', 'fuchsia'],
              ['bound', '5.0', 'teal'],
            ].map(([k, v, c]) => (
              <div key={k} className="flex justify-between items-center text-[10px]">
                <span className="text-slate-400">{k}</span>
                <span className={`font-bold text-${c}-300`}>{v}</span>
              </div>
            ))}
          </div>
        </div>

        <p className="text-center text-[11px] text-slate-400 leading-relaxed px-2 max-w-md mx-auto">
          {step === 0 && 'Rather than hard-coding tensor sizes, train.py peeks at the first dataset sample to measure them dynamically.'}
          {step === 1 && "sample['theta'] carries the 4 PCA coefficients. sample['condition'] carries all 6 sensor context inputs."}
          {step === 2 && 'shape[0] extracts the actual integer dimension — if you add more sensors or PCA components, the model resizes automatically.'}
          {step === 3 && 'All six arguments are passed to PipelineConditionalFlow. dim_theta and dim_condition come from the live data; the rest come from the config block at the top.'}
        </p>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setStep(s => Math.max(0, s - 1))} active={false} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play'}
        </VisualButton>
        <VisualButton onClick={() => setStep(s => Math.min(MAX_STEP, s + 1))} active={true} disabled={step === MAX_STEP || isPlaying}>
          Next <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedTrainingLoop = () => {
  const [epoch, setEpoch] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_EPOCH = 50;

  // Simulated loss curve
  const loss = (e) => +(2.8 * Math.exp(-0.07 * e) + 0.3 + (Math.random() * 0.05 - 0.025)).toFixed(3);
  const [losses, setLosses] = useState([]);

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => {
        setEpoch(e => {
          if (e >= MAX_EPOCH) { setIsPlaying(false); return e; }
          setLosses(prev => [...prev, { e: e + 1, v: loss(e + 1) }]);
          return e + 1;
        });
      }, 120);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  const handleReset = () => { setEpoch(0); setLosses([]); setIsPlaying(false); };

  const maxLoss = 3.2;
  const chartH = 80;
  const chartW = 260;

  const svgPoints = losses.map((l, i) => {
    const x = (i / (MAX_EPOCH - 1)) * chartW;
    const y = chartH - (l.v / maxLoss) * chartH;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">
          SMPCTrainer — Live Training Loop Simulation
        </div>

        {/* Epoch counter */}
        <div className="flex justify-center gap-6 w-full max-w-md mx-auto">
          <div className="flex flex-col items-center bg-slate-800 border border-slate-700 rounded-xl px-5 py-3">
            <span className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Epoch</span>
            <span className="text-3xl font-extrabold text-white font-mono">{String(epoch).padStart(2, '0')}</span>
            <span className="text-[9px] text-slate-500">/ {MAX_EPOCH}</span>
          </div>
          <div className="flex flex-col items-center bg-slate-800 border border-slate-700 rounded-xl px-5 py-3">
            <span className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">NLL Loss</span>
            <span className="text-3xl font-extrabold text-orange-400 font-mono">
              {losses.length > 0 ? losses[losses.length - 1].v : '—'}
            </span>
            <span className="text-[9px] text-slate-500">lower = better</span>
          </div>
        </div>

        {/* Loss chart */}
        <div className="w-full max-w-md mx-auto bg-slate-800/60 border border-slate-700 rounded-xl p-3">
          <div className="text-[9px] text-slate-500 font-bold uppercase tracking-wider mb-2">Training Loss Curve</div>
          <svg width="100%" viewBox={`0 0 ${chartW} ${chartH}`} className="overflow-visible">
            {/* Grid lines */}
            {[0.25, 0.5, 0.75, 1].map(f => (
              <line key={f} x1="0" y1={chartH * (1 - f)} x2={chartW} y2={chartH * (1 - f)}
                stroke="#334155" strokeWidth="0.5" strokeDasharray="3" />
            ))}
            {losses.length > 1 && (
              <polyline points={svgPoints} fill="none" stroke="#f97316" strokeWidth="2"
                className="drop-shadow-[0_0_4px_#f97316]" />
            )}
            {losses.length > 0 && (() => {
              const last = losses[losses.length - 1];
              const x = ((losses.length - 1) / (MAX_EPOCH - 1)) * chartW;
              const y = chartH - (last.v / maxLoss) * chartH;
              return <circle cx={x} cy={y} r="3" fill="#f97316" className="animate-pulse" />;
            })()}
          </svg>
        </div>

        {/* W&B status */}
        <div className="w-full max-w-md mx-auto flex items-center gap-3 bg-slate-800 border border-slate-700 rounded-xl p-3">
          <BarChart2 size={16} className={`flex-shrink-0 ${isPlaying || epoch > 0 ? 'text-emerald-400 animate-pulse' : 'text-slate-600'}`} />
          <div>
            <span className={`text-[11px] font-bold ${isPlaying || epoch > 0 ? 'text-emerald-300' : 'text-slate-500'}`}>
              Weights &amp; Biases Dashboard
            </span>
            <p className="text-[9px] text-slate-500 leading-tight mt-0.5">
              {epoch > 0
                ? `Logging epoch ${epoch}/${MAX_EPOCH} — train_loss, val_loss, grad_norm streamed live`
                : 'wandb.init() called — project: smpc-spline-flow'}
            </p>
          </div>
        </div>

        <p className="text-center text-[11px] text-slate-400 leading-relaxed px-2 max-w-md mx-auto">
          {epoch === 0 && 'Press Play to simulate the training loop. trainer.train() hands off all epoch logic to SMPCTrainer.'}
          {epoch > 0 && epoch < MAX_EPOCH && `Each epoch: forward pass → NLL loss → loss.backward() → AdamW step → val check → W&B log.`}
          {epoch >= MAX_EPOCH && 'Training complete! The best checkpoint is saved to outputs/checkpoints/model_best.pt via SMPCTrainer.'}
        </p>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={handleReset} active={false} disabled={isPlaying}>
          Reset
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying} disabled={epoch >= MAX_EPOCH}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause Training' : 'Start Training'}
        </VisualButton>
      </div>
    </div>
  );
};

// ─── Full Code Walkthrough ───────────────────────────────────────────────────

const AnimatedWalkthrough = () => {
  const [activePart, setActivePart] = useState(0);
  const lineRefs = useRef({});

  const codeLines = [
    { text: "import os, sys, torch", part: null },
    { text: "from torch.utils.data import DataLoader", part: null },
    { text: "import wandb", part: null },
    { text: "", part: null },
    { text: "os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'", part: null },
    { text: "", part: null },
    { text: "# ── CONFIGURATION ──────────────────────────────", part: 0 },
    { text: "DATA_PATH      = 'data/processed'", part: 0 },
    { text: "BATCH_SIZE     = 1024     # Optimised for MPS", part: 0 },
    { text: "EPOCHS         = 50", part: 0 },
    { text: "LEARNING_RATE  = 5e-4     # Tightened for splines", part: 0 },
    { text: "NUM_LAYERS     = 6", part: 0 },
    { text: "HIDDEN_DIM     = 128", part: 0 },
    { text: "NUM_BINS       = 8        # RQ-spline bins", part: 0 },
    { text: "BOUND          = 5.0      # Matches RobustScaler", part: 0 },
    { text: "", part: null },
    { text: "# ── DEVICE ─────────────────────────────────────", part: 1 },
    { text: "device = 'mps' if torch.backends.mps.is_available()", part: 1 },
    { text: "    else 'cuda' if torch.cuda.is_available()", part: 1 },
    { text: "    else 'cpu'", part: 1 },
    { text: "", part: null },
    { text: "# ── W&B INIT ───────────────────────────────────", part: 1 },
    { text: "if LOG_WANDB:", part: 1 },
    { text: "    wandb.init(project='smpc-spline-flow', config={...})", part: 1 },
    { text: "", part: null },
    { text: "# ── DATASETS & DATALOADERS ─────────────────────", part: 2 },
    { text: "train_dataset = SCADAPipelineDataset(DATA_PATH, 'train')", part: 2 },
    { text: "val_dataset   = SCADAPipelineDataset(DATA_PATH, 'val')", part: 2 },
    { text: "test_dataset  = SCADAPipelineDataset(DATA_PATH, 'test')", part: 2 },
    { text: "", part: 2 },
    { text: "dataloader_kwargs = {", part: 2 },
    { text: "    'batch_size': BATCH_SIZE,", part: 2 },
    { text: "    'num_workers': 8,", part: 2 },
    { text: "    'persistent_workers': True,", part: 2 },
    { text: "    'prefetch_factor': 2,", part: 2 },
    { text: "    'pin_memory': True", part: 2 },
    { text: "}", part: 2 },
    { text: "", part: null },
    { text: "train_dataloader = DataLoader(train_dataset, shuffle=True,  **dataloader_kwargs)", part: 2 },
    { text: "val_dataloader   = DataLoader(val_dataset,   shuffle=False, **dataloader_kwargs)", part: 2 },
    { text: "test_dataloader  = DataLoader(test_dataset,  shuffle=False, **dataloader_kwargs)", part: 2 },
    { text: "", part: null },
    { text: "# ── DYNAMIC MODEL SIZING ───────────────────────", part: 3 },
    { text: "sample        = train_dataset[0]", part: 3 },
    { text: "dim_theta     = sample['theta'].shape[0]", part: 3 },
    { text: "dim_condition = sample['condition'].shape[0]", part: 3 },
    { text: "", part: 3 },
    { text: "model = PipelineConditionalFlow(", part: 3 },
    { text: "    dim_theta=dim_theta,", part: 3 },
    { text: "    dim_condition=dim_condition,", part: 3 },
    { text: "    num_layers=NUM_LAYERS,", part: 3 },
    { text: "    hidden_dim=HIDDEN_DIM,", part: 3 },
    { text: "    num_bins=NUM_BINS,", part: 3 },
    { text: "    bound=BOUND", part: 3 },
    { text: ")", part: 3 },
    { text: "", part: null },
    { text: "# ── TRAIN ──────────────────────────────────────", part: 4 },
    { text: "trainer = SMPCTrainer(", part: 4 },
    { text: "    model=model,", part: 4 },
    { text: "    train_dataloader=train_dataloader,", part: 4 },
    { text: "    val_dataloader=val_dataloader,", part: 4 },
    { text: "    learning_rate=LEARNING_RATE,", part: 4 },
    { text: "    epochs=EPOCHS,", part: 4 },
    { text: "    device=device,", part: 4 },
    { text: "    log_to_wandb=LOG_WANDB", part: 4 },
    { text: ")", part: 4 },
    { text: "", part: 4 },
    { text: "trainer.train()", part: 4 },
    { text: "wandb.finish()", part: 4 },
  ];

  const partExplanations = [
    {
      title: 'Configuration Block',
      exp: 'All seven hyperparameters are declared as constants at the very top. This means you only ever edit one place to change training behaviour — no hunting through class constructors.',
    },
    {
      title: 'Device Detection & W&B Init',
      exp: "A single Python ternary probes MPS → CUDA → CPU in priority order and sets 'device'. wandb.init() is called immediately after, registering the full config to the dashboard before any tensors are created.",
    },
    {
      title: 'Datasets & DataLoaders',
      exp: 'Three SCADAPipelineDataset objects are created for the three chronological splits. The dataloader_kwargs dict is shared across all three DataLoader calls — this prevents silent divergence if you tune one loader but forget another.',
    },
    {
      title: 'Dynamic Model Sizing',
      exp: "The model is never hard-coded to dim_theta=4 or dim_condition=6. Instead it peeks at sample[0] from the live dataset. If you change the number of PCA components or add sensors in preprocessing.py, the model auto-resizes with zero code changes here.",
    },
    {
      title: 'SMPCTrainer & Launch',
      exp: 'All epoch logic — forward pass, NLL loss, gradient clipping, AdamW step, validation loop, checkpoint saving, and W&B logging — lives inside SMPCTrainer. train.py simply assembles the pieces and calls trainer.train().',
    },
  ];

  useEffect(() => {
    if (lineRefs.current[activePart]) {
      lineRefs.current[activePart].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activePart]);

  return (
    <div className="relative w-full h-[600px] md:h-full bg-[#0d1117] overflow-hidden flex flex-col md:flex-row">
      <div className="flex-1 overflow-auto py-4 pl-4 pr-4 font-mono text-[10px] sm:text-xs text-slate-300 code-scroll">
        <style dangerouslySetInnerHTML={{__html: `
          .code-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
          .code-scroll::-webkit-scrollbar-track { background: transparent; }
          .code-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        `}} />
        <div className="w-full">
          {codeLines.map((line, i) => (
            <div
              key={i}
              ref={el => { if (line.part !== null && !lineRefs.current[line.part]) lineRefs.current[line.part] = el; }}
              className={`px-2 py-0.5 border-l-[3px] transition-all duration-300 whitespace-pre-wrap break-words ${
                line.part === activePart
                  ? 'bg-orange-500/20 border-orange-500 text-orange-100'
                  : line.part !== null
                    ? 'border-transparent opacity-50 hover:opacity-100 cursor-pointer'
                    : 'border-transparent text-slate-500'
              }`}
              onClick={() => { if (line.part !== null) setActivePart(line.part); }}
              dangerouslySetInnerHTML={{ __html: highlightCode(line.text) || ' ' }}
            />
          ))}
        </div>
      </div>

      <div className="w-full min-h-[180px] md:h-full md:w-80 lg:w-96 flex-shrink-0 bg-slate-800 md:border-l border-t md:border-t-0 border-slate-700 p-4 md:p-5 flex flex-col justify-between z-10 shadow-[0_-4px_15px_rgba(0,0,0,0.3)] md:shadow-[-4px_0_15px_rgba(0,0,0,0.3)] overflow-y-auto">
        <div className="animate-in fade-in slide-in-from-right-2 duration-300" key={activePart}>
          <div className="text-[10px] font-bold text-orange-400 mb-1.5 uppercase tracking-wider">
            Part {activePart + 1} of {partExplanations.length}
          </div>
          <h4 className="text-sm sm:text-base font-bold text-white mb-2">{partExplanations[activePart].title}</h4>
          <p className="text-[11px] sm:text-xs text-slate-300 leading-relaxed bg-slate-900/50 p-3 sm:p-4 rounded-lg border border-slate-700 shadow-inner">
            {partExplanations[activePart].exp}
          </p>
        </div>
        <div className="flex gap-2 justify-between mt-4 md:mt-6">
          <VisualButton onClick={() => setActivePart(p => Math.max(0, p - 1))} disabled={activePart === 0}>
            <ChevronLeft size={16} /> Prev
          </VisualButton>
          <VisualButton onClick={() => setActivePart(p => Math.min(partExplanations.length - 1, p + 1))} disabled={activePart === partExplanations.length - 1} active>
            Next <ChevronRight size={16} />
          </VisualButton>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// MAIN CONFIGURATION & STEPS
// ==========================================

const steps = [
  {
    id: 'config',
    title: '1. Hyperparameter Config',
    icon: Settings,
    codeSnippet: `BATCH_SIZE    = 1024    # Optimised for Apple Silicon MPS\nEPOCHS        = 50\nLEARNING_RATE = 5e-4    # Tightened: splines need lower LR\nNUM_LAYERS    = 6       # Coupling layer depth\nHIDDEN_DIM    = 128     # Residual MLP size\nNUM_BINS      = 8       # Rational-Quadratic knot count\nBOUND         = 5.0     # Box boundary [-5, 5]`,
    description: "All seven training hyperparameters are declared as uppercase constants at the very top of the file. Every downstream object — DataLoaders, the model, and the trainer — reads from these constants.",
    why: "Centralising hyperparameters means a single edit cascades correctly through the entire pipeline. There is no risk of updating BATCH_SIZE in the DataLoader but forgetting to update it in the model or W&B config.",
    Visual: AnimatedConfig,
  },
  {
    id: 'device',
    title: '2. Device Detection',
    icon: Cpu,
    codeSnippet: `device = (\n    "mps"  if torch.backends.mps.is_available() else\n    "cuda" if torch.cuda.is_available()        else\n    "cpu"\n)\n\nif LOG_WANDB:\n    wandb.init(\n        project="smpc-spline-flow",\n        config={"device": device, "epochs": EPOCHS, ...}\n    )`,
    description: "A single Python ternary chain probes three backends in order. MPS (Apple Silicon Metal) is checked first because this project is optimised for M4 Max. W&B is initialised immediately after so all hardware metadata is captured before any tensor is created.",
    why: "MPS unified memory eliminates the CPU↔GPU data copy overhead that plagues CUDA systems. For 1,024-row SCADA batches with 14,400-point trajectory context, this is the most important single performance decision in the entire script.",
    Visual: AnimatedDevice,
  },
  {
    id: 'dataloaders',
    title: '3. Datasets & DataLoaders',
    icon: Database,
    codeSnippet: `train_dataset = SCADAPipelineDataset(\n    data_path='data/processed',  # reads data/processed/train.parquet\n    split='train'\n)\nval_dataset   = SCADAPipelineDataset(\n    data_path='data/processed',\n    split='val'\n)\ndataloader_kwargs = {\n    "batch_size": BATCH_SIZE,\n    "num_workers": 8,\n    "persistent_workers": True,\n    "prefetch_factor": 2,\n    "pin_memory": True\n}\n\ntrain_loader = DataLoader(train_dataset, shuffle=True,  **dataloader_kwargs)\nval_loader   = DataLoader(val_dataset,   shuffle=False, **dataloader_kwargs)`,
    description: "preprocessing.py already saved three separate parquet files — train.parquet, val.parquet, and test.parquet. SCADAPipelineDataset simply reads the correct file for its split. A shared dataloader_kwargs dict is unpacked into all three DataLoader calls so I/O settings stay consistent.",
    why: "Because the split is done at preprocessing time (not here), train.py never touches raw data — it just wraps the ready-made files. Setting num_workers=8 allows 8 CPU threads to pre-fetch batches while the GPU trains, and persistent_workers=True avoids re-spawning those threads each epoch.",
    Visual: AnimatedDataLoader,
  },
  {
    id: 'model',
    title: '4. Dynamic Model Sizing',
    icon: GitMerge,
    codeSnippet: `# Peek at one sample to measure shapes dynamically\nsample        = train_dataset[0]\ndim_theta     = sample['theta'].shape[0]      # → 4\ndim_condition = sample['condition'].shape[0]  # → 6\n\nmodel = PipelineConditionalFlow(\n    dim_theta=dim_theta,\n    dim_condition=dim_condition,\n    num_layers=NUM_LAYERS,\n    hidden_dim=HIDDEN_DIM,\n    num_bins=NUM_BINS,\n    bound=BOUND\n)`,
    description: "Instead of hard-coding tensor dimensions, train.py inspects the first dataset sample to measure them at runtime. dim_theta (4 PCA coefficients) and dim_condition (6 sensor inputs) are passed directly to the model constructor.",
    why: "This makes the entire training script data-driven. If you change N_COMPONENTS in preprocessing.py from 4 to 6, or add new sensor columns to the dataset, train.py self-adjusts with zero code changes — the model simply grows.",
    Visual: AnimatedModelBuild,
  },
  {
    id: 'training',
    title: '5. Training Loop',
    icon: TrendingDown,
    codeSnippet: `trainer = SMPCTrainer(\n    model=model,\n    train_dataloader=train_dataloader,\n    val_dataloader=val_dataloader,\n    learning_rate=LEARNING_RATE,\n    epochs=EPOCHS,\n    device=device,\n    log_to_wandb=LOG_WANDB\n)\n\ntrainer.train()\nwandb.finish()`,
    description: "The fully-assembled model, both data loaders, and all hyperparameters are handed to SMPCTrainer. A single trainer.train() call launches the full training loop — all epoch logic, gradient clipping, checkpoint saving, and W&B logging is encapsulated inside the trainer.",
    why: "Separating 'assembly' (train.py) from 'execution' (trainer.py) keeps each file small and single-purpose. If you want to swap the optimiser or change the loss function, you edit only trainer.py — train.py never changes.",
    Visual: AnimatedTrainingLoop,
  },
  {
    id: 'walkthrough',
    title: '6. Full Code Walkthrough',
    icon: Terminal,
    codeSnippet: '',
    description: 'A holistic line-by-line breakdown of train.py.',
    why: '',
    Visual: AnimatedWalkthrough,
  },
];

// ==========================================
// EXPORTED SECTION COMPONENT
// ==========================================

export default function TrainingSection() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">

      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
        <div className="p-3 bg-orange-100 text-orange-600 rounded-xl shadow-sm border border-orange-200">
          <TrendingDown className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800 tracking-tight">Training Script</h2>
          <p className="text-sm text-slate-500 font-medium">
            Interactive walkthrough of{' '}
            <code className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-700 border border-slate-200">
              scripts/train.py
            </code>
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
                ? 'bg-orange-500 scale-y-110 shadow-sm'
                : idx < currentStep
                ? 'bg-orange-300'
                : 'bg-slate-100 hover:bg-slate-200'
            }`}
          />
        ))}
      </div>

      {/* Main Content */}
      {step.id === 'walkthrough' ? (
        <div className="flex-1 flex flex-col gap-4 min-w-0 min-h-[500px]">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2.5 bg-slate-900 rounded-xl text-white shadow-md">
              <step.icon size={22} className="text-orange-400" />
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
          </div>
          <div className="flex-1 w-full rounded-2xl shadow-xl shadow-slate-200/50 overflow-hidden relative border-4 border-slate-900/5 bg-[#0d1117]">
            <step.Visual />
          </div>
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[500px]">

          {/* Left: Visual + Explanation */}
          <div className="lg:col-span-7 flex flex-col gap-6 min-w-0">
            <div className="w-full h-[380px] rounded-2xl shadow-xl overflow-hidden relative border-4 border-slate-900/5 bg-[#0d1117]">
              <step.Visual />
            </div>

            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex-1 flex flex-col">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2.5 bg-orange-100 rounded-xl text-orange-600 shadow-sm border border-orange-200">
                  <step.icon size={20} />
                </div>
                <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
              </div>

              <p className="text-slate-600 leading-relaxed text-[15px] mb-5">{step.description}</p>

              <div className="bg-amber-50/80 border border-amber-200 p-5 rounded-xl shadow-sm relative mt-auto">
                <div className="absolute top-0 left-0 w-1.5 h-full bg-amber-400 rounded-l-xl"></div>
                <h4 className="text-sm font-bold text-amber-900 mb-2 flex items-center gap-2 uppercase tracking-wide">
                  <Activity size={16} className="text-amber-600" />
                  The PyTorch Logic
                </h4>
                <p className="text-sm text-amber-800/90 leading-relaxed">{step.why}</p>
              </div>
            </div>
          </div>

          {/* Right: Code Snippet */}
          <div className="lg:col-span-5 flex flex-col min-w-0 h-full">
            <div className="flex-1 bg-[#0f172a] rounded-2xl shadow-xl flex flex-col overflow-hidden border border-slate-700">
              <div className="bg-slate-800 px-4 py-3 border-b border-slate-700 flex items-center justify-between shadow-sm">
                <div className="flex items-center gap-2">
                  <Terminal size={14} className="text-orange-400" />
                  <span className="text-xs font-bold font-mono text-slate-300">train.py snippet</span>
                </div>
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-rose-500/80"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80"></div>
                </div>
              </div>
              <div className="p-5 overflow-auto flex-1 text-[13px] font-mono leading-loose text-slate-300 custom-scrollbar">
                <style dangerouslySetInnerHTML={{__html: `
                  .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
                  .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
                  .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
                `}} />
                <pre><code dangerouslySetInnerHTML={{ __html: highlightCode(step.codeSnippet) }}></code></pre>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* Bottom navigation */}
      <div className="flex items-center justify-between mt-8 pt-6 border-t border-slate-200">
        <button
          onClick={() => setCurrentStep(p => Math.max(0, p - 1))}
          disabled={currentStep === 0}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
        >
          <ChevronLeft size={18} /> Previous Step
        </button>
        <button
          onClick={() => setCurrentStep(p => Math.min(steps.length - 1, p + 1))}
          disabled={currentStep === steps.length - 1}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-white bg-orange-500 hover:bg-orange-600 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-40 disabled:hover:translate-y-0 disabled:cursor-not-allowed disabled:shadow-none"
        >
          {currentStep === steps.length - 1 ? 'Finish Walkthrough' : 'Next Step'} <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}
