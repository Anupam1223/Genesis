import React, { useState, useEffect, useRef } from 'react';
import {
  ChevronRight,
  ChevronLeft,
  Activity,
  Terminal,
  Play,
  Pause,
  Cpu,
  Save,
  TrendingDown,
  Zap,
  Shield,
  BarChart2,
  RefreshCw,
  CheckCircle,
} from 'lucide-react';

// ==========================================
// UTILITIES & STYLING
// ==========================================

const highlightCode = (code) => {
  if (!code) return '';
  let html = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const tokens = [];
  const push = (match, cls) => { tokens.push(`<span class="${cls}">${match}</span>`); return `TOKENz${tokens.length - 1}z`; };

  html = html.replace(/(#.*)/g,            (m) => push(m, 'text-slate-500 italic'));
  html = html.replace(/('.*?'|".*?")/g,    (m) => push(m, 'text-emerald-300'));
  html = html.replace(/\b(\d+\.\d+e[-+]?\d+|\d+\.\d+|\d+)\b/g, (m) => push(m, 'text-purple-300'));

  const apiRegex = /\b(AdamW|ReduceLROnPlateau|wandb\.watch|wandb\.log|clip_grad_norm_|compute_loss|model\.train|model\.eval|torch\.no_grad|loss\.backward|optimizer\.zero_grad|optimizer\.step|scheduler\.step|torch\.save|tqdm)\b/g;
  html = html.replace(apiRegex, (m) => push(m, 'text-amber-300'));

  const kwRegex = /\b(def|class|if|else|elif|for|return|import|from|as|not|in|True|False|with|print|self)\b/g;
  html = html.replace(kwRegex, (m) => push(m, 'text-rose-400 font-bold'));

  const varRegex = /\b(model|optimizer|scheduler|loss|avg_train_loss|avg_val_loss|epoch|global_step|current_lr|best_val_loss|theta|condition|batch|is_best|checkpoint_dir|filename|path)\b/g;
  html = html.replace(varRegex, (m) => push(m, 'text-blue-300 italic'));

  for (let i = tokens.length - 1; i >= 0; i--) html = html.replace(`TOKENz${i}z`, tokens[i]);
  return html;
};

const VisualButton = ({ onClick, disabled, children, active }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-bold transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 z-10 ${
      active
        ? 'bg-violet-600 hover:bg-violet-500 text-white shadow-violet-900/50'
        : 'bg-slate-700 hover:bg-slate-600 text-slate-200 shadow-slate-900/50 border border-slate-600'
    }`}
  >
    {children}
  </button>
);

// ==========================================
// VISUAL COMPONENTS
// ==========================================

// ── Step 1: Optimizer & Scheduler ───────────────────────────────────────────
const AnimatedOptimizer = () => {
  const [revealed, setRevealed] = useState(false);
  const [activeCard, setActiveCard] = useState(null);

  const cards = [
    {
      id: 'adamw',
      title: 'AdamW Optimizer',
      color: 'violet',
      icon: <Zap size={16} />,
      params: [
        { k: 'lr', v: '5e-4', note: 'Passed in from train.py config — lower than typical affine flows' },
        { k: 'weight_decay', v: '1e-4', note: 'Tightened from default 1e-2. Too high breaks spline knot positioning' },
      ],
      why: 'AdamW decouples weight decay from the gradient update. For spline flows where the rational-quadratic denominator is extremely sensitive to weight magnitude, this prevents unintended knot drift.',
    },
    {
      id: 'scheduler',
      title: 'ReduceLROnPlateau',
      color: 'sky',
      icon: <TrendingDown size={16} />,
      params: [
        { k: 'mode',     v: "'min'",  note: 'Watches for val_loss going down' },
        { k: 'factor',   v: '0.5',   note: 'Halves the LR each time it fires' },
        { k: 'patience', v: '3',     note: '3 epochs of no improvement before halving' },
        { k: 'min_lr',   v: '1e-6',  note: 'Hard floor — LR never drops below this' },
      ],
      why: 'Spline flows train with a characteristic jagged loss curve — they can stall for 2–3 epochs before suddenly improving. patience=3 gives the optimiser room to breathe without wasting epochs at a too-high LR.',
    },
  ];

  const colorMap = {
    violet: { bg: 'bg-violet-900/30', border: 'border-violet-500/60', text: 'text-violet-300', light: 'bg-violet-500/10' },
    sky:    { bg: 'bg-sky-900/30',    border: 'border-sky-500/60',    text: 'text-sky-300',    light: 'bg-sky-500/10'    },
  };

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">SMPCTrainer.__init__ — Optimizer & Scheduler</div>

        <div className="flex flex-col gap-3 w-full max-w-lg mx-auto">
          {cards.map((card, ci) => {
            const c = colorMap[card.color];
            const isOpen = activeCard === card.id;
            return (
              <div
                key={card.id}
                className={`rounded-xl border-2 transition-all duration-300 cursor-pointer ${isOpen ? `${c.border} ${c.bg}` : 'border-slate-700 bg-slate-800 hover:border-slate-600'}`}
                style={{ transitionDelay: revealed ? `${ci * 80}ms` : '0ms' }}
                onClick={() => setActiveCard(isOpen ? null : card.id)}
              >
                <div className="flex items-center gap-3 px-4 py-3">
                  <span className={`${isOpen ? c.text : 'text-slate-500'} transition-colors`}>{card.icon}</span>
                  <span className={`font-bold text-sm font-mono ${isOpen ? c.text : 'text-slate-300'}`}>{card.title}</span>
                  <ChevronRight size={14} className={`ml-auto text-slate-500 transition-transform duration-300 ${isOpen ? 'rotate-90' : ''}`} />
                </div>

                {isOpen && (
                  <div className="px-4 pb-4 flex flex-col gap-2 animate-in fade-in slide-in-from-top-1 duration-200">
                    {card.params.map(p => (
                      <div key={p.k} className={`flex items-start gap-3 rounded-lg px-3 py-2 ${c.light} border ${c.border}`}>
                        <span className="font-mono text-[10px] font-bold text-slate-400 w-20 flex-shrink-0 pt-0.5">{p.k}</span>
                        <span className={`font-mono text-[12px] font-extrabold ${c.text} w-12 flex-shrink-0`}>{p.v}</span>
                        <span className="text-[10px] text-slate-400 leading-tight">{p.note}</span>
                      </div>
                    ))}
                    <p className="text-[10px] text-slate-300 leading-relaxed mt-1 px-1 italic border-l-2 border-violet-500/40 pl-3">
                      {card.why}
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <p className="text-center text-[11px] text-slate-400 leading-relaxed px-2 max-w-lg mx-auto">
          {activeCard
            ? activeCard === 'adamw'
              ? 'Click ReduceLROnPlateau to see the adaptive scheduling strategy.'
              : 'Together these two work as a team: AdamW controls step direction, the scheduler controls step magnitude.'
            : 'Click either card to inspect its parameters. Both are created in __init__ so they persist across all 50 epochs.'}
        </p>
      </div>
    </div>
  );
};

// ── Step 2: Batch Training Loop ──────────────────────────────────────────────
const AnimatedBatchLoop = () => {
  const [phase, setPhase] = useState(0); // 0–5
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_PHASE = 5;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setPhase(p => {
        if (p >= MAX_PHASE) { setIsPlaying(false); return p; }
        return p + 1;
      }), 1600);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  const stages = [
    {
      label: 'zero_grad()',
      icon: <RefreshCw size={14} />,
      color: 'slate',
      desc: 'Clears all accumulated gradients from the previous batch step. Must happen first — otherwise gradients pile up across batches.',
    },
    {
      label: 'compute_loss(θ, c)',
      icon: <Activity size={14} />,
      color: 'blue',
      desc: 'Forward pass: feeds theta + condition through all 6 coupling layers. Returns the negative log-likelihood — how surprised the model is by this batch. No float16 autocast — RQ division would overflow.',
    },
    {
      label: 'loss.backward()',
      icon: <TrendingDown size={14} />,
      color: 'rose',
      desc: 'PyTorch automatic differentiation walks the computation graph backwards, computing ∂loss/∂param for every weight in every coupling layer.',
    },
    {
      label: 'clip_grad_norm_(1.0)',
      icon: <Shield size={14} />,
      color: 'amber',
      desc: 'If the global gradient norm exceeds 1.0, all gradients are scaled down proportionally. Tightened from 5.0 (affine flows) to 1.0 — spline knot extrapolation is extremely sensitive to large gradient steps.',
    },
    {
      label: 'optimizer.step()',
      icon: <Zap size={14} />,
      color: 'violet',
      desc: "AdamW applies the clipped gradient to every parameter with its per-parameter adaptive learning rate and decoupled weight decay. The model's weights have now been updated for this batch.",
    },
    {
      label: 'wandb.log (every 50)',
      icon: <BarChart2 size={14} />,
      color: 'emerald',
      desc: 'Batch loss is logged to W&B only every 50 global steps to avoid network overhead. global_step increments every batch regardless of whether logging fires.',
    },
  ];

  const colorMap = {
    slate:   { bg: 'bg-slate-700',   text: 'text-slate-200',   ring: 'ring-slate-500'   },
    blue:    { bg: 'bg-blue-700',    text: 'text-blue-100',    ring: 'ring-blue-500'    },
    rose:    { bg: 'bg-rose-700',    text: 'text-rose-100',    ring: 'ring-rose-500'    },
    amber:   { bg: 'bg-amber-700',   text: 'text-amber-100',   ring: 'ring-amber-500'   },
    violet:  { bg: 'bg-violet-700',  text: 'text-violet-100',  ring: 'ring-violet-500'  },
    emerald: { bg: 'bg-emerald-700', text: 'text-emerald-100', ring: 'ring-emerald-500' },
  };

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">Single Batch Training Step — 6 Substages</div>

        {/* Pipeline row */}
        <div className="w-full max-w-lg mx-auto flex flex-wrap gap-2 justify-center">
          {stages.map((s, i) => {
            const c = colorMap[s.color];
            const active = i === phase;
            const done = i < phase;
            return (
              <div
                key={i}
                onClick={() => setPhase(i)}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl font-mono text-[10px] font-bold cursor-pointer transition-all duration-300 border-2 ${
                  active
                    ? `${c.bg} ${c.text} ring-2 ${c.ring} ring-offset-1 ring-offset-slate-900 scale-105 shadow-lg border-transparent`
                    : done
                      ? 'bg-slate-700/50 text-slate-400 border-slate-600'
                      : 'bg-slate-800 text-slate-500 border-slate-700 hover:border-slate-600'
                }`}
              >
                {done ? <CheckCircle size={13} className="text-emerald-400" /> : s.icon}
                {s.label}
              </div>
            );
          })}
        </div>

        {/* Description card */}
        <div
          className="w-full max-w-lg mx-auto bg-slate-800 border border-slate-600 rounded-xl p-4 min-h-[80px] transition-all duration-300"
          key={phase}
        >
          <div className={`text-[10px] font-bold uppercase tracking-wider mb-2 text-${stages[phase].color}-400`}>
            Stage {phase + 1} / {stages.length}
          </div>
          <p className="text-[11px] text-slate-300 leading-relaxed animate-in fade-in duration-200">
            {stages[phase].desc}
          </p>
        </div>

        <p className="text-center text-[11px] text-slate-400 px-2 max-w-lg mx-auto">
          This 6-stage sequence runs for <strong className="text-slate-300">every single batch</strong> in every epoch. Click any stage to jump to it, or press Play to step through automatically.
        </p>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setPhase(p => Math.max(0, p - 1))} disabled={phase === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Play'}
        </VisualButton>
        <VisualButton onClick={() => setPhase(p => Math.min(MAX_PHASE, p + 1))} active disabled={phase === MAX_PHASE || isPlaying}>
          Next <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

// ── Step 3: Validation Phase ─────────────────────────────────────────────────
const AnimatedValidation = () => {
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [progress, setProgress] = useState(0); // 0–100

  const handleStart = () => {
    if (done) { setDone(false); setProgress(0); return; }
    setRunning(true);
  };

  useEffect(() => {
    let int;
    if (running) {
      int = setInterval(() => {
        setProgress(p => {
          if (p >= 100) { setRunning(false); setDone(true); return 100; }
          return p + 4;
        });
      }, 80);
    }
    return () => clearInterval(int);
  }, [running]);

  const valLoss = (0.38 + 0.05 * Math.sin(progress / 10)).toFixed(4);

  const stages = [
    { label: 'model.eval()',         pct: 5,  color: 'sky',     desc: 'Disables dropout & batch-norm training behaviour' },
    { label: 'torch.no_grad()',      pct: 20, color: 'violet',  desc: 'Turns off autograd — no computation graph built, ~2× memory savings' },
    { label: 'compute_loss(θ, c)',   pct: 45, color: 'blue',    desc: 'Same NLL forward pass as training, but gradients are never computed' },
    { label: 'Accumulate val_loss',  pct: 70, color: 'amber',   desc: 'Sums batch losses across all val batches' },
    { label: 'Return avg_val_loss',  pct: 95, color: 'emerald', desc: 'Divided by len(val_dataloader) — returned to train() for scheduler & checkpointing' },
  ];

  const activeStage = stages.findIndex((s, i) => progress < s.pct) - 1;
  const currentStageIdx = Math.max(0, activeStage === -1 ? stages.length - 1 : activeStage);

  const colorMap = {
    sky:    'text-sky-400',
    violet: 'text-violet-400',
    blue:   'text-blue-400',
    amber:  'text-amber-400',
    emerald:'text-emerald-400',
  };

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">evaluate() — Validation Without Gradient Updates</div>

        {/* Difference callout */}
        <div className="w-full max-w-lg mx-auto flex gap-3">
          <div className="flex-1 bg-rose-900/20 border border-rose-500/40 rounded-xl p-3 text-center">
            <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Training Loop</div>
            <div className="font-mono text-[11px] text-rose-300 font-bold">loss.backward() ✓</div>
            <div className="font-mono text-[11px] text-rose-300 font-bold">optimizer.step() ✓</div>
          </div>
          <div className="flex items-center text-slate-600 font-bold text-xl">≠</div>
          <div className="flex-1 bg-sky-900/20 border border-sky-500/40 rounded-xl p-3 text-center">
            <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Validation Loop</div>
            <div className="font-mono text-[11px] text-sky-300 font-bold">no_grad() ✓</div>
            <div className="font-mono text-[11px] text-slate-500 line-through">optimizer.step() ✗</div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="w-full max-w-lg mx-auto">
          <div className="flex justify-between text-[9px] text-slate-500 mb-1 font-mono">
            <span>Validating batches…</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-sky-500 to-violet-500 rounded-full transition-all duration-150"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Stage list */}
        <div className="w-full max-w-lg mx-auto flex flex-col gap-1.5">
          {stages.map((s, i) => {
            const isPast = progress >= s.pct;
            const isCurrent = i === currentStageIdx && running;
            return (
              <div key={i} className={`flex items-start gap-3 px-3 py-2 rounded-lg transition-all duration-300 ${
                isPast ? 'bg-slate-800 opacity-100' : 'bg-slate-800/30 opacity-40'
              } ${isCurrent ? 'ring-1 ring-violet-500/60' : ''}`}>
                <CheckCircle size={13} className={`mt-0.5 flex-shrink-0 ${isPast ? 'text-emerald-400' : 'text-slate-700'}`} />
                <span className={`font-mono text-[10px] font-bold flex-shrink-0 w-36 ${isPast ? colorMap[s.color] : 'text-slate-600'}`}>{s.label}</span>
                <span className="text-[10px] text-slate-400 leading-tight">{s.desc}</span>
              </div>
            );
          })}
        </div>

        {done && (
          <div className="w-full max-w-lg mx-auto bg-emerald-900/20 border border-emerald-500/50 rounded-xl p-3 flex items-center gap-3 animate-in fade-in duration-300">
            <CheckCircle size={18} className="text-emerald-400 flex-shrink-0" />
            <div>
              <span className="text-[11px] font-bold text-emerald-300">avg_val_loss = {valLoss}</span>
              <p className="text-[9px] text-slate-400 mt-0.5">Returned to train() → scheduler.step() + checkpoint decision</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={handleStart} active={!done} disabled={running}>
          {done ? <><RefreshCw size={14} /> Reset</> : <><Play size={14} /> Run Validation</>}
        </VisualButton>
      </div>
    </div>
  );
};

// ── Step 4: LR Scheduler ─────────────────────────────────────────────────────
const AnimatedScheduler = () => {
  const [epoch, setEpoch] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_EPOCH = 30;

  // Simulate a jagged val_loss with a plateau then improvement
  const getValLoss = (e) => {
    if (e === 0) return null;
    const base = 1.8 * Math.exp(-0.04 * e) + 0.35;
    // Plateau between epochs 6–9
    const plateau = (e >= 6 && e <= 9) ? 0.15 : 0;
    // Another plateau around epoch 18–21
    const plateau2 = (e >= 18 && e <= 21) ? 0.1 : 0;
    return +(base + plateau + plateau2 + (Math.random() * 0.03 - 0.015)).toFixed(4);
  };

  const [history, setHistory] = useState([]); // [{e, loss, lr, fired}]
  let lr = useRef(5e-4);

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => {
        setEpoch(e => {
          if (e >= MAX_EPOCH) { setIsPlaying(false); return e; }
          const next = e + 1;
          const loss = getValLoss(next);

          // Simulate patience logic
          setHistory(prev => {
            const recent = prev.slice(-3);
            const stalled = recent.length === 3 && recent.every(r => r.loss >= (prev[prev.length - 4]?.loss ?? Infinity));
            const fired = stalled;
            if (fired) lr.current = Math.max(lr.current * 0.5, 1e-6);
            return [...prev, { e: next, loss, lr: lr.current, fired }];
          });
          return next;
        });
      }, 250);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  const handleReset = () => { setEpoch(0); setHistory([]); lr.current = 5e-4; setIsPlaying(false); };

  const chartH = 80;
  const chartW = 300;
  const lrChartH = 40;

  const lossPoints = history.map((h, i) => {
    const x = (i / (MAX_EPOCH - 1)) * chartW;
    const y = chartH - Math.min((h.loss / 2.2) * chartH, chartH);
    return `${x},${y}`;
  }).join(' ');

  const lrPoints = history.map((h, i) => {
    const x = (i / (MAX_EPOCH - 1)) * chartW;
    const y = lrChartH - (Math.log10(h.lr / 1e-6) / Math.log10(5e-4 / 1e-6)) * lrChartH;
    return `${x},${y}`;
  }).join(' ');

  const fireEpochs = history.filter(h => h.fired).map(h => h.e);
  const currentLr = history.length > 0 ? history[history.length - 1].lr : 5e-4;

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">
          ReduceLROnPlateau — Adaptive LR Halving Simulation
        </div>

        {/* Stats row */}
        <div className="flex justify-center gap-3 w-full max-w-md mx-auto">
          <div className="flex flex-col items-center bg-slate-800 border border-slate-700 rounded-xl px-4 py-2">
            <span className="text-[9px] text-slate-500 uppercase mb-0.5">Epoch</span>
            <span className="text-2xl font-extrabold text-white font-mono">{String(epoch).padStart(2,'0')}</span>
          </div>
          <div className="flex flex-col items-center bg-slate-800 border border-slate-700 rounded-xl px-4 py-2">
            <span className="text-[9px] text-slate-500 uppercase mb-0.5">Val Loss</span>
            <span className="text-2xl font-extrabold text-sky-400 font-mono">
              {history.length > 0 ? history[history.length-1].loss : '—'}
            </span>
          </div>
          <div className="flex flex-col items-center bg-slate-800 border border-slate-700 rounded-xl px-4 py-2">
            <span className="text-[9px] text-slate-500 uppercase mb-0.5">LR</span>
            <span className={`text-lg font-extrabold font-mono ${currentLr < 5e-4 ? 'text-amber-400' : 'text-emerald-400'}`}>
              {currentLr.toExponential(0)}
            </span>
          </div>
        </div>

        {/* Loss chart */}
        <div className="w-full max-w-md mx-auto bg-slate-800/60 border border-slate-700 rounded-xl p-3">
          <div className="text-[9px] text-slate-500 font-bold uppercase tracking-wider mb-1">Val Loss</div>
          <div className="overflow-hidden rounded">
            <svg width="100%" viewBox={`-2 -4 ${chartW + 4} ${chartH + 6}`}>
              {[0.25, 0.5, 0.75, 1].map(f => (
                <line key={f} x1="0" y1={chartH * (1-f)} x2={chartW} y2={chartH * (1-f)} stroke="#334155" strokeWidth="0.5" strokeDasharray="3" />
              ))}
              {history.length > 1 && <polyline points={lossPoints} fill="none" stroke="#38bdf8" strokeWidth="2" />}
              {fireEpochs.map(fe => {
                const x = ((fe - 1) / (MAX_EPOCH - 1)) * chartW;
                return <line key={fe} x1={x} y1={0} x2={x} y2={chartH} stroke="#f59e0b" strokeWidth="1" strokeDasharray="4" />;
              })}
            </svg>
          </div>
          {fireEpochs.length > 0 && (
            <div className="flex items-center gap-2 mt-1">
              <div className="w-4 h-px bg-amber-500 border-dashed border-t border-amber-500"></div>
              <span className="text-[9px] text-amber-400">LR halved at epochs: {fireEpochs.join(', ')}</span>
            </div>
          )}
        </div>

        {/* LR chart */}
        <div className="w-full max-w-md mx-auto bg-slate-800/60 border border-slate-700 rounded-xl p-3">
          <div className="text-[9px] text-slate-500 font-bold uppercase tracking-wider mb-1">Learning Rate</div>
          <div className="overflow-hidden rounded">
            <svg width="100%" viewBox={`-2 -4 ${chartW + 4} ${lrChartH + 6}`}>
              {history.length > 1 && <polyline points={lrPoints} fill="none" stroke="#f59e0b" strokeWidth="2" />}
            </svg>
          </div>
        </div>

        <p className="text-center text-[11px] text-slate-400 px-2 max-w-md mx-auto">
          {epoch === 0 && 'Spline flows often stall for several epochs before improving. ReduceLROnPlateau watches val_loss and halves the LR after 3 stuck epochs.'}
          {epoch > 0 && epoch < MAX_EPOCH && `Epoch ${epoch}/30 — patience=3 means the scheduler waits 3 full epochs before deciding the model is truly stuck.`}
          {epoch >= MAX_EPOCH && `Simulation complete. Each amber dashed line marks a LR halving event. The LR floor is 1e-6 — it never goes below that.`}
        </p>
      </div>

      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={handleReset} disabled={isPlaying}>
          <RefreshCw size={14} /> Reset
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(v => !v)} active={isPlaying} disabled={epoch >= MAX_EPOCH}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? 'Pause' : 'Simulate'}
        </VisualButton>
      </div>
    </div>
  );
};

// ── Step 5: Checkpoint Saving ─────────────────────────────────────────────────
const AnimatedCheckpoint = () => {
  const [scenario, setScenario] = useState(null); // 'best' | 'periodic' | 'neither'

  const scenarios = [
    {
      id: 'best',
      label: 'New Best Model',
      icon: '🌟',
      color: 'emerald',
      condition: 'avg_val_loss < self.best_val_loss',
      filename: 'model_best.pt',
      contents: [
        { k: 'epoch', v: 'current epoch number' },
        { k: 'model_state_dict', v: 'all layer weights & biases' },
        { k: 'optimizer_state_dict', v: 'AdamW momentum buffers' },
        { k: 'train_loss', v: 'avg training NLL this epoch' },
        { k: 'val_loss', v: 'the new best val NLL' },
      ],
      why: 'model_best.pt always holds the single best generalising checkpoint ever seen. If training diverges later, you can reload this and resume.',
    },
    {
      id: 'periodic',
      label: 'Periodic Save (epoch % 10)',
      icon: '💾',
      color: 'sky',
      condition: 'epoch % 10 == 0',
      filename: 'model_epoch_N.pt',
      contents: [
        { k: 'epoch', v: 'e.g. 10, 20, 30…' },
        { k: 'model_state_dict', v: 'snapshot at this epoch' },
        { k: 'optimizer_state_dict', v: 'AdamW state for resuming' },
        { k: 'train_loss', v: 'avg training NLL' },
        { k: 'val_loss', v: 'val NLL at this snapshot' },
      ],
      why: 'Periodic saves are safety nets. Even if no new best is hit for 30 epochs, you keep every-10-epoch snapshots to inspect or fine-tune from.',
    },
    {
      id: 'neither',
      label: 'No Save This Epoch',
      icon: '⏭',
      color: 'slate',
      condition: 'else (skip)',
      filename: '—',
      contents: [],
      why: "If this epoch's val_loss is not a new best and the epoch number isn't divisible by 10, nothing is written to disk. This is intentional — it keeps the checkpoint directory clean.",
    },
  ];

  const colorMap = {
    emerald: { bg: 'bg-emerald-900/30', border: 'border-emerald-500/60', text: 'text-emerald-300' },
    sky:     { bg: 'bg-sky-900/30',     border: 'border-sky-500/60',     text: 'text-sky-300'     },
    slate:   { bg: 'bg-slate-800',      border: 'border-slate-600',      text: 'text-slate-400'   },
  };

  const current = scenarios.find(s => s.id === scenario);

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">save_checkpoint() — Three Possible Outcomes</div>

        {/* 3 scenario buttons */}
        <div className="flex gap-2 w-full max-w-lg mx-auto">
          {scenarios.map(s => {
            const c = colorMap[s.color];
            const active = scenario === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setScenario(active ? null : s.id)}
                className={`flex-1 flex flex-col items-center gap-1.5 py-3 px-2 rounded-xl border-2 transition-all duration-200 cursor-pointer ${
                  active ? `${c.bg} ${c.border} shadow-lg` : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                }`}
              >
                <span className="text-2xl">{s.icon}</span>
                <span className={`text-[9px] font-bold font-mono text-center leading-tight ${active ? c.text : 'text-slate-400'}`}>{s.label}</span>
              </button>
            );
          })}
        </div>

        {/* Detail card */}
        {current && (
          <div className="w-full max-w-lg mx-auto animate-in fade-in slide-in-from-bottom-2 duration-200">
            <div className={`rounded-xl border-2 p-4 flex flex-col gap-3 ${colorMap[current.color].bg} ${colorMap[current.color].border}`}>
              {/* Condition */}
              <div className="flex items-center gap-2">
                <span className="text-[9px] text-slate-500 uppercase tracking-wider">Condition</span>
                <code className="text-[10px] font-mono bg-slate-900/60 px-2 py-0.5 rounded text-slate-300">{current.condition}</code>
              </div>

              {/* Filename */}
              {current.filename !== '—' && (
                <div className="flex items-center gap-2">
                  <Save size={12} className={colorMap[current.color].text} />
                  <span className="font-mono text-[11px] font-bold text-slate-200">outputs/checkpoints/{current.filename}</span>
                </div>
              )}

              {/* Contents */}
              {current.contents.length > 0 && (
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] text-slate-500 uppercase tracking-wider">torch.save() contents</span>
                  {current.contents.map(c => (
                    <div key={c.k} className="flex gap-3 text-[10px]">
                      <span className="font-mono text-slate-300 w-36 flex-shrink-0">{c.k}</span>
                      <span className="text-slate-500">{c.v}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Why */}
              <p className="text-[10px] text-slate-300 leading-relaxed italic border-l-2 border-violet-500/40 pl-3">
                {current.why}
              </p>
            </div>
          </div>
        )}

        {!current && (
          <p className="text-center text-[11px] text-slate-400 px-2 max-w-lg mx-auto">
            At the end of every epoch, the trainer evaluates one of these three outcomes. Click each card to inspect what gets written to disk and why.
          </p>
        )}
      </div>
    </div>
  );
};

// ── Full Code Walkthrough ─────────────────────────────────────────────────────
const AnimatedWalkthrough = () => {
  const [activePart, setActivePart] = useState(0);
  const lineRefs = useRef({});

  const codeLines = [
    { text: "class SMPCTrainer:", part: null },
    { text: "  def __init__(self, model, train_dataloader, val_dataloader,", part: 0 },
    { text: "               learning_rate=5e-4, epochs=50, device='mps', log_to_wandb=False):", part: 0 },
    { text: "    self.model = model.to(device)", part: 0 },
    { text: "    self.optimizer = AdamW(self.model.parameters(),", part: 0 },
    { text: "                          lr=learning_rate, weight_decay=1e-4)", part: 0 },
    { text: "    self.scheduler = ReduceLROnPlateau(", part: 0 },
    { text: "        self.optimizer, mode='min', factor=0.5,", part: 0 },
    { text: "        patience=3, min_lr=1e-6)", part: 0 },
    { text: "    self.best_val_loss = float('inf')", part: 0 },
    { text: "", part: null },
    { text: "  def train(self):", part: null },
    { text: "    wandb.watch(self.model, log='all', log_freq=50)", part: 1 },
    { text: "    global_step = 0", part: 1 },
    { text: "", part: null },
    { text: "    for epoch in range(1, self.epochs + 1):", part: 1 },
    { text: "      self.model.train()", part: 1 },
    { text: "      for batch in pbar_train:", part: 1 },
    { text: "        theta     = batch['theta'].to(self.device)", part: 1 },
    { text: "        condition = batch['condition'].to(self.device)", part: 1 },
    { text: "", part: null },
    { text: "        self.optimizer.zero_grad()", part: 1 },
    { text: "        loss = self.model.compute_loss(theta, condition)  # No float16!", part: 1 },
    { text: "        loss.backward()", part: 1 },
    { text: "        clip_grad_norm_(self.model.parameters(), max_norm=1.0)", part: 1 },
    { text: "        self.optimizer.step()", part: 1 },
    { text: "", part: null },
    { text: "        if self.log_to_wandb and global_step % 50 == 0:", part: 1 },
    { text: "          wandb.log({'train/batch_loss': loss.item()})", part: 1 },
    { text: "        global_step += 1", part: 1 },
    { text: "", part: null },
    { text: "      avg_val_loss = self.evaluate(epoch, avg_train_loss)", part: 2 },
    { text: "", part: null },
    { text: "      self.scheduler.step(avg_val_loss)  # ← may halve LR", part: 3 },
    { text: "      current_lr = self.optimizer.param_groups[0]['lr']", part: 3 },
    { text: "", part: null },
    { text: "      if avg_val_loss < self.best_val_loss:", part: 4 },
    { text: "        self.save_checkpoint(..., is_best=True)  # model_best.pt", part: 4 },
    { text: "      elif epoch % 10 == 0:", part: 4 },
    { text: "        self.save_checkpoint(...)               # model_epoch_N.pt", part: 4 },
    { text: "", part: null },
    { text: "  def evaluate(self, epoch, avg_train_loss):", part: 2 },
    { text: "    self.model.eval()", part: 2 },
    { text: "    with torch.no_grad():", part: 2 },
    { text: "      for batch in pbar_val:", part: 2 },
    { text: "        loss = self.model.compute_loss(theta, condition)", part: 2 },
    { text: "    return avg_val_loss", part: 2 },
    { text: "", part: null },
    { text: "  def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):", part: 4 },
    { text: "    filename = 'model_best.pt' if is_best else f'model_epoch_{epoch}.pt'", part: 4 },
    { text: "    torch.save({'epoch': epoch, 'model_state_dict': ...,", part: 4 },
    { text: "                'optimizer_state_dict': ..., 'val_loss': val_loss}, path)", part: 4 },
  ];

  const parts = [
    {
      title: '__init__ — Wiring Everything Up',
      exp: 'The constructor receives all the assembled pieces from train.py and binds them to self. The model is immediately moved to the target device. AdamW and ReduceLROnPlateau are created here so they persist across all 50 epochs and can accumulate momentum and patience state.',
    },
    {
      title: 'train() — The Batch Loop',
      exp: 'For every epoch, for every batch: zero_grad → compute NLL loss (no float16 — spline division overflows) → backward → clip gradients to 1.0 → AdamW step. Batch loss is logged to W&B every 50 global steps to avoid network overhead.',
    },
    {
      title: 'evaluate() — Validation',
      exp: 'Called once per epoch after all training batches. model.eval() + torch.no_grad() means no computation graph is built and dropout is disabled. The same compute_loss is called, but gradients are never computed — ~2× memory savings on MPS.',
    },
    {
      title: 'scheduler.step() — Adaptive LR',
      exp: 'ReduceLROnPlateau checks if avg_val_loss improved. If not for 3 consecutive epochs (patience=3), the learning rate is multiplied by 0.5. This repeats until the floor of 1e-6 is reached. Splines need this because their loss curves are naturally jagged.',
    },
    {
      title: 'save_checkpoint() — Persistence',
      exp: 'Two conditions trigger a save: (1) a new best val_loss → overwrites model_best.pt, (2) epoch divisible by 10 → writes model_epoch_N.pt. Both save the full model state, optimizer state, and both losses so training can be resumed or rolled back.',
    },
  ];

  useEffect(() => {
    if (lineRefs.current[activePart]) {
      lineRefs.current[activePart].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activePart]);

  return (
    <div className="relative w-full h-[600px] md:h-full bg-[#0d1117] overflow-hidden flex flex-col md:flex-row">
      <div className="flex-1 overflow-auto py-4 px-4 font-mono text-[10px] sm:text-xs text-slate-300 code-scroll">
        <style dangerouslySetInnerHTML={{__html:`
          .code-scroll::-webkit-scrollbar{width:6px;height:6px;}
          .code-scroll::-webkit-scrollbar-track{background:transparent;}
          .code-scroll::-webkit-scrollbar-thumb{background:#334155;border-radius:4px;}
        `}}/>
        {codeLines.map((line, i) => (
          <div
            key={i}
            ref={el => { if (line.part !== null && !lineRefs.current[line.part]) lineRefs.current[line.part] = el; }}
            className={`px-2 py-0.5 border-l-[3px] transition-all duration-300 whitespace-pre-wrap break-words ${
              line.part === activePart
                ? 'bg-violet-500/15 border-violet-500 text-violet-100'
                : line.part !== null
                  ? 'border-transparent opacity-40 hover:opacity-80 cursor-pointer'
                  : 'border-transparent text-slate-600'
            }`}
            onClick={() => { if (line.part !== null) setActivePart(line.part); }}
            dangerouslySetInnerHTML={{ __html: highlightCode(line.text) || ' ' }}
          />
        ))}
      </div>

      <div className="w-full min-h-[180px] md:h-full md:w-80 lg:w-96 flex-shrink-0 bg-slate-800 md:border-l border-t md:border-t-0 border-slate-700 p-4 md:p-5 flex flex-col justify-between z-10 overflow-y-auto">
        <div className="animate-in fade-in slide-in-from-right-2 duration-300" key={activePart}>
          <div className="text-[10px] font-bold text-violet-400 mb-1.5 uppercase tracking-wider">
            Part {activePart + 1} of {parts.length}
          </div>
          <h4 className="text-sm sm:text-base font-bold text-white mb-3">{parts[activePart].title}</h4>
          <p className="text-[11px] sm:text-xs text-slate-300 leading-relaxed bg-slate-900/50 p-3 sm:p-4 rounded-lg border border-slate-700 shadow-inner">
            {parts[activePart].exp}
          </p>
        </div>
        <div className="flex gap-2 justify-between mt-4">
          <VisualButton onClick={() => setActivePart(p => Math.max(0, p - 1))} disabled={activePart === 0}>
            <ChevronLeft size={16} /> Prev
          </VisualButton>
          <VisualButton onClick={() => setActivePart(p => Math.min(parts.length - 1, p + 1))} disabled={activePart === parts.length - 1} active>
            Next <ChevronRight size={16} />
          </VisualButton>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// STEP DEFINITIONS
// ==========================================

const steps = [
  {
    id: 'init',
    title: '1. Optimizer & Scheduler',
    icon: Zap,
    codeSnippet: `self.model = model.to(device)\n\n# AdamW: weight_decay lowered to 1e-4 for splines\n# (too high breaks knot positioning)\nself.optimizer = AdamW(\n    self.model.parameters(),\n    lr=learning_rate,\n    weight_decay=1e-4\n)\n\n# Halves LR when val_loss stalls for 3 epochs\nself.scheduler = ReduceLROnPlateau(\n    self.optimizer,\n    mode='min',\n    factor=0.5,\n    patience=3,\n    min_lr=1e-6\n)\n\nself.best_val_loss = float('inf')`,
    description: "__init__ wires the model, both data loaders, and all training state into self. AdamW uses weight_decay=1e-4 — much lower than the PyTorch default of 1e-2 — because excessive weight decay destabilises the rational-quadratic spline knot positions. ReduceLROnPlateau is set up immediately so it can accumulate patience state across epochs from the very first one.",
    why: "Both optimiser and scheduler need to persist across all 50 epochs to accumulate momentum buffers and patience state. Creating them in __init__ (rather than inside train()) guarantees they are never accidentally reset mid-training.",
    Visual: AnimatedOptimizer,
  },
  {
    id: 'batchloop',
    title: '2. Batch Training Loop',
    icon: Activity,
    codeSnippet: `self.optimizer.zero_grad()\n\n# No torch.autocast float16 — RQ division\n# overflows/underflows in float16 → NaN loss\nloss = self.model.compute_loss(theta, condition)\n\nloss.backward()\n\n# Tightened to 1.0 (was 5.0 for affine flows)\n# Spline knot extrapolation is very grad-sensitive\ntorch.nn.utils.clip_grad_norm_(\n    self.model.parameters(), max_norm=1.0\n)\n\nself.optimizer.step()\n\n# Log every 50 steps to avoid network overhead\nif self.log_to_wandb and global_step % 50 == 0:\n    wandb.log({'train/batch_loss': loss.item()})`,
    description: "Six substages run for every single batch: clear old gradients → forward pass NLL loss (float32 only — no autocast) → backprop → gradient clipping → AdamW weight update → conditional W&B logging every 50 global steps.",
    why: "float16 autocast is explicitly avoided: the rational-quadratic spline formula involves divisions between very small numbers (bin widths, derivatives) that easily underflow to zero in float16, causing NaN loss. The grad clip is tightened from 5.0 (affine flows) to 1.0 because spline knots are far more sensitive to large gradient steps than simple shift-scale couplings.",
    Visual: AnimatedBatchLoop,
  },
  {
    id: 'validation',
    title: '3. Validation Phase',
    icon: Shield,
    codeSnippet: `def evaluate(self, epoch, avg_train_loss):\n    self.model.eval()       # disable dropout / batchnorm\n\n    with torch.no_grad():   # no computation graph\n        for batch in pbar_val:\n            theta     = batch['theta'].to(self.device)\n            condition = batch['condition'].to(self.device)\n\n            loss = self.model.compute_loss(theta, condition)\n            epoch_val_loss += loss.item()\n\n    avg_val_loss = epoch_val_loss / len(self.val_dataloader)\n    return avg_val_loss`,
    description: "evaluate() is called once per epoch, after all training batches complete. model.eval() disables any training-only behaviour. torch.no_grad() prevents PyTorch from building a computation graph, which saves ~2× memory on MPS unified memory and runs faster.",
    why: "The same compute_loss is called as in training — but because no gradients are computed, the model weights are never updated. This gives an unbiased estimate of generalisation: how well does the model fit unseen valve timing data it has never trained on?",
    Visual: AnimatedValidation,
  },
  {
    id: 'scheduler',
    title: '4. LR Scheduling',
    icon: TrendingDown,
    codeSnippet: `# Called once per epoch, after evaluate()\nself.scheduler.step(avg_val_loss)\ncurrent_lr = self.optimizer.param_groups[0]['lr']\n\n# What fires internally:\n# if epochs_no_improve >= patience (3):\n#     new_lr = max(current_lr * factor (0.5),\n#                  min_lr (1e-6))\n#     optimizer.lr = new_lr`,
    description: "scheduler.step(avg_val_loss) is called once per epoch. If avg_val_loss has not improved for 3 consecutive epochs (patience=3), the learning rate is multiplied by 0.5. This repeats until the hard floor of 1e-6 is hit.",
    why: "Spline flows have naturally jagged training curves — they can plateau for 2-3 epochs then suddenly drop. patience=3 gives the model room to breathe before concluding it is stuck. Without this, the LR would decay too aggressively and permanently undershoot the loss minimum.",
    Visual: AnimatedScheduler,
  },
  {
    id: 'checkpoint',
    title: '5. Checkpoint Saving',
    icon: Save,
    codeSnippet: `if avg_val_loss < self.best_val_loss:\n    self.best_val_loss = avg_val_loss\n    self.save_checkpoint(..., is_best=True)\n    # → outputs/checkpoints/model_best.pt\n\nelif epoch % 10 == 0:\n    self.save_checkpoint(...)\n    # → outputs/checkpoints/model_epoch_N.pt\n\ndef save_checkpoint(self, epoch, train_loss, val_loss,\n                   is_best=False):\n    torch.save({\n        'epoch': epoch,\n        'model_state_dict': self.model.state_dict(),\n        'optimizer_state_dict': self.optimizer.state_dict(),\n        'train_loss': train_loss,\n        'val_loss': val_loss,\n    }, path)`,
    description: "At the end of each epoch, two independent conditions are checked. A new best val_loss overwrites model_best.pt. An epoch number divisible by 10 writes a dated snapshot model_epoch_N.pt. Both save the full model + optimizer state so training can be resumed or rolled back.",
    why: "Saving optimizer_state_dict alongside model_state_dict is critical for resuming: AdamW's per-parameter momentum buffers take several epochs to warm up. Restarting from just model weights would cause several wasted epochs of unstable updates while momentum rebuilds.",
    Visual: AnimatedCheckpoint,
  },
  {
    id: 'walkthrough',
    title: '6. Full Code Walkthrough',
    icon: Terminal,
    codeSnippet: '',
    description: 'Complete line-by-line breakdown of trainer.py.',
    why: '',
    Visual: AnimatedWalkthrough,
  },
];

// ==========================================
// EXPORTED SECTION COMPONENT
// ==========================================

export default function TrainerSection() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">

      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
        <div className="p-3 bg-violet-100 text-violet-600 rounded-xl shadow-sm border border-violet-200">
          <Cpu className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800 tracking-tight">SMPCTrainer</h2>
          <p className="text-sm text-slate-500 font-medium">
            Interactive walkthrough of{' '}
            <code className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-700 border border-slate-200">
              src/training/trainer.py
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
                ? 'bg-violet-500 scale-y-110 shadow-sm'
                : idx < currentStep
                ? 'bg-violet-300'
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
              <step.icon size={22} className="text-violet-400" />
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
          </div>
          <div className="flex-1 w-full rounded-2xl shadow-xl overflow-hidden relative border-4 border-slate-900/5 bg-[#0d1117]">
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
                <div className="p-2.5 bg-violet-100 rounded-xl text-violet-600 shadow-sm border border-violet-200">
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
                  <Terminal size={14} className="text-violet-400" />
                  <span className="text-xs font-bold font-mono text-slate-300">trainer.py snippet</span>
                </div>
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-rose-500/80"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80"></div>
                </div>
              </div>
              <div className="p-5 overflow-auto flex-1 text-[13px] font-mono leading-loose text-slate-300 custom-scrollbar">
                <style dangerouslySetInnerHTML={{__html:`
                  .custom-scrollbar::-webkit-scrollbar{width:6px;height:6px;}
                  .custom-scrollbar::-webkit-scrollbar-track{background:transparent;}
                  .custom-scrollbar::-webkit-scrollbar-thumb{background:#334155;border-radius:4px;}
                `}}/>
                <pre><code dangerouslySetInnerHTML={{ __html: highlightCode(step.codeSnippet) }} /></pre>
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
          <ChevronLeft size={18} /> Previous Step
        </button>
        <button
          onClick={() => setCurrentStep(p => Math.min(steps.length - 1, p + 1))}
          disabled={currentStep === steps.length - 1}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-white bg-violet-500 hover:bg-violet-600 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-40 disabled:hover:translate-y-0 disabled:cursor-not-allowed disabled:shadow-none"
        >
          {currentStep === steps.length - 1 ? 'Finish Walkthrough' : 'Next Step'} <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}
