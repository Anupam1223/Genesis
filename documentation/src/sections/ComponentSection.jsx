import React, { useState, useEffect, useRef } from 'react';
import { 
  Layers, 
  ChevronRight, 
  ChevronLeft, 
  Activity,
  Cpu,
  Waves,
  GitMerge,
  Split,
  Terminal,
  MousePointer2,
  CheckCircle,
  Play,
  Pause,
  RefreshCw,
  Search,
  BoxSelect,
  ArrowDown
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

  // 1. Comments
  html = html.replace(/(#.*)/g, (m) => pushToken(m, "text-slate-500 italic"));
  
  // 2. Strings
  html = html.replace(/('.*?'|".*?")/g, (m) => pushToken(m, "text-emerald-300"));

  // 3. Numbers
  html = html.replace(/\b(\d+\.\d+|\d+)\b/g, (m) => pushToken(m, "text-purple-300"));

  // 4. PyTorch functions & modules
  const pytorchRegex = /\b(nn\.Module|nn\.Linear|nn\.ModuleList|nn\.Sequential|nn\.GELU|nn\.Dropout|nn\.init\.zeros_|F\.softmax|F\.softplus|F\.pad|torch\.cumsum|torch\.searchsorted|torch\.gather|torch\.clone|torch\.cat|torch\.zeros_like|torch\.ones_like|torch\.log|torch\.clamp|torch\.sqrt)\b/g;
  html = html.replace(pytorchRegex, (m) => pushToken(m, "text-amber-300"));

  // 5. Python Keywords
  const kwRegex = /\b(def|class|if|else|elif|for|return|import|from|as|not|in)\b/g;
  html = html.replace(kwRegex, (m) => pushToken(m, "text-rose-400 font-bold"));

  // 6. Self & special params
  const paramRegex = /\b(self|inputs|unnorm_widths|unnorm_heights|unnorm_derivs|inverse|bound)\b/g;
  html = html.replace(paramRegex, (m) => pushToken(m, "text-blue-300 italic"));

  // Restore tokens backwards to avoid any accidental nesting replacements
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
        ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/50' 
        : 'bg-slate-700 hover:bg-slate-600 text-slate-200 shadow-slate-900/50 border border-slate-600'
    }`}
  >
    {children}
  </button>
);

// ==========================================
// INTERACTIVE VISUAL COMPONENTS
// ==========================================

const AnimatedResidualMLP = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(s => (s + 1) % 3), 2500);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  return (
    <div className="relative w-full h-full bg-slate-900 flex flex-col items-center justify-center p-6 pb-20">
      <div className="text-xs text-slate-400 mb-8 font-mono">ResidualMLP: The Brain Architecture</div>

      <div className="flex w-full max-w-md items-center justify-between">
        
        {/* Input */}
        <div className={`flex flex-col items-center transition-all duration-500 ${step >= 0 ? 'opacity-100' : 'opacity-30'}`}>
          <div className="bg-blue-900/50 border border-blue-500 text-blue-300 px-4 py-2 rounded-lg text-xs font-bold shadow-[0_0_15px_rgba(59,130,246,0.2)]">Condition (x)</div>
        </div>

        {/* Residual Blocks */}
        <div className="flex flex-col items-center relative flex-1 mx-6">
          <div className="absolute top-1/2 left-0 right-0 border-t-2 border-dashed border-indigo-500/50 -z-10 -translate-y-1/2"></div>
          
          {/* Skip Connection Arc */}
          <svg className={`absolute inset-0 w-full h-24 -top-12 -z-10 transition-all duration-500 ${step >= 1 ? 'opacity-100' : 'opacity-0'}`} viewBox="0 0 100 50" preserveAspectRatio="none">
             <path d="M 10 50 C 30 -10, 70 -10, 90 50" fill="none" stroke="#60a5fa" strokeWidth="2" strokeDasharray="4" className="animate-pulse" />
             <text x="50" y="15" fill="#60a5fa" fontSize="8" textAnchor="middle" fontWeight="bold">+ out (Skip Connection)</text>
          </svg>

          <div className={`bg-indigo-900/40 border border-indigo-500 p-4 rounded-xl shadow-lg flex gap-2 transition-all duration-500 ${step >= 1 ? 'scale-110 shadow-[0_0_20px_rgba(99,102,241,0.3)]' : 'scale-100'}`}>
            <div className="w-4 h-12 bg-indigo-500/50 rounded-sm"></div>
            <div className="w-4 h-12 bg-indigo-500/50 rounded-sm"></div>
            <div className="w-4 h-12 bg-indigo-500/50 rounded-sm"></div>
          </div>
        </div>

        {/* Output (Zero-Init) */}
        <div className={`flex flex-col items-center transition-all duration-500 ${step >= 2 ? 'opacity-100 scale-110' : 'opacity-30 scale-100'}`}>
          <div className={`px-3 py-2 rounded-lg text-xs font-bold font-mono shadow-md border ${step >= 2 ? 'bg-emerald-900/50 border-emerald-500 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.3)]' : 'bg-slate-800 border-slate-600 text-slate-500'}`}>
            {step >= 2 ? '[0.0, 0.0, 0.0]' : '[?, ?, ?]'}
          </div>
          <div className={`text-[8px] mt-2 font-bold uppercase tracking-wider ${step >= 2 ? 'text-emerald-400' : 'text-slate-500'}`}>Zero-Initialized Layer</div>
        </div>

      </div>

      <div className="text-center text-[11px] text-slate-400 mt-12 max-w-[85%] leading-relaxed h-12">
        {step === 0 && "1. The Condition array enters the neural network."}
        {step === 1 && "2. Residual blocks add the input back to the output at each layer. This prevents 'vanishing gradients' in deep networks, stabilizing the physics training."}
        {step === 2 && "3. CRITICAL: The final layer weights/biases are forced to exactly 0.0. This ensures the Spline starts completely flat (an Identity function) before training begins."}
      </div>

      {/* Playback Controls */}
      <div className="absolute bottom-4 flex gap-4">
        <VisualButton onClick={() => setStep(s => Math.max(0, s - 1))} active={false} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev Step
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? "Pause" : "Play Sequence"}
        </VisualButton>
        <VisualButton onClick={() => setStep(s => Math.min(2, s + 1))} active={true} disabled={step === 2 || isPlaying}>
          Next Step <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedZeroInit = () => {
  const [mode, setMode] = useState('random'); // 'random' | 'zero'
  const [epoch, setEpoch] = useState(0);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let int;
    if (running) {
      int = setInterval(() => {
        setEpoch(e => {
          if (e >= 5) { setRunning(false); return e; }
          return e + 1;
        });
      }, 700);
    }
    return () => clearInterval(int);
  }, [running]);

  const reset = () => { setEpoch(0); setRunning(false); };
  const start = () => { reset(); setTimeout(() => setRunning(true), 50); };

  // SVG canvas
  const W = 220; const H = 180; const PAD = 20;
  const toSvgX = x => PAD + ((x + 5) / 10) * (W - 2 * PAD);
  const toSvgY = y => (H - PAD) - ((y + 5) / 10) * (H - 2 * PAD);

  // Identity line: y = x
  const identityPts = [[-5,-5],[5,5]].map(([x,y]) => `${toSvgX(x)},${toSvgY(y)}`).join(' ');

  // Random init spline knots at epoch 0 (wildly bent) — fixed chaos shape
  const randomKnots = [[-5,-5],[-3,-1],[0,3],[2,-2],[5,5]];
  // After "training" with random init, loss explodes — curve gets worse each step
  const randomEpochOffsets = [
    [0,0,0,0,0],       // epoch 0
    [0,0.5,-0.8,0.6,0],// epoch 1: getting more chaotic
    [0,1.2,-1.8,1.4,0],// epoch 2: diverging
    [0,2.5,-3.2,2.8,0],// epoch 3: exploding
    [0,3.5,-4.5,4.0,0],// epoch 4: NaN territory
    [0,3.5,-4.5,4.0,0],// epoch 5: stuck
  ];
  const randomCurveKnots = randomKnots.map(([x,y], i) => [x, Math.max(-5, Math.min(5, y + (randomEpochOffsets[epoch]?.[i] ?? 0)))]);

  // Zero init spline: starts as identity, gradually learns a gentle S-curve
  const zeroEpochKnots = [
    [[-5,-5],[-3,-3],[0,0],[3,3],[5,5]],           // epoch 0: perfect identity
    [[-5,-5],[-3,-2.5],[0,0.3],[3,3.5],[5,5]],     // epoch 1: tiny bend
    [[-5,-5],[-3,-2],  [0,0.6],[3,3.8],[5,5]],     // epoch 2
    [[-5,-5],[-3,-1.5],[0,0.9],[3,4.0],[5,5]],     // epoch 3
    [[-5,-5],[-3,-1.2],[0,1.1],[3,4.1],[5,5]],     // epoch 4
    [[-5,-5],[-3,-1.0],[0,1.2],[3,4.2],[5,5]],     // epoch 5: learned S-curve
  ];
  const zeroCurveKnots = zeroEpochKnots[epoch];

  // Catmull-Rom-ish smooth path from knots
  const knotsToPath = (knots) => {
    const pts = knots.map(([x,y]) => [toSvgX(x), toSvgY(y)]);
    if (pts.length < 2) return '';
    let d = `M ${pts[0][0]} ${pts[0][1]}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(0, i-1)];
      const p1 = pts[i];
      const p2 = pts[i+1];
      const p3 = pts[Math.min(pts.length-1, i+2)];
      const cp1x = p1[0] + (p2[0]-p0[0])/6;
      const cp1y = p1[1] + (p2[1]-p0[1])/6;
      const cp2x = p2[0] - (p3[0]-p1[0])/6;
      const cp2y = p2[1] - (p3[1]-p1[1])/6;
      d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2[0]} ${p2[1]}`;
    }
    return d;
  };

  // Loss values
  const randomLoss = [1.2, 3.8, 12.4, 48.2, 210.0, null][epoch];
  const zeroLoss   = [1.2, 0.9, 0.65, 0.48, 0.35, 0.26][epoch];

  const randomPath = knotsToPath(randomCurveKnots);
  const zeroPath   = knotsToPath(zeroCurveKnots);

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">Why Zero-Init? Watch What Happens on Epoch 0</div>

        {/* Mode toggle */}
        <div className="flex gap-2 justify-center">
          <button onClick={() => { setMode('random'); reset(); }}
            className={`text-[10px] px-3 py-1.5 rounded-lg font-bold border transition-all ${mode === 'random' ? 'bg-rose-900/50 border-rose-500 text-rose-300' : 'bg-slate-800 border-slate-600 text-slate-400 hover:border-slate-500'}`}>
            ⚠️ Random Init (default)
          </button>
          <button onClick={() => { setMode('zero'); reset(); }}
            className={`text-[10px] px-3 py-1.5 rounded-lg font-bold border transition-all ${mode === 'zero' ? 'bg-emerald-900/50 border-emerald-500 text-emerald-300' : 'bg-slate-800 border-slate-600 text-slate-400 hover:border-slate-500'}`}>
            ✅ Zero Init (our code)
          </button>
        </div>

        {/* Side-by-side or single panel */}
        <div className="grid grid-cols-2 gap-3">

          {/* LEFT — spline curve visual */}
          <div className={`bg-slate-800/60 rounded-xl border p-2 flex flex-col gap-1 ${mode === 'random' ? 'border-rose-700/50' : 'border-emerald-700/50'}`}>
            <div className={`text-[9px] font-bold uppercase tracking-wider text-center ${mode === 'random' ? 'text-rose-400' : 'text-emerald-400'}`}>
              Spline Shape — Epoch {epoch}
            </div>
            <div className="overflow-hidden rounded">
              <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
                {/* Grid */}
                {[-4,-2,0,2,4].map(v => (
                  <g key={v}>
                    <line x1={toSvgX(v)} y1={PAD} x2={toSvgX(v)} y2={H-PAD} stroke="#1e293b" strokeWidth="1"/>
                    <line x1={PAD} y1={toSvgY(v)} x2={W-PAD} y2={toSvgY(v)} stroke="#1e293b" strokeWidth="1"/>
                  </g>
                ))}
                {/* Bounding box */}
                <rect x={PAD} y={PAD} width={W-2*PAD} height={H-2*PAD} fill="none" stroke="#334155" strokeWidth="1" strokeDasharray="3"/>
                {/* Identity reference line */}
                <polyline points={identityPts} fill="none" stroke="#334155" strokeWidth="1" strokeDasharray="4"/>
                <text x={toSvgX(3.5)} y={toSvgY(4.2)} fontSize="7" fill="#475569" textAnchor="middle">y=x (identity)</text>
                {/* The actual spline */}
                {mode === 'random' ? (
                  <path d={randomPath} fill="none" stroke="#f87171" strokeWidth="2.5" className="transition-all duration-500"/>
                ) : (
                  <path d={zeroPath} fill="none" stroke="#34d399" strokeWidth="2.5" className="transition-all duration-500"/>
                )}
                {/* Axis labels */}
                <text x={W/2} y={H-2} fontSize="7" fill="#64748b" textAnchor="middle">Input θ</text>
                <text x={6} y={H/2} fontSize="7" fill="#64748b" textAnchor="middle" transform={`rotate(-90,6,${H/2})`}>Output</text>
              </svg>
            </div>
          </div>

          {/* RIGHT — loss bar + explanation */}
          <div className="flex flex-col gap-2">
            <div className={`bg-slate-800/60 rounded-xl border p-2 ${mode === 'random' ? 'border-rose-700/50' : 'border-emerald-700/50'}`}>
              <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400 mb-1.5">Training Loss</div>
              {/* Loss bar chart: epochs 0-5 */}
              <div className="overflow-hidden rounded">
                <svg width="100%" viewBox="0 0 120 70">
                  {[0,1,2,3,4,5].map(e => {
                    const rLoss = [1.2,3.8,12.4,48.2,210,210][e];
                    const zLoss = [1.2,0.9,0.65,0.48,0.35,0.26][e];
                    const loss = mode === 'random' ? rLoss : zLoss;
                    const maxLoss = mode === 'random' ? 210 : 1.2;
                    const barH = Math.min(55, (loss / maxLoss) * 55);
                    const isActive = e === epoch;
                    const color = mode === 'random' ? (e >= 3 ? '#ef4444' : '#f97316') : '#34d399';
                    return (
                      <g key={e}>
                        <rect
                          x={8 + e * 18} y={60 - barH} width={12} height={barH}
                          fill={color} fillOpacity={isActive ? 1 : 0.4} rx="1"
                          className="transition-all duration-500"
                        />
                        <text x={14 + e * 18} y={68} textAnchor="middle" fontSize="6" fill={isActive ? '#e2e8f0' : '#64748b'}>{e}</text>
                      </g>
                    );
                  })}
                  <text x={60} y={8} textAnchor="middle" fontSize="6" fill="#64748b">epoch →</text>
                </svg>
              </div>
              <div className={`text-center font-mono text-[10px] font-bold mt-1 ${mode === 'random' && epoch >= 3 ? 'text-rose-400 animate-pulse' : mode === 'random' ? 'text-orange-300' : 'text-emerald-300'}`}>
                {mode === 'random'
                  ? (epoch >= 4 ? '💥 NaN / exploded!' : `Loss: ${[1.2,3.8,12.4,48.2,'∞','∞'][epoch]}`)
                  : `Loss: ${zeroLoss.toFixed(2)} ✓`
                }
              </div>
            </div>

            {/* What the MLP outputs */}
            <div className={`bg-slate-800/60 rounded-xl border p-2 text-[9px] leading-relaxed ${mode === 'random' ? 'border-rose-700/30 text-rose-200' : 'border-emerald-700/30 text-emerald-200'}`}>
              <div className="font-bold uppercase tracking-wider mb-1 text-slate-400">MLP final layer outputs at Epoch 0:</div>
              {mode === 'random' ? (
                <div className="font-mono text-rose-300">
                  W logits: <span className="text-rose-400">[2.3, -1.8, 4.1, ...]</span><br/>
                  H logits: <span className="text-rose-400">[-3.2, 5.7, -2.1, ...]</span><br/>
                  D logits: <span className="text-rose-400">[1.9, -4.5, 3.3, ...]</span><br/>
                  <span className="text-rose-500 mt-1 block">→ Random bin sizes → bent curve → data violently distorted</span>
                </div>
              ) : (
                <div className="font-mono text-emerald-300">
                  W logits: <span className="text-emerald-400">[0.0, 0.0, 0.0, ...]</span><br/>
                  H logits: <span className="text-emerald-400">[0.0, 0.0, 0.0, ...]</span><br/>
                  D logits: <span className="text-emerald-400">[0.0, 0.0, 0.0, ...]</span><br/>
                  <span className="text-emerald-500 mt-1 block">→ Equal bins → softmax(0)=uniform → curve = y=x ✓</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Why softmax(0) = uniform */}
        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-3 text-[10px] text-slate-400 leading-relaxed">
          <span className="text-white font-bold">The math chain: </span>
          {mode === 'zero' ? (
            <>
              All logits = <code className="text-emerald-300">0.0</code> →{' '}
              <code className="text-amber-300">softmax([0,0,0,0]) = [0.25, 0.25, 0.25, 0.25]</code> → all bins equal width →{' '}
              <code className="text-amber-300">softplus(0) = ln(2) ≈ 0.693</code> → all slopes equal →{' '}
              <span className="text-emerald-300 font-bold">spline = straight diagonal = identity function (y = x)</span>.{' '}
              Training starts from a neutral, undistorted state.
            </>
          ) : (
            <>
              Random logits, e.g. <code className="text-rose-300">[-3.2, 5.7, -2.1]</code> →{' '}
              <code className="text-rose-300">softmax → [0.003, 0.993, 0.004]</code> → one gigantic bin, two tiny ones →{' '}
              spline is wildly bent on the <strong>very first forward pass</strong> →{' '}
              <span className="text-rose-400 font-bold">loss jumps to NaN before a single gradient step.</span>
            </>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="flex-shrink-0 flex justify-center gap-3 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setEpoch(e => Math.max(0, e-1))} disabled={epoch === 0 || running} active={false}>
          <ChevronLeft size={14}/> Prev Epoch
        </VisualButton>
        <VisualButton onClick={start} active={running}>
          <Play size={14}/> {running ? 'Training...' : 'Run Training'}
        </VisualButton>
        <VisualButton onClick={() => setEpoch(e => Math.min(5, e+1))} disabled={epoch === 5 || running} active={true}>
          Next Epoch <ChevronRight size={14}/>
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedConstraints = () => {
  const [applied, setApplied] = useState(false);

  return (
    <div className="relative w-full h-full bg-slate-900 flex flex-col items-center justify-center p-6 pb-14 gap-8">
      <div className="text-xs text-slate-400 font-mono text-center">
        Enforcing Physical Constraints via Activations
      </div>

      <div className="flex gap-8 w-full max-w-md justify-center mt-2">
        
        {/* Softmax for W and H */}
        <div className="flex flex-col items-center gap-2 flex-1">
           <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Widths & Heights</span>
           <div className="bg-slate-800 p-2 rounded-lg border border-slate-700 flex gap-1 h-24 items-end w-full justify-center relative overflow-hidden">
              {applied && <div className="absolute top-2 w-full text-center text-[8px] text-fuchsia-400 font-bold">Sum = 100% of Box</div>}
              <div className={`w-5 bg-fuchsia-500 transition-all duration-700 rounded-t ${applied ? 'h-[20%]' : 'h-[80%]'}`}></div>
              <div className={`w-5 bg-fuchsia-500 transition-all duration-700 rounded-t ${applied ? 'h-[50%]' : 'h-[30%]'}`}></div>
              <div className={`w-5 bg-fuchsia-500 transition-all duration-700 rounded-t ${applied ? 'h-[30%]' : 'h-[110%]'}`}></div>
           </div>
           <div className={`mt-2 px-2 py-1 rounded text-[10px] font-mono font-bold transition-colors duration-500 border ${applied ? 'bg-fuchsia-900/50 border-fuchsia-500 text-fuchsia-300 shadow-[0_0_10px_rgba(217,70,239,0.3)]' : 'bg-slate-800 border-slate-600 text-slate-500'}`}>
              F.softmax(x)
           </div>
        </div>

        {/* Softplus for Derivatives */}
        <div className="flex flex-col items-center gap-2 flex-1">
           <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Derivatives (Slopes)</span>
           <div className="bg-slate-800 p-2 rounded-lg border border-slate-700 flex gap-2 h-24 items-center w-full justify-center relative">
              <div className="absolute w-full h-[1px] bg-slate-500 top-1/2 border-t border-dashed border-slate-400/50"></div>
              <div className={`w-5 bg-teal-500 transition-all duration-700 rounded z-10 ${applied ? 'h-[40%] translate-y-[-20%]' : 'h-[60%] translate-y-[30%]'}`}></div>
              <div className={`w-5 bg-teal-500 transition-all duration-700 rounded z-10 ${applied ? 'h-[80%] translate-y-[-40%]' : 'h-[30%] translate-y-[15%]'}`}></div>
              <div className={`w-5 bg-teal-500 transition-all duration-700 rounded z-10 ${applied ? 'h-[20%] translate-y-[-10%]' : 'h-[50%] translate-y-[25%]'}`}></div>
           </div>
           <div className={`mt-2 px-2 py-1 rounded text-[10px] font-mono font-bold transition-colors duration-500 border ${applied ? 'bg-teal-900/50 border-teal-500 text-teal-300 shadow-[0_0_10px_rgba(20,184,166,0.3)]' : 'bg-slate-800 border-slate-600 text-slate-500'}`}>
              F.softplus(x)
           </div>
        </div>

      </div>

      <div className="text-[11px] text-slate-400 text-center max-w-[90%] leading-relaxed mt-4 h-12">
        {applied 
          ? <span className="text-emerald-300"><strong>Constraints Active:</strong> Widths/Heights now sum perfectly to span the bounding box without leaking. Slopes are forced above zero, ensuring the curve never dips downwards (guaranteeing invertibility).</span>
          : <span><strong>Raw MLP Outputs:</strong> Neural networks output unbounded raw numbers. Widths could explode outside the box, and slopes could be negative, instantly breaking the mathematical flow.</span>}
      </div>

      <div className="absolute bottom-4 right-4">
        <VisualButton onClick={() => setApplied(!applied)} active={applied}>
          <Activity size={14} /> {applied ? "Revert to Raw Outputs" : "Apply PyTorch Constraints"}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedScaffolding = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_STEP = 4;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(s => { if (s + 1 > MAX_STEP) { setIsPlaying(false); return s; } return s + 1; }), 3000);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      {/* Scrollable content area */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">
          Building the Knot Grid (W, H, D) &amp; Finding the Bin
        </div>

        <div className="w-full bg-slate-800/50 p-3 rounded-xl border border-slate-700 flex flex-col gap-2 font-mono text-[10px] shadow-inner">

          {/* ── Step 0: Three constrained inputs ── */}
          <div className="grid grid-cols-3 gap-2">
            <div className="flex flex-col items-center gap-1">
              <span className="text-sky-400 font-bold uppercase tracking-wider text-[9px]">W (X-axis)</span>
              <span className="bg-sky-900/40 text-sky-300 px-2 py-1 rounded border border-sky-500/50 w-full text-center">[2.0, 3.0, 5.0]</span>
              <span className="text-slate-500 text-[8px]">K bin widths</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-rose-400 font-bold uppercase tracking-wider text-[9px]">H (Y-axis)</span>
              <span className="bg-rose-900/40 text-rose-300 px-2 py-1 rounded border border-rose-500/50 w-full text-center">[1.5, 4.5, 4.0]</span>
              <span className="text-slate-500 text-[8px]">K bin heights</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-amber-400 font-bold uppercase tracking-wider text-[9px]">D (Slopes)</span>
              <span className="bg-amber-900/40 text-amber-300 px-2 py-1 rounded border border-amber-500/50 w-full text-center">[0.8, 1.2, 0.9]</span>
              <span className="text-slate-500 text-[8px]">K-1 inner derivs</span>
            </div>
          </div>

          {/* ── Step 1: cumsum on W and H ── */}
          <div className={`grid grid-cols-3 gap-2 transition-all duration-500 ${step >= 1 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none'}`}>
            <div className="flex flex-col items-center gap-1">
              <span className="text-slate-400 text-[8px] flex items-center gap-1"><ArrowDown size={9} className="text-amber-400"/>cumsum</span>
              <span className="bg-amber-900/30 text-amber-300 px-2 py-1 rounded border border-amber-500/50 w-full text-center">[2.0, 5.0, 10.0]</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-slate-400 text-[8px] flex items-center gap-1"><ArrowDown size={9} className="text-amber-400"/>cumsum</span>
              <span className="bg-amber-900/30 text-amber-300 px-2 py-1 rounded border border-amber-500/50 w-full text-center">[1.5, 6.0, 10.0]</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-slate-400 text-[8px] flex items-center gap-1"><ArrowDown size={9} className="text-slate-500"/>no cumsum</span>
              <span className="bg-slate-800 text-slate-500 px-2 py-1 rounded border border-slate-600 w-full text-center italic">unchanged</span>
            </div>
          </div>

          {/* ── Step 2: F.pad + map / pad D with 1.0 ── */}
          <div className={`grid grid-cols-3 gap-2 transition-all duration-500 ${step >= 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none'}`}>
            <div className="flex flex-col items-center gap-1">
              <span className="text-slate-400 text-[8px] flex items-center gap-1"><ArrowDown size={9} className="text-fuchsia-400"/>pad+map→[-5,5]</span>
              <span className="bg-fuchsia-900/30 text-fuchsia-300 px-2 py-1 rounded border border-fuchsia-500/50 w-full text-center">X-knots: [-5,-3,0,5]</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-slate-400 text-[8px] flex items-center gap-1"><ArrowDown size={9} className="text-fuchsia-400"/>pad+map→[-5,5]</span>
              <span className="bg-fuchsia-900/30 text-fuchsia-300 px-2 py-1 rounded border border-fuchsia-500/50 w-full text-center">Y-knots: [-5,-2, 3,5]</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-slate-400 text-[8px] flex items-center gap-1"><ArrowDown size={9} className="text-amber-400"/>pad 1.0 at ends</span>
              <span className="bg-amber-900/30 text-amber-300 px-2 py-1 rounded border border-amber-500/50 w-full text-center">[1.0,0.8,1.2,0.9,1.0]</span>
            </div>
          </div>

          {/* ── Step 3: searchsorted ── */}
          <div className={`mt-1 pt-2 border-t border-slate-600 flex flex-col gap-2 transition-all duration-500 ${step >= 3 ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'}`}>
            <div className="flex justify-between items-center">
              <span className="text-emerald-400 font-bold">Input θ = 2.5</span>
              <span className="bg-emerald-900/30 text-emerald-300 px-2 py-1 rounded border border-emerald-500/50">searchsorted(X-knots)</span>
            </div>
            <div className="w-full h-6 flex rounded overflow-hidden border border-slate-600 relative">
              <div className="flex-1 bg-slate-800 border-r border-slate-700 flex items-center justify-center text-[8px] text-slate-500">Bin 0 [-5,−3]</div>
              <div className="flex-[1.5] bg-slate-800 border-r border-slate-700 flex items-center justify-center text-[8px] text-slate-500">Bin 1 [−3, 0]</div>
              <div className="flex-[2.5] bg-emerald-500/20 flex items-center justify-center text-[8px] text-emerald-300 font-bold">Bin 2 ✓ [0, 5]</div>
              <div className="absolute top-0 bottom-0 w-0.5 bg-emerald-400 shadow-[0_0_6px_#34d399]" style={{ left: '75%' }}>
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 text-emerald-400 text-[8px] font-bold">θ</div>
              </div>
            </div>
          </div>

          {/* ── Step 4: gather W_k, H_k, D_0, D_1 ── */}
          <div className={`pt-2 border-t border-slate-600 transition-all duration-500 ${step >= 4 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none'}`}>
            <div className="text-slate-400 text-[9px] font-bold mb-1.5">torch.gather for Bin 2 →</div>
            <div className="grid grid-cols-4 gap-1.5">
              <div className="flex flex-col items-center gap-0.5">
                <span className="bg-sky-900/40 text-sky-300 px-1.5 py-1 rounded border border-sky-500/50 w-full text-center font-bold">W_k=5.0</span>
                <span className="text-sky-500 text-[8px]">bin width</span>
              </div>
              <div className="flex flex-col items-center gap-0.5">
                <span className="bg-rose-900/40 text-rose-300 px-1.5 py-1 rounded border border-rose-500/50 w-full text-center font-bold">H_k=4.0</span>
                <span className="text-rose-500 text-[8px]">bin height</span>
              </div>
              <div className="flex flex-col items-center gap-0.5">
                <span className="bg-amber-900/40 text-amber-300 px-1.5 py-1 rounded border border-amber-500/50 w-full text-center font-bold">D_0=1.2</span>
                <span className="text-amber-500 text-[8px]">left slope</span>
              </div>
              <div className="flex flex-col items-center gap-0.5">
                <span className="bg-amber-900/40 text-amber-300 px-1.5 py-1 rounded border border-amber-500/50 w-full text-center font-bold">D_1=0.9</span>
                <span className="text-amber-500 text-[8px]">right slope</span>
              </div>
            </div>
          </div>

        </div>

        {/* Step description — inline, no fixed height */}
        <p className="text-center text-[11px] text-slate-400 leading-relaxed px-2">
          {step === 0 && "All 3 MLP outputs enter: W (K bin widths for the X-axis), H (K bin heights for the Y-axis), and D (K-1 inner slopes between knots)."}
          {step === 1 && "torch.cumsum runs on W and H to convert individual bin sizes into running knot positions. D doesn't need a cumsum — it already lives at the boundaries."}
          {step === 2 && "F.pad + normalization maps cumW → exact X-axis knot coords, cumH → exact Y-axis knot coords. D gets 1.0 padded at both ends to connect smoothly to the linear tails outside the box."}
          {step === 3 && "When input θ=2.5 arrives, searchsorted does a fast binary search on the X-knots (cumwidths) to instantly find it lands in Bin 2, spanning [0, 5]."}
          {step === 4 && "torch.gather extracts exactly 4 values for Bin 2: W_k (X-width), H_k (Y-height), D_0 (left-knot slope), D_1 (right-knot slope). These 4 values are all the algebra needs."}
        </p>
      </div>

      {/* Controls pinned to bottom — always visible */}
      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setStep(s => Math.max(0, s - 1))} active={false} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev Step
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? "Pause" : "Play Sequence"}
        </VisualButton>
        <VisualButton onClick={() => setStep(s => Math.min(MAX_STEP, s + 1))} active={true} disabled={step === MAX_STEP || isPlaying}>
          Next Step <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedCumsumExplainer = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_STEP = 4;

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(s => { if (s + 1 > MAX_STEP) { setIsPlaying(false); return s; } return s + 1; }), 2800);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  // The raw widths, their cumsum, padded, and mapped values
  const widths  = [2.0, 3.0, 5.0];
  const cumsum  = [2.0, 5.0, 10.0];
  const padded  = [0.0, 2.0, 5.0, 10.0];
  const mapped  = [-5.0, -3.0, 0.0, 5.0];
  const bound   = 5.0;

  // Ruler bar config
  const totalW = 260;
  const barH   = 28;
  const colors = ['#38bdf8','#f472b6','#a78bfa'];
  const binLabels = ['Bin 0','Bin 1','Bin 2'];

  // SVG ruler: segments sized by W[i]/10 * totalW
  const segments = widths.map((w, i) => ({ w, px: (w / 10) * totalW, color: colors[i], label: binLabels[i] }));
  const cumsumPx = cumsum.map(v => (v / 10) * totalW);
  const mappedPx = mapped.map(v => ((v + bound) / (2 * bound)) * totalW);

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 min-h-0">
        <div className="text-xs text-slate-400 font-mono text-center">cumsum → F.pad → map: Building Knot Positions from Bin Sizes</div>

        {/* ─── STEP 0: Raw sizes ─── */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-3 flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-bold text-sky-400 uppercase tracking-wider">Step 0 — Raw W (bin sizes from MLP)</span>
            <span className="font-mono text-[10px] text-sky-300 bg-sky-900/30 px-2 py-0.5 rounded border border-sky-700">[2.0, 3.0, 5.0]</span>
          </div>
          <p className="text-[10px] text-slate-400 leading-relaxed">These are <strong className="text-sky-300">widths</strong> — how wide each bin is. They do NOT tell you where bins <em>start or end</em>. Think of them like the lengths of 3 planks laid end-to-end.</p>
          {/* Ruler showing segments */}
          <div className="overflow-hidden rounded">
            <svg width="100%" viewBox={`0 0 ${totalW} ${barH + 20}`}>
              {segments.reduce((acc, seg, i) => {
                const x = acc.x;
                return { x: x + seg.px, els: [...acc.els,
                  <g key={i}>
                    <rect x={x} y={0} width={seg.px} height={barH} fill={seg.color} fillOpacity={0.25} stroke={seg.color} strokeWidth="1"/>
                    <text x={x + seg.px/2} y={barH/2 + 4} textAnchor="middle" fontSize="8" fill={seg.color} fontWeight="bold">{seg.label} ({seg.w})</text>
                    <text x={x} y={barH + 13} textAnchor="middle" fontSize="7" fill="#64748b">{i === 0 ? '' : cumsum[i-1]}</text>
                  </g>
                ]};
              }, { x: 0, els: [] }).els}
              <text x={totalW} y={barH + 13} textAnchor="middle" fontSize="7" fill="#64748b">10</text>
              <text x={0} y={barH + 13} textAnchor="middle" fontSize="7" fill="#64748b">0</text>
            </svg>
          </div>
          <p className="text-[10px] text-amber-400 font-mono text-center">Problem: we know widths, but not where each bin <em>starts</em> on the X-axis.</p>
        </div>

        {/* ─── STEP 1: cumsum ─── */}
        <div className={`bg-slate-800/50 border border-amber-700/50 rounded-xl p-3 flex flex-col gap-2 transition-all duration-500 ${step >= 1 ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider">Step 1 — torch.cumsum → running right-edges</span>
            <span className="font-mono text-[10px] text-amber-300 bg-amber-900/30 px-2 py-0.5 rounded border border-amber-700">[2.0, 5.0, 10.0]</span>
          </div>
          <div className="grid grid-cols-3 gap-1 font-mono text-[10px]">
            {widths.map((w, i) => (
              <div key={i} className="flex flex-col items-center gap-0.5">
                <div className="text-slate-500">{widths.slice(0, i+1).join(' + ')}</div>
                <div className="text-amber-300 font-bold">= {cumsum[i]}</div>
                <div className="text-slate-500 text-[8px]">right edge of {binLabels[i]}</div>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-slate-400 leading-relaxed">
            cumsum adds each value to all previous ones. Now <code className="text-amber-300">cumsum[i]</code> is the <strong className="text-amber-300">right-edge position</strong> of bin i.
            We went from 3 sizes → 3 <em>right-edge coordinates</em>. Still 3 numbers, but their meaning changed.
          </p>
        </div>

        {/* ─── STEP 2: F.pad ─── */}
        <div className={`bg-slate-800/50 border border-fuchsia-700/50 rounded-xl p-3 flex flex-col gap-2 transition-all duration-500 ${step >= 2 ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-bold text-fuchsia-400 uppercase tracking-wider">Step 2 — F.pad(_, (1,0), value=0) → add left boundary</span>
            <span className="font-mono text-[10px] text-fuchsia-300 bg-fuchsia-900/30 px-2 py-0.5 rounded border border-fuchsia-700">[0.0, 2.0, 5.0, 10.0]</span>
          </div>
          <p className="text-[10px] text-slate-400 leading-relaxed">
            After cumsum we have right-edges but no left-edge for Bin 0. <code className="text-fuchsia-300">F.pad(_, (1,0))</code> prepends a <strong className="text-fuchsia-300">0.0</strong> at the front.
            Now we have <strong className="text-fuchsia-300">K+1 = 4 boundary coordinates</strong> — one for each knot (left edge of Bin 0 through right edge of Bin 2).
          </p>
          <div className="flex gap-1 font-mono text-[10px] justify-center flex-wrap">
            {padded.map((v, i) => (
              <div key={i} className={`flex flex-col items-center gap-0.5 px-2 py-1 rounded border ${i === 0 ? 'border-fuchsia-500 bg-fuchsia-900/30 text-fuchsia-300' : 'border-slate-600 bg-slate-700/50 text-amber-300'}`}>
                <span className="font-bold">{v.toFixed(1)}</span>
                <span className="text-[7px] text-slate-500">{i === 0 ? '← padded' : i < padded.length - 1 ? `L of B${i}` : 'R of B2'}</span>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-amber-400 font-mono text-center">3 sizes → 3 right-edges → <strong>4 boundary coordinates</strong> (K+1)</p>
        </div>

        {/* ─── STEP 3: map to [-5, 5] ─── */}
        <div className={`bg-slate-800/50 border border-emerald-700/50 rounded-xl p-3 flex flex-col gap-2 transition-all duration-500 ${step >= 3 ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Step 3 — map [0,10] → [-5, 5]</span>
            <span className="font-mono text-[10px] text-emerald-300 bg-emerald-900/30 px-2 py-0.5 rounded border border-emerald-700">[-5, -3, 0, 5]</span>
          </div>
          <p className="text-[10px] text-slate-400 leading-relaxed">
            The formula <code className="text-emerald-300">(2·bound·cumW / cumW[-1]) − bound</code> linearly maps the range <code>[0, 10]</code> → <code>[-5, 5]</code>.
            The first and last knots are then <strong className="text-emerald-300">pinned exactly</strong> to ±bound so tails attach cleanly.
          </p>
          <div className="overflow-hidden rounded">
            <svg width="100%" viewBox={`0 0 ${totalW} ${barH + 20}`}>
              {/* background bar */}
              <rect x={0} y={0} width={totalW} height={barH} fill="#1e293b" rx="3"/>
              {/* colored bins */}
              {segments.reduce((acc, seg, i) => {
                const x = acc.x;
                return { x: x + seg.px, els: [...acc.els,
                  <rect key={i} x={x} y={0} width={seg.px} height={barH} fill={seg.color} fillOpacity={0.2} stroke={seg.color} strokeWidth="1" />
                ]};
              }, { x: 0, els: [] }).els}
              {/* mapped knot lines */}
              {mappedPx.map((px, i) => (
                <g key={i}>
                  <line x1={px} y1={0} x2={px} y2={barH} stroke="#34d399" strokeWidth="1.5"/>
                  <text x={px} y={barH + 13} textAnchor="middle" fontSize="8" fill="#34d399" fontWeight="bold">{mapped[i]}</text>
                </g>
              ))}
              <text x={totalW/2} y={barH/2 + 4} textAnchor="middle" fontSize="8" fill="#94a3b8">X-axis knots</text>
            </svg>
          </div>
        </div>

        {/* ─── STEP 4: why D is different ─── */}
        <div className={`bg-slate-800/50 border border-slate-600 rounded-xl p-3 flex flex-col gap-2 transition-all duration-500 ${step >= 4 ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
          <span className="text-[10px] font-bold text-slate-300 uppercase tracking-wider">Step 4 — Why D skips cumsum</span>
          <p className="text-[10px] text-slate-400 leading-relaxed">
            D (slopes) are already at the <strong className="text-white">boundary points</strong>, not inside bins. They don't need cumsum.
            We just pad <code className="text-amber-300">1.0</code> at both ends: the linear tails outside the box must have slope = 1 to act as identity functions.
            So D grows from K−1 = 2 → K+1 = 4 values, aligned with the 4 knots.
          </p>
          <div className="flex gap-1 justify-center font-mono text-[10px] flex-wrap">
            {['1.0','0.8','1.2','0.9','1.0'].map((v, i) => (
              <div key={i} className={`flex flex-col items-center px-2 py-1 rounded border ${i === 0 || i === 4 ? 'border-amber-500 bg-amber-900/30 text-amber-300' : 'border-slate-600 bg-slate-700/50 text-slate-300'}`}>
                <span className="font-bold">{v}</span>
                <span className="text-[7px] text-slate-500">{i === 0 ? '← pad' : i === 4 ? 'pad →' : `D${i-1}`}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex-shrink-0 flex justify-center gap-4 py-3 border-t border-slate-700/60 bg-slate-900">
        <VisualButton onClick={() => setStep(s => Math.max(0, s - 1))} active={false} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev Step
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? "Pause" : "Play"}
        </VisualButton>
        <VisualButton onClick={() => setStep(s => Math.min(MAX_STEP, s + 1))} active={true} disabled={step === MAX_STEP || isPlaying}>
          Next Step <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedCoupling = () => {
  const [step, setStep] = useState(0); 
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(s => (s + 1) % 4), 2500);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  return (
    <div className="relative w-full h-full bg-slate-900 flex flex-col items-center justify-center p-6 pb-20 border border-slate-800 gap-4">
      <div className="text-xs text-slate-400 mb-4 font-mono text-center">NeuralSplineCouplingLayer: The Orchestrator</div>
      
      <div className="flex w-full max-w-lg items-stretch justify-center h-48 gap-8 relative mt-2">
         
         {/* Input Tensor */}
         <div className="flex flex-col items-center w-20 relative">
            <span className="text-[10px] font-bold text-slate-400 mb-2">Input (θ)</span>
            <div className="flex flex-col w-full h-full gap-1">
               <div className={`w-full bg-blue-500/50 border border-blue-400 rounded transition-all duration-500 ${step >= 1 ? 'h-1/2' : 'h-full'}`}></div>
               <div className={`w-full bg-rose-500/50 border border-rose-400 rounded transition-all duration-500 ${step >= 1 ? 'h-1/2 opacity-100' : 'h-0 opacity-0'}`}></div>
            </div>
            {step >= 1 && <span className="absolute -left-12 top-1/4 text-[9px] text-blue-400 font-bold">θ_1</span>}
            {step >= 1 && <span className="absolute -left-12 bottom-1/4 text-[9px] text-rose-400 font-bold">θ_2</span>}
         </div>

         {/* Center Logic */}
         <div className="flex-1 flex flex-col justify-between py-4 relative">
            {step >= 2 && (
              <div className="absolute top-[20%] left-0 w-full h-[60%] border-l-2 border-t-2 border-b-2 border-dashed border-slate-500 rounded-l-xl -z-10 animate-in fade-in"></div>
            )}
            
            {/* Top Path (Unchanged) */}
            <div className={`flex justify-center transition-all duration-500 ${step >= 3 ? 'opacity-100' : 'opacity-0'}`}>
               <span className="bg-slate-800 text-slate-400 text-[9px] px-2 py-1 rounded border border-slate-700 shadow-sm">Identity Pass-through</span>
            </div>

            {/* Brain & Math */}
            <div className={`bg-slate-800 border-2 border-indigo-500 rounded-xl p-3 flex flex-col items-center shadow-[0_0_15px_rgba(99,102,241,0.2)] transition-all duration-500 z-10 ${step >= 2 ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
               <Cpu size={20} className="text-indigo-400 mb-1"/>
               <span className="text-[9px] font-bold text-white text-center">ResidualMLP +<br/>Spline Engine</span>
            </div>
         </div>

         {/* Output Tensor */}
         <div className="flex flex-col items-center w-20 relative">
            <span className="text-[10px] font-bold text-slate-400 mb-2">Output (Y)</span>
            <div className="flex flex-col w-full h-full gap-1">
               <div className={`w-full bg-blue-500/50 border border-blue-400 rounded transition-all duration-500 ${step >= 3 ? 'h-1/2 opacity-100' : 'h-1/2 opacity-0'}`}></div>
               <div className={`w-full bg-fuchsia-500/80 border border-fuchsia-400 rounded transition-all duration-500 ${step >= 3 ? 'h-1/2 opacity-100' : 'h-1/2 opacity-0 shadow-[0_0_15px_rgba(217,70,239,0.5)]'}`}></div>
            </div>
            {step >= 3 && <span className="absolute -right-12 top-1/4 text-[9px] text-blue-400 font-bold">θ_1</span>}
            {step >= 3 && <span className="absolute -right-16 bottom-1/4 text-[9px] text-fuchsia-400 font-bold">Spline(θ_2)</span>}
         </div>

      </div>

      <div className="text-center text-[11px] text-slate-400 mt-8 max-w-[90%] leading-relaxed h-12">
        {step === 0 && "1. The Orchestrator receives the full tensor array."}
        {step === 1 && "2. It physically splits the array down the middle into θ_1 and θ_2 using tensor slicing."}
        {step === 2 && "3. It routes θ_1 to the MLP, generating parameters to modify θ_2 via the Spline Math."}
        {step === 3 && "4. Finally, it uses torch.cat() to stitch the untouched top half and the morphed bottom half back together!"}
      </div>

      {/* Playback Controls */}
      <div className="absolute bottom-4 flex gap-4">
        <VisualButton onClick={() => setStep(s => Math.max(0, s - 1))} active={false} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev Step
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? "Pause" : "Play Sequence"}
        </VisualButton>
        <VisualButton onClick={() => setStep(s => Math.min(3, s + 1))} active={true} disabled={step === 3 || isPlaying}>
          Next Step <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedWalkthrough = () => {
  const [activePart, setActivePart] = useState(0);
  const lineRefs = useRef({});

  const codeLines = [
    { text: "import torch", part: null },
    { text: "import torch.nn as nn", part: null },
    { text: "import torch.nn.functional as F", part: null },
    { text: "", part: null },
    { text: "class ResidualMLP(nn.Module):", part: 0 },
    { text: "    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):", part: 0 },
    { text: "        super().__init__()", part: 0 },
    { text: "        # ... build standard Linear -> GELU blocks ...", part: 0 },
    { text: "        self.final_layer = nn.Linear(hidden_dim, output_dim)", part: 0 },
    { text: "", part: 0 },
    { text: "        # CRITICAL: Zero Initialization", part: 0 },
    { text: "        nn.init.zeros_(self.final_layer.weight)", part: 0 },
    { text: "        nn.init.zeros_(self.final_layer.bias)", part: 0 },
    { text: "", part: 0 },
    { text: "    def forward(self, x):", part: 1 },
    { text: "        out = self.initial_layer(x)", part: 1 },
    { text: "        for block in self.blocks:", part: 1 },
    { text: "            # Residual Skip Connection", part: 1 },
    { text: "            out = self.activation(block(out) + out)", part: 1 },
    { text: "        return self.final_layer(out)", part: 1 },
    { text: "", part: null },
    { text: "def rational_quadratic_spline(inputs, unnorm_widths, unnorm_heights, unnorm_derivs):", part: 2 },
    { text: "    # STEP 1: APPLY CONSTRAINTS", part: 2 },
    { text: "    widths = F.softmax(unnorm_widths, dim=-1) * 2B", part: 2 },
    { text: "    heights = F.softmax(unnorm_heights, dim=-1) * 2B", part: 2 },
    { text: "    derivatives = F.softplus(unnorm_derivs)  # Must be positive", part: 2 },
    { text: "", part: null },
    { text: "    # STEP 2: BUILD THE SCAFFOLDING (KNOTS)", part: 3 },
    { text: "    cumwidths = torch.cumsum(widths, dim=-1)", part: 3 },
    { text: "    cumheights = torch.cumsum(heights, dim=-1)", part: 3 },
    { text: "    # Pad with zeros to start at the bottom-left of the box", part: 3 },
    { text: "    cumwidths = F.pad(cumwidths, pad=(1, 0), value=0.0)", part: 3 },
    { text: "", part: null },
    { text: "    # STEP 3: FIND THE BIN & LOCAL COORDS", part: 4 },
    { text: "    if not inverse:", part: 4 },
    { text: "        bin_idx = torch.searchsorted(cumwidths, inputs) - 1", part: 4 },
    { text: "    else:", part: 4 },
    { text: "        bin_idx = torch.searchsorted(cumheights, inputs) - 1", part: 4 },
    { text: "", part: 4 },
    { text: "    # Gather W, H, and D for the specific bin we landed in", part: 4 },
    { text: "    W_k = torch.gather(widths, -1, bin_idx)", part: 4 },
    { text: "    H_k = torch.gather(heights, -1, bin_idx)", part: 4 },
    { text: "    D_0 = torch.gather(derivatives, -1, bin_idx)", part: 4 },
    { text: "    D_1 = torch.gather(derivatives, -1, bin_idx + 1)", part: 4 },
    { text: "    S = H_k / W_k  # The Straight-Line Slope", part: 4 },
    { text: "", part: null },
    { text: "    # STEP 4: THE FORWARD MATH (Algebra)", part: 5 },
    { text: "    if not inverse:", part: 5 },
    { text: "        xi = (inputs - start_x) / W_k", part: 5 },
    { text: "        # Top shape / Bottom shape", part: 5 },
    { text: "        numerator = H_k * (S * xi**2 + D_0 * xi * (1 - xi))", part: 5 },
    { text: "        denominator = S + (D_1 + D_0 - 2 * S) * xi * (1 - xi)", part: 5 },
    { text: "        outputs = start_y + numerator / denominator", part: 5 },
    { text: "        # Quotient Rule Derivative for the Volume Penalty", part: 5 },
    { text: "        logabsdet = torch.log(deriv_top) - 2 * torch.log(denominator)", part: 5 },
    { text: "        return outputs, logabsdet", part: 5 },
    { text: "", part: null },
    { text: "    # STEP 5: THE INVERSE MATH (Quadratic Formula)", part: 6 },
    { text: "    else:", part: 6 },
    { text: "        a = H_k * (S - D_0) + y_shifted * (D_1 + D_0 - 2 * S)", part: 6 },
    { text: "        b = H_k * D_0 - y_shifted * (D_1 + D_0 - 2 * S)", part: 6 },
    { text: "        c = -S * y_shifted", part: 6 },
    { text: "        # Instant exact solution: (-b ± sqrt(b² - 4ac)) / 2a", part: 6 },
    { text: "        xi = (2 * c) / (-b - torch.sqrt(b**2 - 4*a*c))", part: 6 },
    { text: "        return start_x + xi * W_k", part: 6 },
    { text: "", part: null },
    { text: "class NeuralSplineCouplingLayer(nn.Module):", part: 7 },
    { text: "    def forward(self, theta, condition):", part: 7 },
    { text: "        # THE SPLIT", part: 7 },
    { text: "        theta_1, theta_2 = theta[:, :half], theta[:, half:]", part: 7 },
    { text: "", part: 7 },
    { text: "        # THE BRAIN", part: 7 },
    { text: "        brain_input = torch.cat([theta_1, condition], dim=-1)", part: 7 },
    { text: "        raw_params = self.brain(brain_input)", part: 7 },
    { text: "", part: null },
    { text: "        # IDENTITY TAILS & EXECUTION", part: 8 },
    { text: "        inside = (theta_2 > -5.0) & (theta_2 < 5.0)", part: 8 },
    { text: "        outputs = torch.clone(theta_2) # Outside box stays unchanged", part: 8 },
    { text: "", part: 8 },
    { text: "        if inside.any():", part: 8 },
    { text: "            outputs[inside], det = rational_quadratic_spline(theta_2[inside])", part: 8 },
    { text: "", part: 8 },
    { text: "        # FINAL ASSEMBLY", part: 8 },
    { text: "        y_final = torch.cat([theta_1, outputs], dim=-1)", part: 8 },
    { text: "        return y_final, det.sum(dim=-1)", part: 8 },
  ];

  const partExplanations = [
    { title: "Zero-Initialization", exp: "The MLP must start out doing absolutely nothing so the data flows cleanly through the network early in training. We explicitly set the final layer weights to exactly 0.0." },
    { title: "Residual Connections", exp: "Deep MLPs suffer from vanishing gradients. By adding the input 'out' back onto the output of each block, we create a mathematical highway for stable gradients." },
    { title: "PyTorch Constraints", exp: "The raw outputs are unbound. F.softmax squishes Widths and Heights into percentages that perfectly sum to the box size. F.softplus forces derivatives to be strictly positive." },
    { title: "Knot Scaffolding", exp: "torch.cumsum runs a running total of the widths and heights, calculating the exact (X,Y) graph coordinate of every knot. We pad with zero to start at the bottom-left of the box." },
    { title: "Finding the Bin", exp: "torch.searchsorted acts as an ultra-fast Binary Search. It instantly figures out which bin our data point landed in, and torch.gather retrieves the correct W, H, and D for that specific bin." },
    { title: "The Forward Math", exp: "This is the mathematical cheat code! PyTorch uses basic arithmetic to divide the Top shape by the Bottom shape. It also runs the rigorous Calculus Quotient Rule to get the exact derivative for the Volume Penalty." },
    { title: "The Inverse Math", exp: "Because the equation is a fraction of two quadratics, PyTorch groups the algebra terms into a, b, and c. It then plugs them into the standard High School Quadratic Formula to instantly reverse the flow!" },
    { title: "The Splitting Orchestrator", exp: "The Coupling layer splits the data in half using tensor slicing. It then concatenates Half A and the SCADA condition and feeds them to the Brain to generate the Spline parameters." },
    { title: "Execution & Reassembly", exp: "It checks the Bounding Box: if the data is outside [-5, 5], it bypasses the math (linear tails). Otherwise, it evaluates the Spline. Finally, torch.cat stitches the untouched Half A and morphed Half B together." }
  ];

  useEffect(() => {
    if (lineRefs.current[activePart]) {
      lineRefs.current[activePart].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activePart]);

  return (
    <div className="relative w-full h-[600px] md:h-full bg-[#0d1117] overflow-hidden flex flex-col md:flex-row">
      
      {/* Left: Code Editor Panel */}
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
                  ? 'bg-emerald-500/20 border-emerald-500 text-emerald-200' 
                  : line.part !== null 
                    ? 'border-transparent opacity-50 hover:opacity-100 cursor-pointer' 
                    : 'border-transparent text-slate-500'
              }`}
              onClick={() => { if (line.part !== null) setActivePart(line.part) }}
              dangerouslySetInnerHTML={{__html: highlightCode(line.text) || ' '}}
            />
          ))}
        </div>
      </div>

      {/* Right (or Bottom on small screens): Explanation Panel */}
      <div className="w-full min-h-[180px] md:h-full md:w-80 lg:w-96 flex-shrink-0 bg-slate-800 md:border-l border-t md:border-t-0 border-slate-700 p-4 md:p-5 flex flex-col justify-between z-10 shadow-[0_-4px_15px_rgba(0,0,0,0.3)] md:shadow-[-4px_0_15px_rgba(0,0,0,0.3)] overflow-y-auto">
        <div className="animate-in fade-in slide-in-from-right-2 duration-300" key={activePart}>
           <div className="text-[10px] sm:text-xs font-bold text-emerald-400 mb-1.5 uppercase tracking-wider">
              Part {activePart + 1} of {partExplanations.length}
           </div>
           <h4 className="text-sm sm:text-base font-bold text-white mb-2">{partExplanations[activePart].title}</h4>
           <p className="text-[11px] sm:text-xs text-slate-300 leading-relaxed bg-slate-900/50 p-3 sm:p-4 rounded-lg border border-slate-700 shadow-inner">
             {partExplanations[activePart].exp}
           </p>
        </div>

        <div className="flex gap-2 justify-between mt-4 md:mt-6">
           <VisualButton 
             onClick={() => setActivePart(p => Math.max(0, p - 1))} 
             disabled={activePart === 0}
           >
             <ChevronLeft size={16}/> Prev
           </VisualButton>
           <VisualButton 
             onClick={() => setActivePart(p => Math.min(partExplanations.length - 1, p + 1))} 
             disabled={activePart === partExplanations.length - 1}
           >
             Next <ChevronRight size={16}/>
           </VisualButton>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// MAIN CONFIGURATION & COMPONENT
// ==========================================

const steps = [
  {
    id: 'mlp',
    title: '1. The Brain (Residual MLP)',
    icon: Cpu,
    codeSnippet: `class ResidualMLP(nn.Module):\n    def __init__(self, in_dim, hid_dim, out_dim):\n        super().__init__()\n        self.initial = nn.Linear(in_dim, hid_dim)\n        self.blocks = nn.ModuleList([...])\n        \n        # Output Layer: (3K - 1) params per dimension\n        self.final_layer = nn.Linear(hid_dim, out_dim)\n        \n        # CRITICAL: Zero Initialization\n        # Forces the flow to start as an Identity function\n        nn.init.zeros_(self.final_layer.weight)\n        nn.init.zeros_(self.final_layer.bias)\n\n    def forward(self, x):\n        out = self.initial(x)\n        for block in self.blocks:\n            out = block(out) + out # Skip connection\n        return self.final_layer(out)`,
    description: "The MLP takes the Condition array and outputs the raw W, H, and D parameters. It uses 'skip connections' to prevent vanishing gradients in deep networks, but its most critical feature is the Zero-Initialization of the final layer.",
    why: "If initialized with random weights, the MLP would output random slopes and widths. The Spline Math engine would bend randomly, violently distorting the SCADA data on the very first pass and instantly causing the loss to explode to NaN. Zero-init ensures the curve starts perfectly flat.",
    Visual: AnimatedResidualMLP
  },
  {
    id: 'zeroinit',
    title: '1b. Why Zero-Init Matters',
    icon: Waves,
    codeSnippet: `# WITHOUT zero-init (default PyTorch random weights):\n# final_layer outputs random logits, e.g.:\n#   W_logits = [2.3, -1.8, 4.1, ...]  <- random!\n#   softmax([2.3,-1.8,4.1]) = [0.18, 0.02, 0.80]\n# → one bin takes 80% of the box, two bins are tiny\n# → spline is wildly bent before ANY training\n# → loss = NaN on the very first forward pass 💥\n\n# WITH zero-init (our code):\nnn.init.zeros_(self.final_layer.weight)\nnn.init.zeros_(self.final_layer.bias)\n# final_layer outputs all zeros:\n#   W_logits = [0.0, 0.0, 0.0, 0.0]  <- flat\n#   softmax([0,0,0,0]) = [0.25, 0.25, 0.25, 0.25]\n# → all bins equal width → spline is a straight y=x line\n# → loss starts at a sensible value → training is stable ✓`,
    description: "Zero-initialization of the final MLP layer forces ALL output logits to 0.0 at the start of training. Because softmax([0,0,...,0]) = uniform distribution, every spline bin starts equally sized. The resulting spline is a perfectly straight diagonal line — the identity function y=x. The model passes data through completely unchanged until training finds a reason to bend the curve.",
    why: "Random PyTorch weights produce random logits. softmax of random logits creates unequal bin sizes. An unequal spline is bent. A bent spline on epoch 0 violently distorts every SCADA data point before a single gradient has been computed. The resulting loss is astronomically large or NaN, and the model never recovers.",
    Visual: AnimatedZeroInit
  },
  {
    id: 'constraints',
    title: '2. Enforcing Math Constraints',
    icon: Activity,
    codeSnippet: `# 1. Widths & Heights: Softmax -> Percentages\nwidths = F.softmax(unnorm_widths, dim=-1)\nheights = F.softmax(unnorm_heights, dim=-1)\n\n# Scale percentages by Total Box Size (e.g., 2 * 5.0)\nwidths = widths * (bound * 2.0)\nheights = heights * (bound * 2.0)\n\n# 2. Derivatives: Softplus -> Strictly Positive\nderivs = min_deriv + F.softplus(unnorm_derivs)\n\n# 3. Tails: Pad ends with 1.0 for linear tails\npad = torch.ones_like(derivs[..., :1])\nderivs = torch.cat([pad, derivs, pad], dim=-1)`,
    description: "Neural Networks output unbounded, raw numbers (logits). We must use PyTorch activation functions as physical constraints before passing them to the math engine. F.softmax ensures the Widths and Heights sum perfectly to the bounding box width. F.softplus ensures the slopes are strictly positive.",
    why: "Without Softmax, the predicted widths could sum to 15.0, leaking outside the [-5, 5] bounding box. Without Softplus, the curve could dip downwards (negative slope), destroying the 1-to-1 mathematical invertibility required to generate future scenarios.",
    Visual: AnimatedConstraints
  },
  {
    id: 'scaffolding',
    title: '3. Knot Scaffolding & Bin Search',
    icon: Search,
    codeSnippet: `# W and H both need cumsum → pad → map to [-bound, bound]\ncumwidths  = F.pad(torch.cumsum(widths,  dim=-1), (1,0), value=0.0)\ncumheights = F.pad(torch.cumsum(heights, dim=-1), (1,0), value=0.0)\ncumwidths  = (bound * 2.0) * cumwidths  / cumwidths[..., -1:]  - bound\ncumheights = (bound * 2.0) * cumheights / cumheights[..., -1:] - bound\n\n# D just gets 1.0 padded at both ends (for smooth linear tails)\npad = torch.ones_like(derivs[..., :1])\nderivs = torch.cat([pad, derivs, pad], dim=-1)  # shape: K+1\n\n# Find the bin: forward searches X-axis, inverse searches Y-axis\nif not inverse:\n    bin_idx = torch.searchsorted(cumwidths, inputs) - 1\nelse:\n    bin_idx = torch.searchsorted(cumheights, inputs) - 1\n\n# Gather ALL 4 values the algebra needs for that specific bin:\nW_k = torch.gather(widths,   -1, bin_idx)      # X bin width\nH_k = torch.gather(heights,  -1, bin_idx)      # Y bin height\nD_0 = torch.gather(derivs,   -1, bin_idx)      # left-knot slope\nD_1 = torch.gather(derivs,   -1, bin_idx + 1)  # right-knot slope`,
    description: "All three MLP outputs — W (bin widths), H (bin heights), and D (slopes) — are processed here. W and H each go through cumsum → pad → map to build the exact X-axis and Y-axis knot coordinates. D is simply padded with 1.0 at both ends to attach linear tails outside the box. Once the grid is built, torch.searchsorted does a fast binary search to find the bin, and torch.gather extracts W_k, H_k, D_0, and D_1 — the four values the algebra in Step 4 needs.",
    why: "W alone tells you where bins sit on the X-axis. H tells you where they sit on the Y-axis. Without H, you cannot compute the output coordinate or the straight-line slope S = H_k / W_k. Without D_0 and D_1, the spline curve has no smooth tangent constraints at the knot boundaries, breaking invertibility.",
    Visual: AnimatedScaffolding
  },
  {
    id: 'cumsum',
    title: '3b. Deep Dive: cumsum → pad → map',
    icon: ArrowDown,
    codeSnippet: `# STEP 1: cumsum converts bin SIZES → right-edge POSITIONS\n# W = [2.0, 3.0, 5.0]  (sizes — how wide each bin is)\ncumwidths = torch.cumsum(widths, dim=-1)\n# cumwidths = [2.0, 5.0, 10.0]  (right edges of each bin)\n\n# STEP 2: F.pad prepends 0.0 — adds the LEFT edge of Bin 0\n# (1,0) means "pad 1 on the left, 0 on the right"\ncumwidths = F.pad(cumwidths, (1, 0), value=0.0)\n# cumwidths = [0.0, 2.0, 5.0, 10.0]  ← now K+1 = 4 values\n#              ↑ new!  = left edge of Bin 0\n\n# STEP 3: map [0, 10] → [-bound, +bound]  e.g. [-5, 5]\ncumwidths = (bound * 2.0) * cumwidths / cumwidths[..., -1:] - bound\n# cumwidths = [-5.0, -3.0, 0.0, 5.0]  ← final X-knot positions\n\n# D is different — no cumsum, just pad with 1.0 at both ends:\n# D = [0.8, 1.2, 0.9]  →  [1.0, 0.8, 1.2, 0.9, 1.0]`,
    description: "The MLP outputs bin SIZES (how wide/tall each bin is), not positions. cumsum converts sizes → right-edge positions. F.pad adds the missing left boundary (0). The map re-scales from [0,10] to [-5,5]. This gives K+1 knot coordinates from K bin sizes.",
    why: "searchsorted needs absolute positions, not sizes. Without pad, Bin 0 has no left boundary. Without the map, knots live in [0,10] instead of the model's [-5,5] bounding box.",
    Visual: AnimatedCumsumExplainer
  },
  {
    id: 'coupling',
    title: '4. The Coupling Orchestrator',
    icon: Split,
    codeSnippet: `class NeuralSplineCouplingLayer(nn.Module):\n    def forward(self, theta, condition):\n        # 1. The Split\n        half = theta.shape[-1] // 2\n        t_1, t_2 = theta[:, :half], theta[:, half:]\n        \n        # 2. The Brain\n        brain_in = torch.cat([t_1, condition], dim=-1)\n        params = self.brain(brain_in)\n        \n        # 3. The Mask (Check Bounding Box)\n        inside = (t_2 > -bound) & (t_2 < bound)\n        outputs = torch.clone(t_2)\n        \n        # 4. The Math Engine (Only evaluate inside box)\n        if inside.any():\n            spline_out, det = rational_quadratic_spline(\n                t_2[inside], params[inside]\n            )\n            outputs[inside] = spline_out\n            \n        # 5. Reassembly\n        return torch.cat([t_1, outputs], dim=-1)`,
    description: "The Coupling Layer acts as the traffic controller. It uses tensor slicing to split the data. It passes Half A and the SCADA condition to the MLP, modifies Half B using the Spline math, and stitches them back together with torch.cat().",
    why: "Because Half A is never transformed by its own parameters, the Jacobian matrix becomes mathematically lower-triangular. This 'Jacobian Shortcut' allows PyTorch to skip massive matrix inversions and compute the volume penalty in real-time.",
    Visual: AnimatedCoupling
  },
  {
    id: 'walkthrough',
    title: '5. Full Code Walkthrough',
    icon: Terminal,
    codeSnippet: ``, // Not used for this step
    description: "A holistic line-by-line breakdown of components.py.",
    why: "",
    Visual: AnimatedWalkthrough
  }
];

export default function ComponentsSection() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  const handleNext = () => setCurrentStep(prev => Math.min(prev + 1, steps.length - 1));
  const handlePrev = () => setCurrentStep(prev => Math.max(prev - 1, 0));

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">
      
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
        <div className="p-3 bg-emerald-100 text-emerald-600 rounded-xl shadow-sm border border-emerald-200">
          <Layers className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800 tracking-tight">PyTorch Components Sandbox</h2>
          <p className="text-sm text-slate-500 font-medium">Interactive Architecture Walkthrough of <code className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-700 border border-slate-200">components.py</code></p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full flex gap-1.5 mb-6">
        {steps.map((s, idx) => (
          <div 
            key={s.id} 
            onClick={() => setCurrentStep(idx)}
            className={`h-2.5 flex-1 rounded-full cursor-pointer transition-all duration-300 ${
              idx === currentStep ? 'bg-emerald-600 scale-y-110 shadow-sm' : 
              idx < currentStep ? 'bg-emerald-300' : 'bg-slate-100 hover:bg-slate-200'
            }`}
            title={s.title}
          />
        ))}
      </div>

      {/* Main Content Area */}
      {step.id === 'walkthrough' ? (
        <div className="flex-1 flex flex-col gap-4 min-w-0 min-h-[500px]">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2.5 bg-slate-900 rounded-xl text-white shadow-md">
              <step.icon size={22} className="text-emerald-400" />
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
          </div>
          <div className="flex-1 w-full rounded-2xl shadow-xl shadow-slate-200/50 overflow-hidden relative border-4 border-slate-900/5 bg-[#0d1117]">
             <step.Visual />
          </div>
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[500px]">
          
          {/* Left Column: Visuals & Explanation */}
          <div className="lg:col-span-7 flex flex-col gap-6 min-w-0">
            {/* Visual Container */}
            <div className={`w-full rounded-2xl shadow-xl overflow-hidden relative border-4 border-slate-900/5 bg-[#0d1117] ${step.id === 'scaffolding' ? 'h-[480px]' : 'h-[360px]'}`}>
               <step.Visual />
            </div>

            {/* Explanation Container */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex-1 flex flex-col">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2.5 bg-emerald-100 rounded-xl text-emerald-600 shadow-sm border border-emerald-200">
                  <step.icon size={20} />
                </div>
                <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
              </div>
              
              <p className="text-slate-600 leading-relaxed text-[15px] mb-5">
                {step.description}
              </p>

              <div className="bg-amber-50/80 border border-amber-200 p-5 rounded-xl shadow-sm relative mt-auto">
                <div className="absolute top-0 left-0 w-1.5 h-full bg-amber-400"></div>
                <h4 className="text-sm font-bold text-amber-900 mb-2 flex items-center gap-2 uppercase tracking-wide">
                  <Activity size={16} className="text-amber-600"/>
                  The PyTorch Logic
                </h4>
                <p className="text-sm text-amber-800/90 leading-relaxed">{step.why}</p>
              </div>
            </div>
          </div>

          {/* Right Column: Code Editor */}
          <div className="lg:col-span-5 flex flex-col min-w-0 h-full">
            <div className="flex-1 bg-[#0f172a] rounded-2xl shadow-xl flex flex-col overflow-hidden border border-slate-700">
              <div className="bg-slate-800 px-4 py-3 border-b border-slate-700 flex items-center justify-between shadow-sm">
                <div className="flex items-center gap-2">
                  <Terminal size={14} className="text-emerald-400"/>
                  <span className="text-xs font-bold font-mono text-slate-300">components.py snippet</span>
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

      {/* Controls */}
      <div className="flex items-center justify-between mt-8 pt-6 border-t border-slate-200">
        <button 
          onClick={handlePrev}
          disabled={currentStep === 0}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
        >
          <ChevronLeft size={18} /> Previous Step
        </button>
        
        <button 
          onClick={handleNext}
          disabled={currentStep === steps.length - 1}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-white bg-emerald-600 hover:bg-emerald-700 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-40 disabled:hover:translate-y-0 disabled:cursor-not-allowed disabled:shadow-none"
        >
          {currentStep === steps.length - 1 ? 'Finish Walkthrough' : 'Next Step'} <ChevronRight size={18} />
        </button>
      </div>

    </div>
  );
}