import React, { useState, useEffect, useRef } from 'react';
import { 
  Layers, 
  ChevronRight, 
  ChevronLeft, 
  Activity,
  Cpu,
  Terminal,
  Play,
  Pause,
  ArrowRightLeft,
  Bell,
  Target,
  Zap,
  Repeat,
  Calculator
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
  html = html.replace(/('.*?'|".*?")/g, (m) => pushToken(m, "text-indigo-300"));

  // 3. Numbers
  html = html.replace(/\b(\d+\.\d+|\d+)\b/g, (m) => pushToken(m, "text-purple-300"));

  // 4. PyTorch functions & modules
  const pytorchRegex = /\b(nn\.Module|nn\.Linear|nn\.ModuleList|nn\.Sequential|MultivariateNormal|torch\.zeros|torch\.eye|torch\.flip|torch\.cat|layer\.inverse|loss\.mean|blueprint\.sample|blueprint\.log_prob)\b/g;
  html = html.replace(pytorchRegex, (m) => pushToken(m, "text-amber-300"));

  // 5. Python Keywords
  const kwRegex = /\b(def|class|if|else|elif|for|return|import|from|as|not|in|reversed|enumerate)\b/g;
  html = html.replace(kwRegex, (m) => pushToken(m, "text-rose-400 font-bold"));

  // 6. Self & special params
  const paramRegex = /\b(self|dim_theta|dim_condition|num_layers|hidden_dim|num_bins|bound|theta|condition|z|log_det)\b/g;
  html = html.replace(paramRegex, (m) => pushToken(m, "text-blue-300 italic"));

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
        ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/50' 
        : 'bg-slate-700 hover:bg-slate-600 text-slate-200 shadow-slate-900/50 border border-slate-600'
    }`}
  >
    {children}
  </button>
);

// ==========================================
// INTERACTIVE VISUAL COMPONENTS
// ==========================================

const AnimatedBlueprint = () => {
  return (
    <div className="relative w-full h-full bg-slate-900 flex flex-col items-center justify-center p-6 pb-14 gap-8">
      <div className="text-xs text-slate-400 font-mono text-center">
        The Blueprint: Registering the Grading Rubric
      </div>

      <div className="flex gap-8 w-full max-w-lg justify-center items-center mt-2">
        
        {/* GPU Buffers */}
        <div className="flex flex-col items-center gap-2">
           <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">GPU Buffers</span>
           <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 flex flex-col gap-3 shadow-lg w-40">
              <div className="flex flex-col gap-1">
                <span className="text-[9px] text-blue-400 font-mono">blueprint_loc (μ=0)</span>
                <div className="flex justify-between">
                  {Array.from({length: 4}).map((_,i) => <div key={i} className="w-5 h-5 bg-blue-900/50 border border-blue-500/50 rounded flex items-center justify-center text-[8px] text-blue-300 font-bold">0</div>)}
                </div>
              </div>
              <div className="flex flex-col gap-1 mt-2">
                <span className="text-[9px] text-emerald-400 font-mono">blueprint_cov (σ=1)</span>
                <div className="grid grid-cols-4 gap-1 w-full">
                  {Array.from({length: 16}).map((_,i) => {
                    const isDiag = Math.floor(i/4) === i%4;
                    return <div key={i} className={`w-full aspect-square rounded flex items-center justify-center text-[7px] font-bold ${isDiag ? 'bg-emerald-900/80 border border-emerald-500 text-emerald-300' : 'bg-slate-900 border border-slate-700 text-slate-600'}`}>{isDiag ? '1' : '0'}</div>
                  })}
                </div>
              </div>
           </div>
        </div>

        <ChevronRight size={24} className="text-slate-600" />

        {/* Multivariate Normal */}
        <div className="flex flex-col items-center gap-2">
           <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">MultivariateNormal</span>
           <div className="bg-slate-800 p-4 rounded-xl border-2 border-indigo-500 shadow-[0_0_20px_rgba(99,102,241,0.2)] w-40 h-40 flex items-center justify-center relative overflow-hidden">
              <div className="absolute inset-0 bg-indigo-500/10 z-0"></div>
              <svg viewBox="0 0 100 100" className="w-full h-full z-10 overflow-visible">
                 {/* Multi-dimensional bell curve rings */}
                 <ellipse cx="50" cy="50" rx="40" ry="20" fill="none" stroke="#818cf8" strokeWidth="1" className="opacity-20" />
                 <ellipse cx="50" cy="50" rx="30" ry="15" fill="none" stroke="#818cf8" strokeWidth="1.5" className="opacity-40" />
                 <ellipse cx="50" cy="50" rx="20" ry="10" fill="none" stroke="#818cf8" strokeWidth="2" className="opacity-60" />
                 <ellipse cx="50" cy="50" rx="10" ry="5" fill="#818cf8" className="opacity-80" />
                 <circle cx="50" cy="50" r="2" fill="#fff" className="shadow-[0_0_5px_#fff]" />
              </svg>
              <div className="absolute bottom-2 text-[9px] text-indigo-300 font-bold bg-slate-900/80 px-2 py-0.5 rounded border border-indigo-500/30">N(0, I) Target</div>
           </div>
        </div>

      </div>
    </div>
  );
};

const AnimatedForwardPass = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(s => (s >= 6 ? 0 : s + 1)), 1500);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  return (
    <div className="relative w-full h-full bg-slate-900 flex flex-col items-center justify-center p-6 pb-20 border border-slate-800 gap-4">
      <div className="text-xs text-slate-400 mb-2 font-mono text-center">The Forward Relay Race (Training)</div>
      
      <div className="flex items-center gap-1 w-full max-w-2xl px-4 mt-6">
        
        <div className="flex flex-col items-center w-12 flex-shrink-0">
          <span className="text-[10px] font-bold text-slate-400 mb-2">Input</span>
          <div className="w-8 h-12 flex flex-col gap-0.5 border border-slate-600 p-0.5 rounded bg-slate-800">
            <div className="w-full h-1/2 bg-blue-500/80 rounded-sm"></div>
            <div className="w-full h-1/2 bg-rose-500/80 rounded-sm"></div>
          </div>
          <span className="text-[10px] font-bold text-white mt-1">θ</span>
        </div>

        {Array.from({length: 6}).map((_, i) => {
          const isActive = step > i;
          const isCurrent = step === i;
          const flipState = i % 2 === 0; // Even layers have blue on top entering, odd have rose on top entering
          
          return (
            <React.Fragment key={i}>
              <div className="flex-1 flex flex-col items-center relative">
                {/* Connecting Line */}
                <div className="w-full h-0.5 bg-slate-700 absolute top-1/2 -translate-y-1/2 -z-10"></div>
                {isCurrent && <div className="w-1/2 h-0.5 bg-blue-400 absolute left-0 top-1/2 -translate-y-1/2 -z-10 animate-pulse"></div>}
                {isActive && <div className="w-full h-0.5 bg-blue-500 absolute top-1/2 -translate-y-1/2 -z-10 shadow-[0_0_5px_#3b82f6]"></div>}
                
                {/* Coupling Layer */}
                <div className={`w-8 h-12 flex flex-col justify-center items-center rounded border-2 transition-all duration-300 z-10 ${isActive ? 'bg-indigo-900 border-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.4)]' : isCurrent ? 'bg-slate-700 border-indigo-400 animate-pulse' : 'bg-slate-800 border-slate-600'}`}>
                  <span className={`text-[8px] font-bold ${isActive || isCurrent ? 'text-indigo-200' : 'text-slate-500'}`}>L{i+1}</span>
                </div>

                {/* Flip Indicator */}
                {i < 5 && (
                  <div className={`absolute -right-4 top-1/2 -translate-y-1/2 flex flex-col items-center transition-all duration-300 delay-300 ${isActive ? 'opacity-100 scale-100' : 'opacity-0 scale-50'}`}>
                    <ArrowRightLeft size={12} className="text-slate-400 rotate-90" />
                    <span className="text-[6px] text-slate-500 font-mono mt-0.5">flip</span>
                  </div>
                )}
              </div>
            </React.Fragment>
          )
        })}

        <div className="flex flex-col items-center w-12 flex-shrink-0 ml-4">
          <span className="text-[10px] font-bold text-slate-400 mb-2">Output</span>
          <div className={`w-8 h-12 flex flex-col gap-0.5 border-2 p-0.5 rounded transition-all duration-500 ${step >= 6 ? 'border-emerald-400 bg-emerald-900/50 shadow-[0_0_15px_rgba(52,211,153,0.3)]' : 'border-slate-600 bg-slate-800 opacity-30'}`}>
            <div className="w-full h-1/2 bg-blue-400 rounded-sm flex items-center justify-center"><span className="text-[6px] text-white">z₁</span></div>
            <div className="w-full h-1/2 bg-rose-400 rounded-sm flex items-center justify-center"><span className="text-[6px] text-white">z₂</span></div>
          </div>
          <span className="text-[10px] font-bold text-emerald-400 mt-1">Z</span>
        </div>

      </div>

      <div className="mt-8 flex flex-col items-center w-full max-w-md bg-slate-800/80 p-3 rounded-lg border border-slate-600">
        <span className="text-[10px] text-slate-400 font-bold mb-1">Accumulated Volume Penalty</span>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-white">total_log_det =</span>
          <div className="flex gap-1">
             {Array.from({length: 6}).map((_, i) => (
               <span key={i} className={`text-xs font-mono transition-colors duration-300 ${step > i ? 'text-teal-400' : 'text-slate-600'}`}>
                 log_det_{i+1} {i < 5 ? '+' : ''}
               </span>
             ))}
          </div>
        </div>
      </div>

      {/* Playback Controls */}
      <div className="absolute bottom-4 flex gap-4">
        <VisualButton onClick={() => setStep(s => Math.max(0, s - 1))} active={false} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev Layer
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? "Pause Flow" : "Flow Forward"}
        </VisualButton>
        <VisualButton onClick={() => setStep(s => Math.min(6, s + 1))} active={true} disabled={step >= 6 || isPlaying}>
          Next Layer <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedLoss = () => {
  return (
    <div className="relative w-full h-full bg-slate-900 flex flex-col items-center justify-center p-6 pb-14 gap-8">
      <div className="text-xs text-slate-400 font-mono text-center">
        Evaluating the Final Loss
      </div>

      <div className="flex gap-8 w-full max-w-lg justify-center items-center mt-2">
        
        {/* Score Component */}
        <div className="flex flex-col items-center gap-2 flex-1">
           <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Blueprint Score</span>
           <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col items-center justify-center w-full h-24 relative shadow-inner">
              <svg viewBox="0 0 100 50" className="w-full h-full absolute inset-0 overflow-visible">
                 <path d="M 10 50 C 30 50, 40 10, 50 10 C 60 10, 70 50, 90 50" fill="rgba(99, 102, 241, 0.2)" stroke="#818cf8" strokeWidth="2" />
                 {/* Z Landing Spot */}
                 <circle cx="50" cy="10" r="3" fill="#fff" className="shadow-[0_0_8px_#fff] animate-pulse" />
                 <line x1="50" y1="10" x2="50" y2="50" stroke="#fff" strokeWidth="1" strokeDasharray="2" />
              </svg>
              <div className="absolute top-2 right-2 text-[9px] bg-indigo-900/80 text-indigo-300 px-1.5 py-0.5 rounded border border-indigo-500/50">High Score!</div>
           </div>
           <div className="mt-2 text-[10px] font-mono font-bold text-indigo-300">
              log_prob(Z_final)
           </div>
        </div>

        <div className="text-xl font-bold text-slate-500">+</div>

        {/* Penalty Component */}
        <div className="flex flex-col items-center gap-2 flex-1">
           <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Volume Penalty</span>
           <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col items-center justify-center w-full h-24 relative shadow-inner">
              <svg viewBox="0 0 100 50" className="w-full h-full absolute inset-0 overflow-visible">
                 <path d="M 0 50 L 10 20 L 30 45 L 50 10 L 70 30 L 90 5 L 100 50" fill="rgba(20, 184, 166, 0.2)" stroke="#2dd4bf" strokeWidth="2" />
              </svg>
              <div className="absolute bottom-2 text-[9px] bg-teal-900/80 text-teal-300 px-1.5 py-0.5 rounded border border-teal-500/50">Sum of Slopes</div>
           </div>
           <div className="mt-2 text-[10px] font-mono font-bold text-teal-300">
              total_log_det
           </div>
        </div>

      </div>

      <div className="w-full max-w-md bg-rose-950/40 border-2 border-rose-500/80 p-4 rounded-xl shadow-[0_0_20px_rgba(225,29,72,0.15)] flex flex-col items-center justify-center mt-2">
         <span className="text-white font-mono text-sm font-bold">loss = -1 * (Score + Penalty)</span>
         <span className="text-[10px] text-rose-300 mt-2 text-center leading-relaxed">
            Neural Networks always minimize to zero. We want to MAXIMIZE the log-likelihood (Score + Penalty), so we multiply by -1 before calling <code>loss.backward()</code>.
         </span>
      </div>

    </div>
  );
};

const AnimatedSample = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let int;
    if (isPlaying) {
      int = setInterval(() => setStep(s => (s >= 4 ? 0 : s + 1)), 2000);
    }
    return () => clearInterval(int);
  }, [isPlaying]);

  return (
    <div className="relative w-full h-full bg-slate-900 flex flex-col items-center justify-center p-6 pb-20 border border-slate-800 gap-4">
      <div className="text-xs text-slate-400 mb-2 font-mono text-center">The Reverse Relay Race (Generation)</div>
      
      <div className="flex items-center gap-2 w-full max-w-2xl px-4 mt-6">
        
        {/* Step 0: Sample Noise */}
        <div className={`flex flex-col items-center w-16 flex-shrink-0 transition-all duration-500 ${step >= 0 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
          <span className="text-[10px] font-bold text-indigo-400 mb-2">1. Sample Z</span>
          <div className="w-12 h-12 flex items-center justify-center border-2 border-indigo-500 rounded bg-indigo-900/30 shadow-[0_0_15px_rgba(99,102,241,0.3)] relative overflow-hidden">
             <div className="absolute w-1 h-1 bg-white rounded-full top-2 left-3"></div>
             <div className="absolute w-1.5 h-1.5 bg-white rounded-full top-5 left-5 shadow-[0_0_5px_#fff]"></div>
             <div className="absolute w-1 h-1 bg-white rounded-full bottom-3 right-3"></div>
             <div className="absolute w-0.5 h-0.5 bg-white rounded-full top-8 right-2"></div>
          </div>
          <span className="text-[8px] font-bold text-slate-400 mt-1 text-center">N(0,1)<br/>Noise</span>
        </div>

        <ArrowLeftIcon active={step >= 1} />

        {/* Step 1: Reversed Loop */}
        <div className={`flex-1 flex flex-col items-center transition-all duration-500 ${step >= 1 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
           <span className="text-[10px] font-bold text-slate-300 mb-2 font-mono">for layer in reversed()</span>
           <div className="bg-slate-800 border border-slate-600 p-3 rounded-xl flex gap-2 shadow-inner w-full justify-center">
              <div className="w-6 h-8 bg-slate-700 rounded border border-slate-500 flex items-center justify-center"><span className="text-[8px] text-slate-400">L6</span></div>
              <div className="w-6 h-8 bg-slate-700 rounded border border-slate-500 flex items-center justify-center"><span className="text-[8px] text-slate-400">...</span></div>
              <div className="w-6 h-8 bg-slate-700 rounded border border-slate-500 flex items-center justify-center"><span className="text-[8px] text-slate-400">L1</span></div>
           </div>
        </div>

        <ArrowLeftIcon active={step >= 2} />

        {/* Step 2: Flip Undo */}
        <div className={`flex flex-col items-center w-16 transition-all duration-500 ${step >= 2 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
           <span className="text-[10px] font-bold text-amber-400 mb-2">2. Undo Flip</span>
           <div className="bg-amber-900/30 border border-amber-500 p-2 rounded-lg flex flex-col items-center shadow-[0_0_10px_rgba(245,158,11,0.2)]">
              <ArrowRightLeft size={16} className="text-amber-400 rotate-90 mb-1" />
              <span className="text-[8px] text-amber-200">dim=-1</span>
           </div>
        </div>

        <ArrowLeftIcon active={step >= 3} />

        {/* Step 3: Layer Inverse */}
        <div className={`flex flex-col items-center w-16 transition-all duration-500 ${step >= 3 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
           <span className="text-[10px] font-bold text-teal-400 mb-2">3. Math Inv</span>
           <div className="bg-teal-900/30 border border-teal-500 p-2 rounded-lg flex flex-col items-center shadow-[0_0_10px_rgba(20,184,166,0.2)]">
              <Calculator size={16} className="text-teal-400 mb-1" />
              <span className="text-[8px] text-teal-200">Quadratic</span>
           </div>
        </div>

        <ArrowLeftIcon active={step >= 4} />

        {/* Output */}
        <div className={`flex flex-col items-center w-16 flex-shrink-0 transition-all duration-500 ${step >= 4 ? 'opacity-100 scale-110' : 'opacity-30 scale-90'}`}>
          <span className="text-[10px] font-bold text-rose-400 mb-2">4. Output θ</span>
          <div className="w-12 h-12 flex items-center justify-center border-2 border-rose-500 rounded bg-rose-900/30 shadow-[0_0_20px_rgba(225,29,72,0.4)] relative overflow-hidden p-1">
             <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
               <path d="M0,80 Q25,20 50,70 T100,10" fill="none" stroke="#f43f5e" strokeWidth="6" className="drop-shadow-[0_0_3px_#f43f5e]" />
             </svg>
          </div>
          <span className="text-[8px] font-bold text-slate-400 mt-1 text-center">Simulated<br/>SCADA</span>
        </div>

      </div>

      <div className="text-center text-[11px] text-slate-400 mt-8 max-w-[90%] leading-relaxed h-12">
        {step === 0 && "1. We start by sampling pure, random Gaussian noise (Z) from our N(0,1) Blueprint."}
        {step === 1 && "2. We loop through all the trained coupling layers in REVERSE order."}
        {step === 2 && "3. CRITICAL: We must undo the array flip first so the dimensions align with what the layer expects."}
        {step === 3 && "4. We execute layer.inverse(), leveraging the fast Algebraic Quadratic Formula to undo the Spline."}
        {step === 4 && "5. After all layers, the noise has been perfectly morphed into a hyper-realistic future SCADA trajectory (θ)!"}
      </div>

      {/* Playback Controls */}
      <div className="absolute bottom-4 flex gap-4">
        <VisualButton onClick={() => setStep(s => Math.max(0, s - 1))} active={false} disabled={step === 0 || isPlaying}>
          <ChevronLeft size={14} /> Prev Step
        </VisualButton>
        <VisualButton onClick={() => setIsPlaying(!isPlaying)} active={isPlaying}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? "Pause" : "Play Process"}
        </VisualButton>
        <VisualButton onClick={() => setStep(s => Math.min(4, s + 1))} active={true} disabled={step >= 4 || isPlaying}>
          Next Step <ChevronRight size={14} />
        </VisualButton>
      </div>
    </div>
  );
};

const ArrowLeftIcon = ({ active }) => (
  <div className={`flex items-center transition-opacity duration-300 ${active ? 'opacity-100' : 'opacity-20'}`}>
     <div className="w-4 h-0.5 bg-slate-600"></div>
     <ChevronLeft size={16} className="text-slate-500 -ml-2" />
  </div>
);


const AnimatedWalkthrough = () => {
  const [activePart, setActivePart] = useState(0);
  const lineRefs = useRef({});

  const codeLines = [
    { text: "import torch", part: null },
    { text: "import torch.nn as nn", part: null },
    { text: "from torch.distributions import MultivariateNormal", part: null },
    { text: "from .components import ResidualMLP, NeuralSplineCouplingLayer", part: null },
    { text: "", part: null },
    { text: "class PipelineConditionalFlow(nn.Module):", part: 0 },
    { text: "    def __init__(self, dim_theta, dim_condition, num_layers=6, hidden_dim=128, num_bins=8, bound=5.0):", part: 0 },
    { text: "        super().__init__()", part: 0 },
    { text: "", part: 0 },
    { text: "        # The Grading Rubric (N(0,1))", part: 0 },
    { text: "        self.register_buffer('blueprint_loc', torch.zeros(dim_theta))", part: 0 },
    { text: "        self.register_buffer('blueprint_cov', torch.eye(dim_theta))", part: 0 },
    { text: "", part: 0 },
    { text: "        # The Spline Layers", part: 0 },
    { text: "        self.layers = nn.ModuleList([", part: 0 },
    { text: "            NeuralSplineCouplingLayer(dim_theta, dim_condition, hidden_dim, num_bins, bound)", part: 0 },
    { text: "            for _ in range(num_layers)", part: 0 },
    { text: "        ])", part: 0 },
    { text: "", part: null },
    { text: "    def get_blueprint(self):", part: 0 },
    { text: "        return MultivariateNormal(self.blueprint_loc, self.blueprint_cov)", part: 0 },
    { text: "", part: null },
    { text: "    def forward(self, theta, condition):", part: 1 },
    { text: "        total_log_det = 0", part: 1 },
    { text: "        z = theta", part: 1 },
    { text: "", part: 1 },
    { text: "        for i, layer in enumerate(self.layers):", part: 1 },
    { text: "            z, log_det = layer(z, condition)", part: 1 },
    { text: "            total_log_det += log_det", part: 1 },
    { text: "", part: 1 },
    { text: "            # THE SWAP: Alternating arrays between layers", part: 1 },
    { text: "            z = torch.flip(z, dims=[-1])", part: 1 },
    { text: "", part: 1 },
    { text: "        return z, total_log_det", part: 1 },
    { text: "", part: null },
    { text: "    def compute_loss(self, theta, condition):", part: 2 },
    { text: "        # 1. Forward Pass", part: 2 },
    { text: "        z_final, total_volume_penalty = self.forward(theta, condition)", part: 2 },
    { text: "", part: 2 },
    { text: "        # 2. Evaluate Blueprint Score", part: 2 },
    { text: "        blueprint = self.get_blueprint()", part: 2 },
    { text: "        blueprint_score = blueprint.log_prob(z_final)", part: 2 },
    { text: "", part: 2 },
    { text: "        # 3. Total Loss Objective", part: 2 },
    { text: "        loss = -1 * (blueprint_score + total_volume_penalty)", part: 2 },
    { text: "        return loss.mean()", part: 2 },
    { text: "", part: null },
    { text: "    def sample(self, num_samples, condition):", part: 3 },
    { text: "        batch_size = condition.shape[0]", part: 3 },
    { text: "", part: 3 },
    { text: "        # Start with Noise", part: 3 },
    { text: "        blueprint = self.get_blueprint()", part: 3 },
    { text: "        z = blueprint.sample((batch_size,))", part: 3 },
    { text: "", part: 3 },
    { text: "        # Reversed Loop", part: 3 },
    { text: "        for layer in reversed(self.layers):", part: 3 },
    { text: "            # Undo the array flip first!", part: 3 },
    { text: "            z = torch.flip(z, dims=[-1])", part: 3 },
    { text: "            z = layer.inverse(z, condition)", part: 3 },
    { text: "", part: 3 },
    { text: "        return z", part: 3 },
  ];

  const partExplanations = [
    { title: "Initialization & Rubric", exp: "We assemble the multi-layer pipeline here. We also use PyTorch's 'register_buffer' to define our N(0,1) grading rubric. This ensures the rubric moves to the GPU automatically, but does NOT receive gradient updates during training." },
    { title: "The Forward Pass", exp: "We loop through the layers, passing data and accumulating the volume penalties (total_log_det). Critically, we call torch.flip() after each layer to swap Half A and Half B, ensuring all variables get bent by the splines." },
    { title: "Computing the Loss", exp: "We push the SCADA data through the flow to get Z_final. We grade Z_final against the bell curve to get the 'blueprint_score'. Finally, we sum the score and the penalty, and multiply by -1 because PyTorch optimizers minimize to zero." },
    { title: "Inverse Pass (Sampling)", exp: "To generate a future scenario, we sample pure noise from the blueprint. We loop through the layers in reverse. We MUST call torch.flip() first to undo the swap, then we call layer.inverse() to run the fast Quadratic Spline algebra." }
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
                  ? 'bg-blue-500/20 border-blue-500 text-blue-200' 
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
           <div className="text-[10px] sm:text-xs font-bold text-blue-400 mb-1.5 uppercase tracking-wider">
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
    id: 'blueprint',
    title: '1. The Grading Rubric',
    icon: Bell,
    codeSnippet: `self.register_buffer('blueprint_loc', torch.zeros(dim_theta))\nself.register_buffer('blueprint_cov', torch.eye(dim_theta))\n\ndef get_blueprint(self):\n    return MultivariateNormal(self.blueprint_loc, self.blueprint_cov)`,
    description: "The Flow requires a fixed Latent Space rubric to grade the incoming data against. We define a Standard Normal distribution (Bell Curve) with a mean of 0 and a variance of 1.",
    why: "We use PyTorch's 'register_buffer' instead of defining it dynamically. This ensures the rubric parameters (0 and 1) are physically pushed to the GPU memory alongside the model weights, but crucially, it prevents the optimizer from trying to 'train' or alter the rubric during backpropagation.",
    Visual: AnimatedBlueprint
  },
  {
    id: 'forward',
    title: '2. The Forward Relay Race',
    icon: Layers,
    codeSnippet: `total_log_det = 0\nz = theta\n\nfor layer in self.layers:\n    z, log_det = layer(z, condition)\n    total_log_det += log_det\n    z = torch.flip(z, dims=[-1])`,
    description: "During training, raw SCADA data is passed through each Spline Coupling Layer sequentially. We accumulate the 'log_det' (the sum of the spline slopes) at each step to build our total volume penalty. Critically, we physically flip the tensor array between each layer.",
    why: "A single coupling layer only transforms Half B of the data. By calling torch.flip() between layers, Half A becomes Half B, ensuring that by the end of the multi-layer relay race, every single dimension of the SCADA data has been warped by the splines.",
    Visual: AnimatedForwardPass
  },
  {
    id: 'loss',
    title: '3. Evaluating the Loss',
    icon: Target,
    codeSnippet: `z_final, penalty = self.forward(theta, condition)\n\nblueprint = self.get_blueprint()\nscore = blueprint.log_prob(z_final)\n\nloss = -1 * (score + penalty)\nreturn loss.mean()`,
    description: "Once the data completes the forward pass, it emerges as Z_final. We plug Z_final into the Bell Curve's probability formula to get a 'Score'. We add the accumulated Spline 'Penalty' from the forward pass, flip the sign, and return it to the optimizer.",
    why: "In probability theory, we want to MAXIMIZE the log-likelihood of our data fitting the bell curve. However, deep learning optimizers (like AdamW) are built to MINIMIZE a number to zero. Multiplying by -1 neatly converts our Maximum Likelihood math into a standard Minimum Loss objective.",
    Visual: AnimatedLoss
  },
  {
    id: 'sample',
    title: '4. The Inverse Generator',
    icon: Repeat,
    codeSnippet: `z = blueprint.sample((batch_size,))\n\nfor layer in reversed(self.layers):\n    z = torch.flip(z, dims=[-1])\n    z = layer.inverse(z, condition)\n\nreturn z`,
    description: "To generate realistic future SCADA scenarios for edge hardware, we reverse the entire flow. We sample random noise from the Bell Curve, loop through the layers backwards, undo the array flip, and execute the Algebraic Quadratic Formula to morph the noise back into physical data.",
    why: "The order of operations is vital. In the forward pass, we transformed the data THEN flipped it. Therefore, when moving backwards through time, we must UNDO the flip first, then UNDO the transformation. Failing to flip first would pass the wrong array halves into the math engine, creating garbage outputs.",
    Visual: AnimatedSample
  },
  {
    id: 'walkthrough',
    title: '5. Full Code Walkthrough',
    icon: Terminal,
    codeSnippet: ``, // Not used
    description: "A holistic line-by-line breakdown of flow_model.py.",
    why: "",
    Visual: AnimatedWalkthrough
  }
];

export default function FlowModelSection() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  const handleNext = () => setCurrentStep(prev => Math.min(prev + 1, steps.length - 1));
  const handlePrev = () => setCurrentStep(prev => Math.max(prev - 1, 0));

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">
      
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
        <div className="p-3 bg-blue-100 text-blue-600 rounded-xl shadow-sm border border-blue-200">
          <Zap className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800 tracking-tight">Flow Model Wrapper</h2>
          <p className="text-sm text-slate-500 font-medium">Interactive Architecture Walkthrough of <code className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-700 border border-slate-200">flow_model.py</code></p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full flex gap-1.5 mb-6">
        {steps.map((s, idx) => (
          <div 
            key={s.id} 
            onClick={() => setCurrentStep(idx)}
            className={`h-2.5 flex-1 rounded-full cursor-pointer transition-all duration-300 ${
              idx === currentStep ? 'bg-blue-600 scale-y-110 shadow-sm' : 
              idx < currentStep ? 'bg-blue-300' : 'bg-slate-100 hover:bg-slate-200'
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
              <step.icon size={22} className="text-blue-400" />
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
            <div className="w-full h-[360px] rounded-2xl shadow-xl overflow-hidden relative border-4 border-slate-900/5 bg-[#0d1117]">
               <step.Visual />
            </div>

            {/* Explanation Container */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex-1 flex flex-col">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2.5 bg-blue-100 rounded-xl text-blue-600 shadow-sm border border-blue-200">
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
                  <Terminal size={14} className="text-blue-400"/>
                  <span className="text-xs font-bold font-mono text-slate-300">flow_model.py snippet</span>
                </div>
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-rose-500/80"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80"></div>
                  <div className="w-2.5 h-2.5 rounded-full bg-blue-500/80"></div>
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
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-white bg-blue-600 hover:bg-blue-700 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-40 disabled:hover:translate-y-0 disabled:cursor-not-allowed disabled:shadow-none"
        >
          {currentStep === steps.length - 1 ? 'Finish Walkthrough' : 'Next Step'} <ChevronRight size={18} />
        </button>
      </div>

    </div>
  );
}