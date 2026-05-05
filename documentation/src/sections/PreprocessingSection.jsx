import React, { useState, useEffect } from 'react';
import { 
  Layers, 
  ChevronRight, 
  ChevronLeft, 
  Activity,
  Scissors,
  Scale,
  Maximize,
  Minimize2,
  FileCode2,
  AlertTriangle,
  CheckCircle,
  Play,
  RefreshCw,
  Search,
  Save,
  Package,
  Database,
  FileSearch
} from 'lucide-react';

// ==========================================
// INTERACTIVE "GIF-LIKE" VISUAL COMPONENTS
// ==========================================

const VisualButton = ({ onClick, disabled, children, active }) => (
  <button 
    onClick={onClick} 
    disabled={disabled}
    className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-bold transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 z-10 ${
      active 
        ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/50' 
        : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/50'
    }`}
  >
    {children}
  </button>
);

const AnimatedRawData = () => {
  const [offset, setOffset] = useState(0);
  const [scanned, setScanned] = useState(false);
  
  useEffect(() => {
    const int = setInterval(() => setOffset(prev => (prev - 1) % 100), 50);
    return () => clearInterval(int);
  }, []);

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 p-4 pb-12">
      <div className={`absolute top-4 left-4 flex gap-2 transition-opacity duration-300 ${scanned ? 'opacity-100' : 'opacity-0'}`}>
         <span className="bg-red-500/20 text-red-400 text-xs px-2 py-1 rounded flex items-center gap-1 border border-red-500/30">
           <AlertTriangle size={12}/> Missing Gaps (NaN)
         </span>
         <span className="bg-orange-500/20 text-orange-400 text-xs px-2 py-1 rounded flex items-center gap-1 border border-orange-500/30">
           <AlertTriangle size={12}/> Extreme Noise
         </span>
      </div>
      <svg viewBox="0 0 200 100" className="w-full h-32 mt-6">
        <defs>
          <linearGradient id="fade" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="20%" stopColor="#ef4444" />
            <stop offset="80%" stopColor="#ef4444" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
        </defs>
        {/* Messy signal with gaps */}
        <path 
          d="M0,50 L20,30 L30,80 L40,40 M60,60 L70,20 L80,90 L90,50 L100,50 M120,40 L130,60 L140,10 L150,70 L160,50 M180,30 L190,80 L200,40" 
          fill="none" 
          stroke="url(#fade)" 
          strokeWidth="1.5" 
          strokeDasharray="200"
          strokeDashoffset={offset}
        />
        {/* Highlight Gap Boxes */}
        <g className={`transition-opacity duration-300 ${scanned ? 'opacity-100' : 'opacity-0'}`}>
          <rect x="40" y="20" width="20" height="60" fill="#ef4444" fillOpacity="0.2" stroke="#ef4444" strokeDasharray="2,2" />
          <rect x="100" y="20" width="20" height="60" fill="#ef4444" fillOpacity="0.2" stroke="#ef4444" strokeDasharray="2,2" />
          <rect x="160" y="20" width="20" height="60" fill="#ef4444" fillOpacity="0.2" stroke="#ef4444" strokeDasharray="2,2" />
        </g>
      </svg>
      <div className="text-xs text-slate-500 mt-2 font-mono">Raw telemetry streaming from SCADA...</div>
      
      <div className="absolute bottom-3 right-3">
        <VisualButton onClick={() => setScanned(!scanned)} active={scanned}>
          <Search size={14} /> {scanned ? "Hide Anomalies" : "Scan Anomalies"}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedCleaning = () => {
  const [noise, setNoise] = useState([]);
  const [isFilled, setIsFilled] = useState(false);
  const [isNoised, setIsNoised] = useState(false);

  useEffect(() => {
    let int;
    if (isNoised) {
      int = setInterval(() => {
        setNoise(Array.from({ length: 20 }, () => (Math.random() - 0.5) * 4));
      }, 100);
    } else {
      setNoise(Array.from({ length: 20 }, () => 0));
    }
    return () => clearInterval(int);
  }, [isNoised]);

  const handleReset = () => {
    setIsFilled(false);
    setIsNoised(false);
  };

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 p-4 pb-14 gap-6">
      
      {/* Gap Filling Demo */}
      <div className="w-full flex items-center justify-between px-8">
        <div className="text-xs text-slate-400 w-24">1. Forward Fill</div>
        <svg viewBox="0 0 100 20" className="flex-1 h-8 mx-4">
          <path d="M0,10 L30,10" stroke="#475569" strokeWidth="2" fill="none" />
          {!isFilled && <path d="M30,10 L70,10" stroke="#ef4444" strokeWidth="2" strokeDasharray="2,2" fill="none" />}
          {isFilled && <path d="M30,10 L70,10" stroke="#10b981" strokeWidth="2" fill="none" className="drop-shadow-[0_0_5px_#10b981] transition-all duration-500" />}
          <path d="M70,10 L100,10" stroke="#475569" strokeWidth="2" fill="none" />
        </svg>
        <CheckCircle size={16} className={`transition-colors ${isFilled ? 'text-emerald-500' : 'text-slate-600'}`} />
      </div>

      {/* Micro-noise Demo */}
      <div className="w-full flex items-center justify-between px-8">
        <div className="text-xs text-slate-400 w-24">2. Micro-Noise (Anti-Flatline)</div>
        <svg viewBox="0 0 100 20" className="flex-1 h-8 mx-4">
          {!isNoised ? (
            <path d="M0,10 L100,10" stroke="#ef4444" strokeWidth="2" fill="none" />
          ) : (
            <path 
              d={`M0,10 ${noise.map((n, i) => `L${i * 5 + 5},${10 + n}`).join(' ')}`} 
              stroke="#8b5cf6" 
              strokeWidth="1.5" 
              fill="none" 
              className="drop-shadow-[0_0_5px_#8b5cf6] transition-all duration-75"
            />
          )}
        </svg>
        <CheckCircle size={16} className={`transition-colors ${isNoised ? 'text-purple-500' : 'text-slate-600'}`} />
      </div>

      <div className="absolute bottom-3 right-3 flex gap-2">
        {(isFilled || isNoised) && (
          <button onClick={handleReset} className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 px-2">
            <RefreshCw size={12} /> Reset
          </button>
        )}
        <VisualButton onClick={() => isFilled ? setIsNoised(true) : setIsFilled(true)} disabled={isFilled && isNoised}>
          <Play size={14} /> {isFilled ? "2. Add Noise" : "1. Fill Gaps"}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedSplit = () => {
  const [progress, setProgress] = useState(0);
  const [isSplitting, setIsSplitting] = useState(false);

  const handleSplit = () => {
    if (progress >= 100) {
      setProgress(0);
      return;
    }
    setIsSplitting(true);
    setProgress(0);
    const int = setInterval(() => {
      setProgress(p => {
        if (p >= 100) {
          clearInterval(int);
          setIsSplitting(false);
          return 100;
        }
        return p + 2;
      });
    }, 20);
  };

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden flex flex-col items-center justify-center p-6 pb-14 border border-slate-800 gap-4">
      <div className="text-xs text-slate-400 mb-2">Chronological Data Stream</div>
      
      {/* Stream */}
      <div className="w-full h-4 bg-slate-800 rounded-full overflow-hidden relative">
        <div 
          className="absolute top-0 left-0 h-full bg-slate-500 transition-all duration-75"
          style={{ width: `${progress}%` }}
        />
        {/* Scissor lines */}
        <div className="absolute top-0 left-[70%] h-full w-1 bg-white/50 z-10" />
        <div className="absolute top-0 left-[85%] h-full w-1 bg-white/50 z-10" />
      </div>

      {/* Buckets */}
      <div className="flex w-full gap-2 mt-4 h-24 text-xs font-bold">
        <div className="flex-1 bg-blue-900/40 border-2 border-blue-600/50 rounded-lg flex flex-col justify-end overflow-hidden relative">
          <div className="absolute top-2 text-center w-full text-blue-400">TRAIN 70%</div>
          <div className="bg-blue-600 w-full transition-all duration-75" style={{ height: `${Math.min(progress, 70) * (100/70)}%` }} />
        </div>
        <div className="w-1/5 bg-amber-900/40 border-2 border-amber-500/50 rounded-lg flex flex-col justify-end overflow-hidden relative">
          <div className="absolute top-2 text-center w-full text-amber-400">VAL 15%</div>
          <div className="bg-amber-500 w-full transition-all duration-75" style={{ height: `${Math.max(0, Math.min(progress - 70, 15)) * (100/15)}%` }} />
        </div>
        <div className="w-1/5 bg-emerald-900/40 border-2 border-emerald-500/50 rounded-lg flex flex-col justify-end overflow-hidden relative">
          <div className="absolute top-2 text-center w-full text-emerald-400">TEST 15%</div>
          <div className="bg-emerald-500 w-full transition-all duration-75" style={{ height: `${Math.max(0, Math.min(progress - 85, 15)) * (100/15)}%` }} />
        </div>
      </div>

      <div className="absolute bottom-3 right-3">
        <VisualButton onClick={handleSplit} disabled={isSplitting} active={progress >= 100}>
          {progress >= 100 ? <><RefreshCw size={14}/> Reset</> : <><Scissors size={14}/> Execute Split</>}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedScale = () => {
  const [view, setView] = useState('inputs'); // 'inputs' | 'targets'
  const [isScaled, setIsScaled] = useState(false);

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 p-6 pb-14 pt-16">
      
      {/* Toggle View Header */}
      <div className="absolute top-4 w-full flex justify-center gap-2 px-6">
        <button 
          onClick={() => { setView('inputs'); setIsScaled(false); }}
          className={`flex-1 py-1.5 text-xs font-bold rounded-lg transition-colors border ${view === 'inputs' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'}`}
        >
          X & U (Inputs)
        </button>
        <button 
          onClick={() => { setView('targets'); setIsScaled(false); }}
          className={`flex-1 py-1.5 text-xs font-bold rounded-lg transition-colors border ${view === 'targets' ? 'bg-emerald-600 border-emerald-500 text-white' : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'}`}
        >
          Theta (Targets)
        </button>
      </div>

      {view === 'inputs' ? (
        <div className="flex flex-col items-center w-full mt-2 animate-in fade-in zoom-in-95 duration-300">
          <div className="flex w-full justify-between mb-4">
            <div className={`text-xs px-3 py-1 rounded font-bold transition-colors ${!isScaled ? 'bg-slate-700 text-slate-300' : 'text-slate-600'}`}>Raw Pressures / Speeds</div>
            <div className={`text-xs px-3 py-1 rounded font-bold transition-colors ${isScaled ? 'bg-blue-500/20 text-blue-400' : 'text-slate-600'}`}>StandardScaler (Z-Score)</div>
          </div>

          <div className="relative w-full h-32 border-b border-l border-slate-700 flex items-end justify-around pb-2">
            <div className={`absolute w-full border-t border-dashed border-blue-500/50 transition-all duration-700 ${isScaled ? 'top-16 opacity-100' : 'top-16 opacity-0'}`} />
            <div className={`absolute right-2 text-[10px] text-blue-400 transition-all duration-700 ${isScaled ? 'top-12 opacity-100' : 'top-12 opacity-0'}`}>Mean = 0</div>

            {/* Well-behaved data points */}
            {[50, 65, 45, 55, 70, 40].map((h, i) => (
              <div 
                key={i} 
                className="w-8 bg-blue-500 rounded-t transition-all duration-700" 
                style={{ 
                  height: isScaled ? `${(h/100) * 80}%` : `${h/2 + 20}%`, 
                  transform: isScaled ? `translateY(-${(h < 55 ? 20 : 0)}px)` : 'translateY(0)' 
                }} 
              />
            ))}
          </div>
          <div className="text-center text-[10px] text-slate-500 mt-3 max-w-sm">
            Inputs are generally stable. StandardScaler mathematically centers them cleanly around zero.
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center w-full mt-2 animate-in fade-in zoom-in-95 duration-300">
          <div className="flex w-full justify-between mb-4">
            <div className={`text-xs px-3 py-1 rounded font-bold transition-colors ${!isScaled ? 'bg-red-500/20 text-red-400' : 'text-slate-600'}`}>Raw Target w/ Spike</div>
            <div className={`text-xs px-3 py-1 rounded font-bold transition-colors ${isScaled ? 'bg-emerald-500/20 text-emerald-400' : 'text-slate-600'}`}>RobustScaler + Clipping</div>
          </div>

          <div className="relative w-full h-32 border-b border-l border-slate-700 flex items-end justify-around pb-2">
            {/* Clipping Guides */}
            <div className={`absolute w-full border-t border-dashed border-emerald-500/50 transition-all duration-700 ${isScaled ? 'top-6 opacity-100' : 'top-0 opacity-0'}`} />
            <div className={`absolute right-2 text-[10px] text-emerald-500 transition-all duration-700 ${isScaled ? 'top-2 opacity-100' : 'top-0 opacity-0'}`}>+20 limit</div>

            {/* Data Points */}
            <div className={`w-8 rounded-t transition-all duration-700 ${isScaled ? 'bg-emerald-500' : 'bg-slate-600'}`} style={{ height: isScaled ? '30%' : '5%' }} />
            <div className={`w-8 rounded-t transition-all duration-700 ${isScaled ? 'bg-emerald-500' : 'bg-slate-600'}`} style={{ height: isScaled ? '50%' : '10%' }} />
            
            {/* The Outlier Spike */}
            <div className={`w-8 rounded-t transition-all duration-700 flex items-start justify-center ${isScaled ? 'bg-emerald-400' : 'bg-red-500'}`} 
                 style={{ height: isScaled ? '85%' : '98%' }}>
                 {!isScaled && <span className="text-[10px] text-white mt-1 animate-pulse font-bold">SPIKE!</span>}
            </div>
            
            <div className={`w-8 rounded-t transition-all duration-700 ${isScaled ? 'bg-emerald-500' : 'bg-slate-600'}`} style={{ height: isScaled ? '40%' : '8%' }} />
            <div className={`w-8 rounded-t transition-all duration-700 ${isScaled ? 'bg-emerald-500' : 'bg-slate-600'}`} style={{ height: isScaled ? '60%' : '12%' }} />
          </div>
          <div className="text-center text-[10px] text-slate-500 mt-3 max-w-sm">
            Targets have massive physical spikes. RobustScaler uses medians to completely ignore the spike, preserving the normal data patterns.
          </div>
        </div>
      )}

      <div className="absolute bottom-3 right-3">
        <VisualButton onClick={() => setIsScaled(!isScaled)} active={isScaled}>
          <Scale size={14} /> {isScaled ? "Revert to Raw" : "Apply Scaling"}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedWindow = () => {
  const [pos, setPos] = useState(0);
  const [extracted, setExtracted] = useState([]);

  const slideNext = () => {
    setPos(p => {
      const next = p + 15;
      if (next > 80) {
        setExtracted([]);
        return 0; // Reset
      }
      setExtracted(curr => [...curr, next].slice(-4)); // Keep last 4 visible
      return next;
    });
  };

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 p-4 pb-14">
      <div className="text-xs text-slate-400 mb-4 font-mono">Continuous Data Array (1D)</div>
      
      {/* Base timeline with sliding window */}
      <svg viewBox="0 0 100 20" className="w-[90%] h-12 overflow-visible">
        <path d="M0,10 Q20,0 40,10 T80,10 T100,10" fill="none" stroke="#475569" strokeWidth="1" />
        {/* The sliding window box */}
        <rect x={pos} y="0" width="20" height="20" fill="#3b82f6" fillOpacity="0.3" stroke="#60a5fa" strokeWidth="1" rx="2" className="transition-all duration-300" />
      </svg>

      <div className="flex gap-2 mt-4 items-center">
        <Activity size={16} className={`transition-colors ${extracted.length > 0 ? 'text-blue-500 animate-pulse' : 'text-slate-600'}`} />
        <div className="text-xs text-slate-500">Extracting 2D Matrix...</div>
      </div>

      {/* Stack of extracted windows */}
      <div className="flex flex-col gap-1 mt-4 h-24 justify-end perspective-1000">
        {extracted.map((_, i) => (
          <div key={`${_}-${i}`} className="w-48 h-4 bg-blue-900/50 border border-blue-500/30 rounded flex items-center justify-around px-2 animate-in slide-in-from-top-2 fade-in">
             <div className="w-1 h-1 rounded-full bg-blue-400"></div>
             <div className="w-1 h-1 rounded-full bg-blue-400"></div>
             <div className="w-1 h-1 rounded-full bg-blue-400"></div>
             <div className="w-1 h-1 rounded-full bg-blue-400"></div>
             <div className="text-[8px] text-blue-300 ml-2">t={_} to t={_+20}</div>
          </div>
        ))}
      </div>

      <div className="absolute bottom-3 right-3">
        <VisualButton onClick={slideNext} active={pos > 0}>
           {pos > 80 ? <><RefreshCw size={14} /> Reset Matrix</> : <><Maximize size={14} /> Extract Window</>}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedPCA = () => {
  const [isCompressed, setIsCompressed] = useState(false);

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 gap-6 p-6 pb-14">
      <div className="flex justify-between w-full items-center">
        
        {/* High Dim Input */}
        <div className="flex flex-col items-center group">
          <div className="w-20 h-24 rounded-lg border border-slate-700 bg-slate-800/50 flex flex-wrap gap-[2px] p-2 overflow-hidden relative">
            {Array.from({length: 80}).map((_, i) => (
              <div key={i} className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-pulse" style={{ animationDelay: `${i * 0.05}s` }} />
            ))}
            <div className={`absolute inset-0 bg-gradient-to-t transition-colors duration-500 ${isCompressed ? 'from-slate-900/80 to-slate-900/80' : 'from-slate-900 to-transparent'}`} />
          </div>
          <div className="text-xs text-red-400 mt-2 font-bold bg-red-500/10 px-2 py-1 rounded">1,920 Dimensions</div>
        </div>
        
        {/* Compression Engine */}
        <div className="flex flex-col items-center relative">
          <div className={`text-[10px] absolute -top-6 whitespace-nowrap px-2 py-1 rounded-full transition-colors duration-500 ${isCompressed ? 'text-blue-200 bg-blue-700 shadow-[0_0_10px_#2563eb]' : 'text-slate-500 bg-slate-800'}`}>PCA Math Engine</div>
          <svg width="60" height="40" viewBox="0 0 60 40">
             <path d="M0,0 L60,15 L60,25 L0,40 Z" fill={isCompressed ? "#2563eb" : "#334155"} fillOpacity={isCompressed ? "0.4" : "0.2"} stroke={isCompressed ? "#3b82f6" : "#475569"} strokeWidth="1" className="transition-all duration-500" />
             {isCompressed && <line x1="10" y1="20" x2="50" y2="20" stroke="#60a5fa" strokeWidth="2" strokeDasharray="4,4" className="animate-custom-dash" />}
          </svg>
        </div>

        {/* Low Dim Output */}
        <div className="flex flex-col items-center">
          <div className="w-20 h-24 flex items-end gap-1 border-b border-slate-700 pb-1 relative">
            {[80, 60, 45, 35, 25, 15, 10, 8, 5, 4, 3, 2].map((val, i) => (
              <div 
                key={i} 
                className="w-1.5 bg-emerald-500 rounded-t transition-all duration-700" 
                style={{ 
                  height: isCompressed ? `${val}%` : '0%', 
                  opacity: 1 - (i * 0.06),
                  boxShadow: (i < 3 && isCompressed) ? '0 0 5px #10b981' : 'none',
                  transitionDelay: `${i * 50}ms`
                }} 
              />
            ))}
          </div>
          <div className={`text-xs mt-2 font-bold px-2 py-1 rounded transition-colors duration-500 ${isCompressed ? 'text-emerald-400 bg-emerald-500/10' : 'text-slate-600 bg-slate-800'}`}>12 Components</div>
        </div>

      </div>
      
      <div className="text-[10px] text-center text-slate-400 max-w-[80%]">
        Extracts 90%+ of the variance (the most important patterns) while squashing the size by 160x. Deep learning models train exponentially faster.
      </div>

      <div className="absolute bottom-3 right-3">
        <VisualButton onClick={() => setIsCompressed(!isCompressed)} active={isCompressed}>
          <Minimize2 size={14} /> {isCompressed ? "Reset Matrix" : "Run Compression"}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedExport = () => {
  const [isExporting, setIsExporting] = useState(false);

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 gap-6 p-6 pb-14">
      
      <div className="flex items-center justify-between w-full max-w-sm relative mt-4">
        {/* 3 Separate Dataframes Merging */}
        <div className="flex flex-col gap-2 relative z-10">
          <div className={`w-20 h-8 rounded bg-blue-900/50 border-2 border-blue-500 flex items-center justify-center text-xs font-bold text-blue-400 transition-transform duration-700 ${isExporting ? 'translate-x-12 translate-y-10' : ''}`}>X (Cond)</div>
          <div className={`w-20 h-8 rounded bg-emerald-900/50 border-2 border-emerald-500 flex items-center justify-center text-xs font-bold text-emerald-400 transition-transform duration-700 ${isExporting ? 'translate-x-12 z-20' : ''}`}>U (Ctrl)</div>
          <div className={`w-20 h-8 rounded bg-purple-900/50 border-2 border-purple-500 flex items-center justify-center text-xs font-bold text-purple-400 transition-transform duration-700 ${isExporting ? 'translate-x-12 -translate-y-10' : ''}`}>Theta (PCA)</div>
        </div>

        <ChevronRight size={24} className={`text-slate-600 transition-opacity duration-500 ${isExporting ? 'opacity-0' : 'opacity-100'}`} />

        {/* Disk Storage & Models Output */}
        <div className="flex flex-col gap-3">
          <div className={`flex items-center gap-2 transition-all duration-700 delay-300 ${isExporting ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4'}`}>
            <Database size={18} className="text-amber-400" />
            <div className="text-[10px] font-mono text-amber-200 bg-amber-900/30 px-2 py-0.5 rounded leading-tight">train.parquet<br/>val.parquet<br/>test.parquet</div>
          </div>
          <div className={`flex items-center gap-2 transition-all duration-700 delay-700 ${isExporting ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4'}`}>
            <Package size={18} className="text-indigo-400" />
            <div className="text-[10px] font-mono text-indigo-200 bg-indigo-900/30 px-2 py-0.5 rounded leading-tight">x_scaler.pkl<br/>u_scaler.pkl<br/>pca_model.pkl</div>
          </div>
        </div>
      </div>

      <div className="text-center text-[10px] text-slate-500 mt-4">Reassembling matrices and saving production artifacts.</div>

      <div className="absolute bottom-3 right-3">
        <VisualButton onClick={() => setIsExporting(!isExporting)} active={isExporting}>
          {isExporting ? <RefreshCw size={14} /> : <Save size={14} />} {isExporting ? "Reset Workspace" : "Execute Export"}
        </VisualButton>
      </div>
    </div>
  );
};

const AnimatedAnatomy = () => {
  const [view, setView] = useState('parquet'); // 'parquet' or 'pkl'

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden flex flex-col p-4 pb-14 border border-slate-800">
      
      {/* Header Toggle */}
      <div className="flex justify-center gap-3 mb-6 mt-2">
         <VisualButton onClick={() => setView('parquet')} active={view === 'parquet'}>
            <Database size={14}/> .parquet Contents
         </VisualButton>
         <VisualButton onClick={() => setView('pkl')} active={view === 'pkl'}>
            <Package size={14}/> .pkl Contents
         </VisualButton>
      </div>

      {/* Dynamic Content */}
      <div className="flex-1 flex flex-col justify-center px-2">
        {view === 'parquet' ? (
          <div className="flex flex-col gap-3 h-full justify-center animate-in fade-in duration-300 slide-in-from-bottom-2">
            <div className="text-[11px] text-slate-400 text-center font-mono bg-slate-800/50 py-1 rounded w-max mx-auto px-3 border border-slate-700">
               Shape: <span className="text-white">(N_Samples, 18)</span>
            </div>
            
            <div className="flex w-full rounded-lg border border-slate-700 overflow-hidden text-center text-[10px] font-bold shadow-lg mt-2">
               <div className="w-[25%] bg-blue-900/50 text-blue-400 py-3 border-r border-slate-700 flex flex-col justify-center">
                  <span className="text-sm">X</span>
                  <span>(3 Cols)</span>
                  <span className="text-[8px] font-normal text-blue-300 mt-1 opacity-80">Condition<br/>(Z-Scaled)</span>
               </div>
               <div className="w-[25%] bg-emerald-900/50 text-emerald-400 py-3 border-r border-slate-700 flex flex-col justify-center">
                  <span className="text-sm">U</span>
                  <span>(3 Cols)</span>
                  <span className="text-[8px] font-normal text-emerald-300 mt-1 opacity-80">Control<br/>(Z-Scaled)</span>
               </div>
               <div className="w-[50%] bg-purple-900/50 text-purple-400 py-3 flex flex-col justify-center">
                  <span className="text-sm">Theta</span>
                  <span>(12 Cols)</span>
                  <span className="text-[8px] font-normal text-purple-300 mt-1 opacity-80">PCA Compressed Curves<br/>(Robust Scaled)</span>
               </div>
            </div>
            
            <div className="flex gap-2 mt-4 text-[9px] text-slate-400 justify-center font-mono">
               <span className="bg-slate-800/80 px-2 py-1 rounded border border-slate-700">train.parquet</span>
               <span className="bg-slate-800/80 px-2 py-1 rounded border border-slate-700">val.parquet</span>
               <span className="bg-slate-800/80 px-2 py-1 rounded border border-slate-700">test.parquet</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-4 h-full justify-center animate-in fade-in duration-300 slide-in-from-bottom-2 font-mono text-[11px] text-slate-300">
            
            <div className="bg-[#0f172a] p-3 rounded-xl border border-slate-700 shadow-lg relative group hover:border-slate-500 transition-colors">
               <div className="absolute top-0 right-0 bg-slate-700 text-white px-2 py-0.5 rounded-bl-lg rounded-tr-xl text-[9px] font-bold">x_scaler.pkl</div>
               <span className="text-amber-400">StandardScaler</span> {'{'}
               <div className="pl-4 py-1 leading-relaxed">
                 <span className="text-blue-300">mean_</span>: <span className="text-emerald-300">[120.4, 45.2, 8.9]</span>,<br/>
                 <span className="text-blue-300">scale_</span>: <span className="text-emerald-300">[2.1, 0.4, 1.2]</span>
               </div>
               {'}'}
            </div>
            
            <div className="bg-[#0f172a] p-3 rounded-xl border border-slate-700 shadow-lg relative group hover:border-slate-500 transition-colors">
               <div className="absolute top-0 right-0 bg-slate-700 text-white px-2 py-0.5 rounded-bl-lg rounded-tr-xl text-[9px] font-bold">pca_model.pkl</div>
               <span className="text-amber-400">PCA</span> {'{'}
               <div className="pl-4 py-1 leading-relaxed">
                 <span className="text-blue-300">components_</span>: Matrix<span className="text-slate-400">(12, 1920)</span>,<br/>
                 <span className="text-blue-300">mean_</span>: Array<span className="text-slate-400">(1920,)</span>
               </div>
               {'}'}
            </div>

          </div>
        )}
      </div>

    </div>
  );
};


// ==========================================
// MAIN CONFIGURATION & COMPONENT
// ==========================================

const steps = [
  {
    id: 'raw',
    title: '1. Raw SCADA Ingestion',
    icon: Activity,
    codeSnippet: `df = pd.read_parquet(INPUT_PATH)\ndf.columns = df.columns.str.strip().str.replace(r'\\s+', '_')`,
    description: "SCADA systems generate massive amounts of continuous high-frequency data from turbines and compressors. We start by loading the raw Parquet dataset, standardizing column names, and categorizing our features into Condition (X), Control (U), and Target (Theta).",
    why: "Model impact: Raw data is far too noisy and chaotic for a neural network to learn from. Feeding this directly into a model results in a phenomenon called 'garbage in, garbage out', preventing the network from converging.",
    Visual: AnimatedRawData
  },
  {
    id: 'clean',
    title: '2. Cleaning & Micro-Noise',
    icon: FileCode2,
    codeSnippet: `df.replace([np.inf, -np.inf], np.nan, inplace=True)\ndf = df.ffill().bfill().fillna(0.0)\n\n# Inject micro-noise to flatlined sensors\nif df[col].std() < 1e-6:\n    df[col] += np.random.normal(0, 1e-5, size=len(df))`,
    description: "Industrial sensors frequently drop out (creating NaNs) or get stuck (flatlining). We use forward/backward filling to patch missing gaps. Crucially, we detect variables with ~0 variance and inject microscopic Gaussian noise.",
    why: "Model impact: Deep learning matrices require mathematical variance to invert. A completely flat line will cause singular matrix errors (crashes). Micro-noise acts as a mathematical lubricant, saving the training loop without altering the physics.",
    Visual: AnimatedCleaning
  },
  {
    id: 'split',
    title: '3. Chronological Split',
    icon: Scissors,
    codeSnippet: `M_total = len(df) - WINDOW_SIZE + 1\ntrain_end = int(M_total * 0.70)\nval_end = int(M_total * 0.85)`,
    description: "Before ANY scaling or transformations occur, we must slice the timeline strictly chronologically: 70% Train, 15% Validation, 15% Test. We subtract the WINDOW_SIZE from the end so our targets don't request data that hasn't happened yet.",
    why: "Model impact: If we apply scaling before splitting, the mean/variance of the future test set 'leaks' into the training set. This causes artificially high accuracy in the lab, but total failure when deployed to a live plant.",
    Visual: AnimatedSplit
  },
  {
    id: 'scale',
    title: '4. Targeted Scaling Strategies',
    icon: Scale,
    codeSnippet: `# Standard Scaler for well-behaved X/U\nx_scaler.fit(x_array[:train_end])\nx_scaled = x_scaler.transform(x_array)\n\n# Robust Scaler for volatile Theta\ntheta_scaler.fit(theta_array[:train_end + WINDOW_SIZE])\ntheta_scaled = np.clip(theta_scaler.transform(theta_array), -20, 20)`,
    description: "We split our scaling strategy based on the data profile. Input variables (X and U) like Pressure and Shaft Speed are stable, so we apply StandardScaler (Z-Score) to center them. Target variables (Theta) are highly volatile, so we use RobustScaler and explicit clipping.",
    why: "Model impact: If we used StandardScaler on Theta, massive 1-in-a-million turbine startup spikes would skew the mean calculation, squishing 99% of normal operational data into a flat line of zeroes. RobustScaler uses medians instead, preserving true operational variance.",
    Visual: AnimatedScale
  },
  {
    id: 'window',
    title: '5. Sliding Windows',
    icon: Maximize,
    codeSnippet: `WINDOW_SIZE = 14400 # 4 hours\nDOWNSAMPLE_RATE = 60 # 1-min intervals\n\nwindows = np.lib.stride_tricks.sliding_window_view(theta_scaled, window_shape=WINDOW_SIZE, axis=0)\npca_input = windows[:, ::DOWNSAMPLE_RATE, :].reshape(windows.shape[0], -1)`,
    description: "Instead of predicting a single point in the future, we use NumPy stride tricks to extract massive 14,400-step (4 hour) future curves. We then downsample these curves (taking every 60th point) to strip out micro-fluctuations.",
    why: "Model impact: Predicting an entire curve forces the AI to learn long-term thermodynamic trends rather than just reacting to the immediate next second. Downsampling removes high-frequency noise that acts as a distraction to the neural network.",
    Visual: AnimatedWindow
  },
  {
    id: 'pca',
    title: '6. PCA Compression',
    icon: Minimize2,
    codeSnippet: `pca = PCA(n_components=12)\n\n# Fit on 20k random Train samples (Apple Silicon fix included)\nfit_batch = np.ascontiguousarray(pca_input[fit_indices], dtype=np.float64)\npca.fit(fit_batch)\n\ntheta_pca_features = pca.transform(...)`,
    description: "Our extracted windows are still too large (8 target variables × 240 steps = 1,920 dimensions). We fit a Principal Component Analysis (PCA) model to learn the 12 most dominant 'shapes' that can reconstruct any future trajectory.",
    why: "Model impact: Neural networks struggle with the 'curse of dimensionality'. Reducing the target from 1,920 down to 12 scalar coefficients compresses the problem by 160x, allowing models to converge infinitely faster and use standard MSE loss functions.",
    Visual: AnimatedPCA
  },
  {
    id: 'export',
    title: '7. Matrix Reassembly & Export',
    icon: Save,
    codeSnippet: `final_df = pd.concat([x_df_scaled, u_df_scaled, theta_pca_df], axis=1)\n\nfinal_df.iloc[:train_end].to_parquet("train.parquet")\n# ...\njoblib.dump(pca, "trajectory_pca_model.pkl")`,
    description: "After all mathematical transformations are complete, we concatenate the scaled Condition (X), Control (U), and compressed Target (Theta PCA) arrays back together into a single master dataframe. We then physically dump the chunks to disk.",
    why: "Model impact: Exporting to Parquet (instead of CSV) makes data loading into PyTorch 50x faster. We also split the files physically so the neural network dataloader doesn't accidentally read future test data into RAM.",
    Visual: AnimatedExport
  },
  {
    id: 'anatomy',
    title: '8. Artifact Anatomy (Deep Dive)',
    icon: FileSearch,
    codeSnippet: `print(final_df.columns)\n# [X_1..X_3, U_1..U_3, PCA_1..PCA_12]\n\n# --- LIVE INFERENCE PSEUDOCODE ---\nprod_pca = joblib.load("pca_model.pkl")\nprod_x_scale = joblib.load("x_scaler.pkl")`,
    description: "Let's unpack the final output files. The Parquet files contain our fully concatenated 18-column matrix. The Pickle files (.pkl) contain the internal mathematical states (means, variances, component arrays) of our preprocessing objects.",
    why: "Model impact: When deploying the ML model to a live turbine, we must transform the live sensor streaming data using the EXACT same mathematical scales learned during training. The .pkl files act as the mandatory 'translation dictionary' for live AI inference.",
    Visual: AnimatedAnatomy
  }
];

export default function PreprocessingSection() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  const handleNext = () => setCurrentStep(prev => Math.min(prev + 1, steps.length - 1));
  const handlePrev = () => setCurrentStep(prev => Math.max(prev - 1, 0));

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">
      
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
        <div className="p-3 bg-blue-100 text-blue-600 rounded-xl">
          <Layers className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Data Preprocessing Pipeline</h2>
          <p className="text-sm text-slate-500 font-medium">Interactive Walkthrough of <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-700">preprocessing.py</code></p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full flex gap-1.5 mb-8">
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
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-8 min-h-[420px]">
        
        {/* Left Col: Explanations & Code */}
        <div className="flex flex-col gap-5 min-w-0">
          <div className="flex items-center gap-3 mb-1">
            <div className="p-2.5 bg-slate-900 rounded-xl text-white shadow-md">
              <step.icon size={22} />
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">{step.title}</h3>
          </div>
          
          <div className="text-slate-600 leading-relaxed text-[15px]">
            {step.description}
          </div>

          <div className="bg-amber-50/80 border border-amber-200 p-4 rounded-xl shadow-sm relative overflow-hidden">
            <div className="absolute top-0 left-0 w-1 h-full bg-amber-400"></div>
            <h4 className="text-sm font-bold text-amber-900 mb-1.5 flex items-center gap-2">
              <Activity size={16} className="text-amber-600"/>
              The ML Logic
            </h4>
            <p className="text-sm text-amber-800/90 leading-relaxed">{step.why}</p>
          </div>

          <div className="mt-auto pt-2">
            <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 pl-1">Python Implementation</h4>
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl blur opacity-20 group-hover:opacity-30 transition-opacity"></div>
              <pre className="relative bg-[#0f172a] text-[#e2e8f0] p-4 rounded-xl text-xs sm:text-sm overflow-x-auto font-mono shadow-inner border border-slate-700/50">
                <code>{step.codeSnippet}</code>
              </pre>
            </div>
          </div>
        </div>

        {/* Right Col: Visual Representation */}
        <div className="flex flex-col min-w-0">
          <div className="flex justify-between items-center mb-3 px-1">
            <span className="text-sm font-bold text-slate-700">Live Process Visualization</span>
            <span className="font-mono bg-blue-100 text-blue-700 px-2.5 py-1 rounded-md text-xs font-bold">
              Step {currentStep + 1} / {steps.length}
            </span>
          </div>
          
          <div className="flex-1 w-full rounded-2xl shadow-lg shadow-slate-200/50 overflow-hidden relative border-4 border-slate-900/5 bg-slate-900">
             <step.Visual />
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mt-8 pt-6 border-t border-slate-200">
        <button 
          onClick={handlePrev}
          disabled={currentStep === 0}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
        >
          <ChevronLeft size={18} /> Previous
        </button>
        
        <button 
          onClick={handleNext}
          disabled={currentStep === steps.length - 1}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-white bg-blue-600 hover:bg-blue-700 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-40 disabled:hover:translate-y-0 disabled:cursor-not-allowed disabled:shadow-none"
        >
          {currentStep === steps.length - 1 ? 'Finish Pipeline' : 'Next Step'} <ChevronRight size={18} />
        </button>
      </div>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes custom-dash {
          to {
            stroke-dashoffset: -8;
          }
        }
        .animate-custom-dash {
          animation: custom-dash 1s linear infinite;
        }
      `}} />
    </div>
  );
}