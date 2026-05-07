import React, { useState, useEffect } from 'react';
import { 
  Calculator, 
  ChevronRight, 
  ChevronLeft, 
  Target,
  Maximize2,
  TrendingUp,
  BarChart2,
  Network,
  Cpu,
  Layers
} from 'lucide-react';

// ==========================================
// INTERACTIVE VISUAL COMPONENTS
// ==========================================

const VisualButton = ({ onClick, disabled, children, active }) => (
  <button 
    onClick={onClick} 
    disabled={disabled}
    className={`flex items-center justify-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-bold transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 z-10 ${
      active 
        ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/50' 
        : 'bg-slate-700 hover:bg-slate-600 text-slate-200 shadow-slate-900/50 border border-slate-600'
    }`}
  >
    {children}
  </button>
);

// 1. Mean vs Median
const AnimatedCenters = () => {
  const [hasOutlier, setHasOutlier] = useState(false);

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl flex flex-col items-center justify-center border border-slate-800 p-6 pb-16">
      <div className="text-xs text-slate-400 mb-8 font-mono text-center">
        A SCADA sensor normally reads ~10-20 psi.<br/>Suddenly, a startup transient spikes to 120 psi.
      </div>

      <div className="relative w-full max-w-md h-24 border-b-2 border-slate-700 mb-8">
        {/* Normal Data Points */}
        {[10, 15, 20, 25, 30].map((val, i) => (
          <div key={i} className="absolute bottom-0 w-4 h-4 bg-blue-400 rounded-full -mb-2 shadow-[0_0_10px_#60a5fa]" style={{ left: `${(val/130)*100}%`, transform: 'translateX(-50%)' }} />
        ))}
        
        {/* Outlier Data Point */}
        <div className={`absolute bottom-0 w-4 h-4 bg-red-500 rounded-full -mb-2 shadow-[0_0_15px_#ef4444] transition-all duration-700 ${hasOutlier ? 'opacity-100 scale-100' : 'opacity-0 scale-0'}`} style={{ left: `${(120/130)*100}%`, transform: 'translateX(-50%)' }} />

        {/* Median Line (Stays relatively stable) */}
        <div className="absolute top-2 w-0.5 h-24 bg-emerald-400 transition-all duration-700 z-0" style={{ left: `${(hasOutlier ? 22.5 : 20)/130*100}%` }}>
          <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-emerald-900/80 text-emerald-300 text-[10px] px-2 py-0.5 rounded border border-emerald-500 font-bold">
            Median: {hasOutlier ? '22.5' : '20.0'}
          </div>
        </div>

        {/* Mean Line (Gets dragged heavily) */}
        <div className="absolute top-10 w-0.5 h-16 bg-amber-400 transition-all duration-700 z-0" style={{ left: `${(hasOutlier ? 36.6 : 20)/130*100}%` }}>
          <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-amber-900/80 text-amber-300 text-[10px] px-2 py-0.5 rounded border border-amber-500 font-bold whitespace-nowrap">
            Mean (μ): {hasOutlier ? '36.6' : '20.0'}
          </div>
        </div>
      </div>

      <div className="text-[11px] text-slate-400 text-center max-w-sm">
        The <strong>Mean</strong> calculates the mathematical average, so it gets dragged violently by the spike. The <strong>Median</strong> just picks the "middle person in line", completely ignoring the magnitude of the outlier.
      </div>

      <div className="absolute bottom-4">
        <VisualButton onClick={() => setHasOutlier(!hasOutlier)} active={hasOutlier}>
          {hasOutlier ? "Remove Outlier" : "Trigger Outlier Spike"}
        </VisualButton>
      </div>
    </div>
  );
};

// 2. Std Dev vs IQR
const AnimatedSpread = () => {
  const [hasOutlier, setHasOutlier] = useState(false);

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl flex flex-col items-center justify-center border border-slate-800 p-6 pb-16">
      <div className="text-xs text-slate-400 mb-8 font-mono text-center">
        Measuring the "spread" or "variance" of the data.
      </div>

      <div className="relative w-full max-w-md h-32 border-b-2 border-slate-700 mb-6 flex items-end">
        {/* Core Distribution (Bell curve shape) */}
        <svg viewBox="0 0 100 50" className="w-full h-full absolute inset-0 preserve-3d overflow-visible">
          <path d="M10,50 Q20,50 30,10 T50,50" fill="rgba(96, 165, 250, 0.2)" stroke="#60a5fa" strokeWidth="2" />
          
          {/* Outlier blip */}
          <path d="M85,50 Q90,30 95,50" fill="rgba(239, 68, 68, 0.3)" stroke="#ef4444" strokeWidth="2" className={`transition-all duration-700 ${hasOutlier ? 'opacity-100' : 'opacity-0'}`} />
        </svg>

        {/* IQR Range (Fixed on core 50%) */}
        <div className="absolute bottom-4 left-[20%] w-[20%] h-6 border-l-2 border-r-2 border-emerald-400 bg-emerald-400/10 flex items-center justify-center transition-all duration-700">
          <span className="text-[10px] text-emerald-400 font-bold bg-slate-900 px-1">IQR</span>
        </div>

        {/* Std Dev Range (Expands) */}
        <div className={`absolute bottom-14 h-6 border-l-2 border-r-2 border-amber-400 bg-amber-400/10 flex items-center justify-center transition-all duration-700 ${hasOutlier ? 'left-[15%] w-[45%]' : 'left-[15%] w-[30%]'}`}>
          <span className="text-[10px] text-amber-400 font-bold bg-slate-900 px-1">± 1 Std Dev (σ)</span>
        </div>
      </div>

      <div className="text-[11px] text-slate-400 text-center max-w-sm">
        <strong>Standard Deviation (σ)</strong> looks at distance from the mean, so an outlier artificially inflates the "normal" operating band. The <strong>Interquartile Range (IQR)</strong> rigidly measures the distance between the 25th and 75th percentiles, safely boxing in the core physics.
      </div>

      <div className="absolute bottom-4">
        <VisualButton onClick={() => setHasOutlier(!hasOutlier)} active={hasOutlier}>
          {hasOutlier ? "Remove Outlier" : "Trigger Outlier Spike"}
        </VisualButton>
      </div>
    </div>
  );
};

// 3. RobustScaler vs StandardScaler
const AnimatedScaling = () => {
  const [mode, setMode] = useState('raw'); // 'raw', 'standard', 'robust'

  const getPath = () => {
    if (mode === 'raw') return "M0,80 L20,70 L40,85 L50,10 L60,75 L80,80 L100,75"; // Spike up to 10
    if (mode === 'standard') return "M0,55 L20,52 L40,58 L50,10 L60,54 L80,55 L100,54"; // Squashed near mean
    if (mode === 'robust') return "M0,60 L20,40 L40,70 L50,-50 L60,50 L80,60 L100,50"; // Core expanded, spike goes off chart
  };

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl flex flex-col items-center justify-center border border-slate-800 p-6 pb-16">
      
      <div className="flex gap-2 mb-6 absolute top-4">
         <button onClick={() => setMode('raw')} className={`text-[10px] px-3 py-1 rounded font-bold border transition-colors ${mode === 'raw' ? 'bg-slate-700 border-slate-500 text-white' : 'border-slate-700 text-slate-500'}`}>Raw Data</button>
         <button onClick={() => setMode('standard')} className={`text-[10px] px-3 py-1 rounded font-bold border transition-colors ${mode === 'standard' ? 'bg-amber-600 border-amber-500 text-white' : 'border-slate-700 text-slate-500'}`}>StandardScaler</button>
         <button onClick={() => setMode('robust')} className={`text-[10px] px-3 py-1 rounded font-bold border transition-colors ${mode === 'robust' ? 'bg-emerald-600 border-emerald-500 text-white' : 'border-slate-700 text-slate-500'}`}>RobustScaler + Clip</button>
      </div>

      <div className="relative w-full h-40 border-l border-b border-slate-600 overflow-hidden mt-8">
        {/* Zero line */}
        {(mode === 'standard' || mode === 'robust') && (
          <div className="absolute top-1/2 w-full border-t border-dashed border-slate-500 z-0"></div>
        )}
        
        {/* Upper Clip limit */}
        {mode === 'robust' && (
          <div className="absolute top-2 w-full border-t border-red-500/50 bg-red-500/10 h-10 z-0">
             <span className="text-[8px] text-red-400 absolute right-1">np.clip(20)</span>
          </div>
        )}

        <svg viewBox="0 0 100 100" className="w-full h-full absolute inset-0 preserveAspectRatio-none z-10" style={{ overflow: 'visible' }}>
          <path d={getPath()} fill="none" stroke={mode === 'raw' ? '#94a3b8' : mode === 'standard' ? '#fbbf24' : '#34d399'} strokeWidth="3" className="transition-all duration-700" strokeLinejoin="round"/>
        </svg>
      </div>

      <div className="text-[11px] text-slate-400 text-center mt-6 h-10">
        {mode === 'raw' && "Raw SCADA data with a massive 1-in-a-million compressor surge spike."}
        {mode === 'standard' && "StandardScaler centers on the inflated Mean. Because the spike makes the Variance huge, it divides the normal data by a massive number, squashing the core physics into a flat line of zeroes."}
        {mode === 'robust' && "RobustScaler centers on the Median and scales by IQR. The core physics retain their true shape. We explicitly clip the spike to prevent it from overflowing the Neural Network."}
      </div>
    </div>
  );
};

// 4. PCA Intuition (Finding the Axis)
const AnimatedPCAAxis = () => {
  const [isFitted, setIsFitted] = useState(false);

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl flex flex-col items-center justify-center border border-slate-800 p-6 pb-16">
      
      <div className="text-xs text-slate-400 mb-4 font-mono text-center">
        Suppose Suction Pressure (X) and Temperature (Y) are highly correlated.
      </div>

      <div className="relative w-48 h-48 border-l border-b border-slate-600 bg-slate-800/50 rounded-tr-lg">
        {/* Data points (diagonal cluster) */}
        {Array.from({length: 30}).map((_, i) => {
          const x = 20 + i*2 + (Math.random()*15 - 7.5);
          const y = 20 + i*2 + (Math.random()*15 - 7.5);
          return (
            <div key={i} className={`absolute w-1.5 h-1.5 rounded-full transition-all duration-1000 ${isFitted ? 'bg-indigo-500' : 'bg-slate-400'}`} style={{ left: `${x}%`, bottom: `${y}%` }} />
          )
        })}

        {/* Principal Component 1 */}
        <div className={`absolute bottom-0 left-0 w-[140%] h-0.5 bg-indigo-400 origin-bottom-left transition-all duration-1000 shadow-[0_0_10px_#818cf8] ${isFitted ? 'rotate-[-45deg] opacity-100' : 'rotate-0 opacity-0'}`}>
           <span className="absolute right-4 -top-4 text-[9px] font-bold text-indigo-300">PC1 (Max Variance)</span>
        </div>
        
        {/* Principal Component 2 (Orthogonal) */}
        <div className={`absolute bottom-1/2 left-1/2 w-[60%] h-0.5 bg-slate-500 origin-center transition-all duration-1000 ${isFitted ? 'rotate-[45deg] opacity-50' : 'rotate-0 opacity-0'}`}>
           <span className="absolute left-2 -top-4 text-[9px] font-bold text-slate-400">PC2 (Noise)</span>
        </div>
      </div>

      <div className="text-[11px] text-slate-400 text-center mt-6">
        Instead of using 2 separate variables, PCA finds the "vector of maximum variance" (PC1). We can project all points onto this single diagonal line, dropping PC2 entirely. <strong>We just compressed 2 dimensions into 1 without losing the core physics.</strong>
      </div>

      <div className="absolute bottom-4">
        <VisualButton onClick={() => setIsFitted(!isFitted)} active={isFitted}>
          {isFitted ? "Reset Space" : "Calculate Eigenvectors (Fit PCA)"}
        </VisualButton>
      </div>
    </div>
  );
};

// 5. PCA Scree Plot
const AnimatedScreePlot = () => {
  const variances = [45, 20, 10, 7, 5, 3, 2, 2, 1, 1, 1, 1, 1, 0.5, 0.5];
  
  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl flex flex-col items-center justify-center border border-slate-800 p-6">
      
      <div className="text-xs text-slate-400 mb-6 font-mono text-center">
        Extending PCA to our 1,920-dimension SCADA trajectory.<br/>
        <span className="text-blue-400">pca.explained_variance_ratio_.cumsum()</span>
      </div>

      <div className="relative w-full max-w-md h-40 border-l border-b border-slate-700 pt-4 flex items-end justify-between px-2">
         {/* 90% Threshold */}
         <div className="absolute bottom-[90%] w-full border-t border-dashed border-red-500/50 z-0">
           <span className="absolute right-0 -top-4 text-[9px] text-red-400 font-bold">90% Information Cutoff</span>
         </div>

         {variances.map((val, i) => {
            const cumSum = variances.slice(0, i+1).reduce((a,b)=>a+b,0);
            const isKept = cumSum <= 93; // Rough visual threshold for top 12
            
            return (
              <div key={i} className="relative w-full h-full flex flex-col items-center justify-end group">
                
                {/* Cumulative Plot Line */}
                <div className="absolute w-2 h-2 rounded-full bg-orange-400 z-20 shadow-[0_0_5px_#f97316]" style={{ bottom: `${cumSum}%`, left: '50%', transform: 'translateX(-50%) translateY(50%)' }}></div>
                {i < variances.length - 1 && (
                  <svg className="absolute w-[200%] h-full z-10 pointer-events-none" style={{ left: '50%', bottom: 0 }}>
                    <line x1="0" y1={`${100-cumSum}%`} x2="100%" y2={`${100-(cumSum + variances[i+1])}%`} stroke="#f97316" strokeWidth="1.5" strokeDasharray="2,2"/>
                  </svg>
                )}

                {/* Individual Bar */}
                <div className={`w-[70%] rounded-t border-t border-l border-r transition-all duration-500 ${isKept ? 'bg-blue-600 border-blue-400' : 'bg-slate-700 border-slate-600'}`} style={{ height: `${val}%` }}></div>
                <div className="text-[8px] text-slate-500 mt-1">C{i+1}</div>
              </div>
            )
         })}

         {/* Dropped dimensions indicator */}
         <div className="absolute -right-8 bottom-2 flex items-center gap-1 text-[10px] text-slate-500">
            ... 1,908 dropped
         </div>
      </div>

      <div className="text-[11px] text-slate-400 text-center mt-6">
        Component 1 (C1) captures massive phenomena (e.g., daily ambient temperature sweeps). By the 12th component, we've captured &gt;90% of all real physics. The remaining 1,908 dimensions are microscopic sensor jitter and noise.
      </div>
    </div>
  );
};

// 6. PCA Matrix Reconstruction (Einsum)
const AnimatedEinsum = () => {
  return (
    <div className="relative w-full h-full bg-slate-900 rounded-xl flex flex-col items-center justify-center border border-slate-800 p-6 pb-6">
      
      <div className="text-xs text-slate-400 mb-6 font-mono text-center">
        How do we turn 12 numbers back into a 4-hour SCADA curve?<br/>
        <span className="text-emerald-400 font-bold">np.einsum('k,kj-&gt;j', coeffs, components) + mean</span>
      </div>

      <div className="flex flex-col md:flex-row items-center gap-4 bg-[#0a0f18] p-4 rounded-xl border border-slate-700 shadow-inner w-full max-w-lg">
         
         {/* Coeffs (1x12) */}
         <div className="flex flex-col gap-1 p-2 bg-indigo-900/30 border border-indigo-500/50 rounded items-center w-24">
           <div className="text-[8px] text-indigo-300 font-bold">Coeffs Vector<br/>[1 × 12]</div>
           <div className="grid grid-rows-4 grid-cols-3 gap-0.5 mt-1">
             {Array.from({length: 12}).map((_,i) => <div key={i} className="w-4 h-2 bg-indigo-400 rounded-sm"></div>)}
           </div>
         </div>

         <div className="text-slate-500 font-bold">×</div>

         {/* Components (12x1920) */}
         <div className="flex flex-col gap-1 p-2 bg-emerald-900/30 border border-emerald-500/50 rounded items-center flex-1">
           <div className="text-[8px] text-emerald-300 font-bold">pca.components_<br/>[12 × 1,920]</div>
           <div className="flex flex-col gap-0.5 mt-1 w-full">
             {Array.from({length: 4}).map((_,i) => (
               <div key={i} className="flex justify-between w-full opacity-70">
                 {Array.from({length: 15}).map((_,j) => <div key={j} className="w-2 h-1 bg-emerald-400 rounded-sm"></div>)}
               </div>
             ))}
           </div>
           <div className="text-[8px] text-emerald-500">...</div>
         </div>

         <div className="text-slate-500 font-bold">+</div>

         {/* Mean (1x1920) */}
         <div className="flex flex-col gap-1 p-2 bg-amber-900/30 border border-amber-500/50 rounded items-center w-16">
           <div className="text-[8px] text-amber-300 font-bold">pca.mean_<br/>[1 × 1920]</div>
           <div className="flex justify-between w-full mt-1">
             {Array.from({length: 5}).map((_,j) => <div key={j} className="w-2 h-1 bg-amber-400 rounded-sm"></div>)}
           </div>
         </div>

         <div className="text-slate-500 font-bold">=</div>

         {/* Output Trajectory */}
         <div className="relative w-32 h-16 border-b border-l border-slate-600 bg-slate-800 rounded flex-shrink-0 overflow-hidden">
           <svg className="w-full h-full absolute inset-0" viewBox="0 0 100 40" preserveAspectRatio="none">
              <path d="M0,30 Q25,0 50,20 T100,10" fill="none" stroke="#38bdf8" strokeWidth="2.5" className="drop-shadow-[0_0_4px_#38bdf8]"/>
           </svg>
           <div className="absolute -bottom-0.5 right-1 text-[8px] text-sky-400 font-bold">Output Trajectory</div>
         </div>

      </div>

      <div className="text-[11px] text-slate-400 text-center mt-6 max-w-md">
        By multiplying our 12 coefficients against the 12 "Eigenvector shapes" (components) and adding back the dataset average (mean), we perfectly expand the compressed data back into a full physical wave.
      </div>
    </div>
  );
};


// ==========================================
// MAIN CONFIGURATION & COMPONENT
// ==========================================

const steps = [
  {
    id: 'center',
    chapter: 'Scaling & Normalization',
    title: '1. Defining the "Center"',
    icon: Target,
    Visual: AnimatedCenters,
    description: "Before machine learning models can learn, data must be centered around zero. Standard libraries typically use the Mean (μ). However, physical systems like turbocompressors experience massive, short-lived spikes (e.g., startup transients, surge events)."
  },
  {
    id: 'spread',
    chapter: 'Scaling & Normalization',
    title: '2. Defining the "Spread"',
    icon: Maximize2,
    Visual: AnimatedSpread,
    description: "Once centered, data is divided by its spread so all variables operate on a scale of ~1.0. Standard Deviation (σ) squares the distance of every point from the mean, meaning a single outlier will drastically inflate σ. The Interquartile Range (IQR) completely ignores the outer 50% of the data."
  },
  {
    id: 'scaling',
    chapter: 'Scaling & Normalization',
    title: '3. StandardScaler vs. RobustScaler',
    icon: TrendingUp,
    Visual: AnimatedScaling,
    description: "StandardScaler applies Z-Score normalization: Z = (X - μ) / σ. RobustScaler applies: R = (X - Median) / IQR. Notice how RobustScaler protects the true thermodynamic curves of the target variable (Theta), while explicit clipping safely truncates the anomaly to prevent network overflow."
  },
  {
    id: 'pca-axis',
    chapter: 'Dimensionality Reduction',
    title: '4. The Axis of Maximum Variance',
    icon: Network,
    Visual: AnimatedPCAAxis,
    description: "Our 4-hour sliding windows contain 1,920 dimensions (8 sensors x 240 timesteps). Neural networks suffer from the 'Curse of Dimensionality'—they get lost in huge, sparse matrices. Principal Component Analysis (PCA) solves this by finding diagonal axes (Eigenvectors) that capture the maximum variance, allowing us to drop less important orthogonal axes."
  },
  {
    id: 'pca-scree',
    chapter: 'Dimensionality Reduction',
    title: '5. The Scree Plot (Cumulative Variance)',
    icon: BarChart2,
    Visual: AnimatedScreePlot,
    description: "Every time PCA creates an axis, we calculate how much of the original dataset's physics (Variance) it captured. By summing these percentages cumulatively, we see that the top 12 components capture >90% of the movement. We can throw away the remaining 1,908 dimensions as pure sensor noise."
  },
  {
    id: 'pca-recon',
    chapter: 'Dimensionality Reduction',
    title: '6. Matrix Reconstruction (Einsum)',
    icon: Cpu,
    Visual: AnimatedEinsum,
    description: "During live inference, our ML model predicts those 12 scalar coefficients. To turn them back into a physical 4-hour curve, we mathematically expand them against the PCA components matrix. We use np.einsum instead of standard BLAS multiplication to safely bypass known memory-corruption bugs on Apple Silicon hardware."
  }
];

export default function MathIntuitionSection() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  const handleNext = () => setCurrentStep(prev => Math.min(prev + 1, steps.length - 1));
  const handlePrev = () => setCurrentStep(prev => Math.max(prev - 1, 0));

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">
      
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
        <div className="p-3 bg-indigo-100 text-indigo-600 rounded-xl">
          <Calculator className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Mathematical Intuition</h2>
          <p className="text-sm text-slate-500 font-medium">Deconstructing the Data Processing Math Engine</p>
        </div>
      </div>

      {/* Chapter Indicator */}
      <div className="w-full flex items-center justify-between mb-2">
         <span className="text-xs font-bold uppercase tracking-widest text-indigo-500">{step.chapter}</span>
         <span className="font-mono bg-slate-100 text-slate-600 px-2.5 py-1 rounded-md text-xs font-bold">
            Concept {currentStep + 1} / {steps.length}
         </span>
      </div>

      {/* Progress Bar */}
      <div className="w-full flex gap-1 mb-6">
        {steps.map((s, idx) => (
          <div 
            key={s.id} 
            onClick={() => setCurrentStep(idx)}
            className={`h-2 flex-1 rounded-full cursor-pointer transition-all duration-300 ${
              idx === currentStep ? 'bg-indigo-600 scale-y-125' : 
              idx < currentStep ? 'bg-indigo-300' : 'bg-slate-200 hover:bg-slate-300'
            }`}
            title={s.title}
          />
        ))}
      </div>

      {/* Main Content Layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 min-h-[450px]">
        
        {/* Left Col: Explanations (Takes up 4 cols on large screens) */}
        <div className="lg:col-span-4 flex flex-col gap-5 min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2.5 bg-slate-900 rounded-xl text-white shadow-md flex-shrink-0">
              <step.icon size={20} />
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 tracking-tight leading-tight">{step.title}</h3>
          </div>
          
          <div className="text-slate-600 leading-relaxed text-[15px]">
            {step.description}
          </div>

          <div className="mt-auto pt-4 border-t border-slate-100">
             <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex items-start gap-3">
               <Layers className="text-indigo-400 flex-shrink-0 mt-0.5" size={18}/>
               <p className="text-xs text-slate-600">
                 These mathematical transformations are essential steps required to satisfy the "Data-Driven Joint Distribution" outlined in Phase II of the Genesis Mission Proposal.
               </p>
             </div>
          </div>
        </div>

        {/* Right Col: Interactive Visual (Takes up 8 cols on large screens) */}
        <div className="lg:col-span-8 flex flex-col min-w-0">
          <div className="flex-1 w-full rounded-2xl shadow-xl overflow-hidden relative border-4 border-slate-900/5 bg-[#0d1117]">
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
          <ChevronLeft size={18} /> Previous Concept
        </button>
        
        <button 
          onClick={handleNext}
          disabled={currentStep === steps.length - 1}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all text-white bg-indigo-600 hover:bg-indigo-700 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-40 disabled:hover:translate-y-0 disabled:cursor-not-allowed disabled:shadow-none"
        >
          {currentStep === steps.length - 1 ? 'Finish Exploration' : 'Next Concept'} <ChevronRight size={18} />
        </button>
      </div>

    </div>
  );
}