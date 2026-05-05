import React from 'react';

export default function DatasetSection() {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-slate-800">2. Runtime Dataloader (dataset.py)</h2>
      <p className="text-slate-600">
        Our PyTorch Dataset is highly optimized for Apple Silicon (M4 Max) RAM. It loads the exact Parquet block containing `x`, `u`, and the 
        compressed `theta` scenario coefficients immediately into memory.
      </p>

      <div className="bg-white border rounded-xl shadow-sm mt-6">
          <div className="bg-slate-100 border-b px-4 py-3 font-mono text-sm font-semibold flex items-center justify-between text-slate-700">
            <span>SCADAPipelineDataset.__getitem__</span>
            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full border border-blue-200">Float32 Tensors</span>
          </div>
          <div className="p-5">
            <pre className="text-sm bg-slate-800 text-slate-50 p-4 rounded-lg overflow-x-auto leading-relaxed">
<span className="text-pink-400">def</span> <span className="text-blue-300">__getitem__</span>(self, idx):
    <span className="text-pink-400">return</span> {'{'}
        <span className="text-green-300">"theta"</span>: self.theta_tensor[idx],
        <span className="text-green-300">"condition"</span>: self.condition_tensor[idx] 
    {'}'}
            </pre>
            <p className="mt-4 text-sm text-slate-600 italic">
              * `condition` is a concatenated tensor containing both the instantaneous controls (`u`) and measured-now variables (`x`).
            </p>
          </div>
      </div>
    </div>
  );
}