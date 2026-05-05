import React from 'react';

export default function IntroSection() {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-slate-800">Genesis Documentation Explorer</h2>
      <p className="text-slate-600 leading-relaxed text-lg">
        Welcome to the interactive code documentation for the physics-informed AI modeling pipeline.
        This slider helps you explore exactly how our code maps onto the theoretical framework.
      </p>
      
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-md">
        <h3 className="font-semibold text-blue-800 mb-2">Project Goals</h3>
        <ul className="list-disc pl-5 text-blue-900 space-y-1">
          <li>Scenario Encoding & Preprocessing via Functional PCA</li>
          <li>Normalizing Flows for joint target distributions</li>
          <li>In-memory robust PyTorch Tensor scaling and sliding windows</li>
        </ul>
      </div>
      
      <p className="text-slate-600">👈 Select a module from the sidebar to inspect the interactive slides.</p>
    </div>
  );
}