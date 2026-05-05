import React, { useState } from 'react';
import { Layers, Database, Box } from 'lucide-react';

import IntroSection from './sections/IntroSection';
import PreprocessingSection from './sections/PreprocessingSection';
import DatasetSection from './sections/DatasetSection';

export default function App() {
  const [activeTab, setActiveTab] = useState('intro');

  const tabs = [
    { id: 'intro', label: 'Introduction', icon: Box },
    { id: 'preprocessing', label: 'Data Preprocessing', icon: Layers },
    { id: 'dataset', label: 'Dataset Interface', icon: Database },
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 p-4 md:p-6 font-sans">
      <div className="max-w-[1200px] mx-auto">
        <header className="mb-8 text-center md:text-left md:ml-64">
          <h1 className="text-4xl font-extrabold text-slate-900 mb-2 tracking-tight">Genesis Project Architecture</h1>
          <p className="text-lg text-slate-600">Interactive Pipeline Modules</p>
        </header>

        <div className="flex flex-col md:flex-row gap-6">
          {/* Sidebar Navigation */}
          <aside className="w-full md:w-64 flex-shrink-0 overflow-y-auto max-h-[calc(100vh-150px)] pr-2 pb-8 custom-scrollbar">
            <div className="flex flex-col gap-1.5">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 text-left ${
                      isActive 
                        ? 'bg-blue-600 text-white shadow-md' 
                        : 'text-slate-600 hover:bg-white hover:shadow-sm border border-transparent hover:border-slate-200'
                    }`}
                  >
                    <Icon size={18} className={isActive ? 'text-blue-100' : 'text-slate-400'} />
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </aside>

          {/* Main Content Area */}
          <main className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 p-6 md:p-8 min-h-[500px]">
            {activeTab === 'intro' && <IntroSection />}
            {activeTab === 'preprocessing' && <PreprocessingSection />}
            {activeTab === 'dataset' && <DatasetSection />}
          </main>
        </div>
      </div>
    </div>
  );
}