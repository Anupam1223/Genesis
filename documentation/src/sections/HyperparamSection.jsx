import React, { useState } from 'react';
import { Settings, Cpu, Database, Activity, BarChart2, Sliders, ChevronDown, ChevronUp, Info } from 'lucide-react';

// ─────────────────────────────────────────────
// DATA — all parameter metadata lives here
// ─────────────────────────────────────────────
const SECTIONS = [
  {
    id: 'training',
    label: 'Training',
    icon: Activity,
    color: 'rose',
    bg: 'bg-rose-50',
    border: 'border-rose-200',
    badge: 'bg-rose-100 text-rose-700',
    iconColor: 'text-rose-500',
    params: [
      {
        key: 'epochs',
        value: '60',
        type: 'int',
        short: 'Full passes over the training set.',
        long: 'One epoch = the model has seen every sample in the training split exactly once. Set to 60 because early_stopping_patience (15) and ReduceLROnPlateau will always terminate training well before epoch 60 — this is just a hard ceiling. The scheduler halves the LR every 3 plateau epochs, so training naturally slows as it converges.',
        tradeoff: { up: 'Higher ceiling = more room for LR decay phases to play out', down: 'Risk of overfitting if early stopping is misconfigured' },
        range: 'Typical: 40–100 when early stopping is active',
      },
      {
        key: 'batch_size',
        value: '4096',
        type: 'int',
        short: 'Samples processed in one forward + backward pass.',
        long: 'Larger batches produce smoother gradient estimates (less noise per step) but consume more GPU memory per batch. 4096 is tuned for the M4 Max 128 GB unified memory. If you see MPS out-of-memory errors, halve to 2048. Too small (< 256) introduces so much gradient noise that spline knots oscillate and loss becomes jagged.',
        tradeoff: { up: 'Smoother gradients, faster per-epoch wall time', down: 'Higher peak memory, may escape sharp minima' },
        range: '512 – 8192 on M4 Max',
      },
      {
        key: 'learning_rate',
        value: '5e-5',
        type: 'float',
        short: 'Initial step size for the AdamW optimiser.',
        long: 'Reduced from 2e-4 to 5e-5 after observing that 2e-4 caused the splines to overfit aggressively — training loss reached −15 (below the theoretical minimum of ~17 nats for a 12D normal) while val loss exploded. At 5e-5 the updates are conservative enough that the spline knots settle into a stable configuration rather than chasing the training distribution. The ReduceLROnPlateau scheduler will further halve this automatically at each plateau.',
        tradeoff: { up: 'Faster convergence early in training', down: 'Too high → splines overfit and drive NLL below theoretical minimum' },
        range: 'Safe range: 1e-5 – 1e-4 for RQ-spline flows',
      },
      {
        key: 'grad_clip',
        value: '1.0',
        type: 'float',
        short: 'Maximum L2 norm of the full gradient vector.',
        long: 'Before each optimiser step, PyTorch computes the global gradient norm across all parameters. If that norm exceeds grad_clip, every gradient is scaled down proportionally. Relaxed from 0.5 to 1.0 because with the reduced model capacity (4 layers, 128 dim) and lower LR (5e-5), gradients are naturally smaller — 0.5 was clipping legitimate learning signal in early epochs.',
        tradeoff: { up: 'Prevents training instability from gradient spikes', down: 'Too tight (< 0.5) suppresses legitimate gradients and slows convergence' },
        range: 'Typical: 0.5 – 5.0',
      },
      {
        key: 'weight_decay',
        value: '5e-3',
        type: 'float',
        short: 'AdamW L2 regularisation on weight magnitudes.',
        long: 'Increased from 1e-4 to 5e-3 to fight the strong overfitting observed in earlier runs (train loss −15, val loss +38). At 5e-3, AdamW actively shrinks weight magnitudes every step, preventing the MLP neurons from growing large enough to memorise the specific training time period. The val/train gap narrowed significantly after this increase.',
        tradeoff: { up: 'Stronger regularisation reduces train/val gap', down: 'Too high (> 1e-2) crushes spline knot gradients — the flow cannot fit the data' },
        range: 'Safe range: 1e-3 – 1e-2 when overfitting is observed',
      },
      {
        key: 'lr_scheduler_factor',
        value: '0.5',
        type: 'float',
        short: 'Multiplier applied to LR on each plateau.',
        long: 'When ReduceLROnPlateau detects no val_loss improvement for lr_scheduler_patience epochs, it performs: new_lr = old_lr × factor. factor=0.5 halves the learning rate. This is applied repeatedly — if the model plateaus again, LR halves again, down to lr_scheduler_min.',
        tradeoff: { up: 'Aggressive decay (0.3) escapes plateaus faster', down: 'Too aggressive risks permanently under-stepping in early training' },
        range: 'Typical: 0.3 – 0.7',
      },
      {
        key: 'lr_scheduler_patience',
        value: '3',
        type: 'int',
        short: 'Epochs of no improvement before LR is halved.',
        long: 'The scheduler monitors val_loss at the end of each epoch. If val_loss does not decrease for this many consecutive epochs, it triggers a LR reduction. patience=3 means: tolerate 3 "stuck" epochs before intervening. Too low (1) causes premature LR decay — the model never gets a chance to work through a local saddle point.',
        tradeoff: { up: 'Lower patience = faster reaction to plateaus', down: 'Too low = premature LR decay, training stalls' },
        range: 'Typical: 3 – 10',
      },
      {
        key: 'lr_scheduler_min',
        value: '1e-6',
        type: 'float',
        short: 'Hard floor — LR never drops below this.',
        long: 'Without a minimum, repeated LR halvings would eventually reduce the learning rate to effectively zero, causing the optimiser to make no meaningful progress. lr_scheduler_min caps the decay, ensuring training always continues at some minimum pace.',
        tradeoff: { up: 'Prevents the optimiser from freezing entirely', down: 'Setting too high prevents fine-tuning in late epochs' },
        range: 'Typical: 1e-7 – 1e-5',
      },
      {
        key: 'early_stopping_patience',
        value: '15',
        type: 'int',
        short: 'Epochs of no val improvement before training terminates.',
        long: 'Stops training early if val_loss has not improved for this many consecutive epochs, saving compute and preserving the best checkpoint. Critically, this must be significantly larger than lr_scheduler_patience (3) — the rule is: early_stopping_patience ≥ lr_scheduler_patience × number_of_LR_decay_phases_you_want. With patience=15 and scheduler_patience=3, the LR halves at +3, +6, +9, +12 epochs after the best — giving 4 full recovery windows before giving up.',
        tradeoff: { up: 'More patience = more chances for LR decay to rescue a plateau', down: 'Too high wastes compute on a run that will never recover' },
        range: 'Rule: always > lr_scheduler_patience × 3',
      },
    ],
  },
  {
    id: 'model',
    label: 'Model Architecture',
    icon: Cpu,
    color: 'violet',
    bg: 'bg-violet-50',
    border: 'border-violet-200',
    badge: 'bg-violet-100 text-violet-700',
    iconColor: 'text-violet-500',
    params: [
      {
        key: 'num_layers',
        value: '4',
        type: 'int',
        short: 'Number of NeuralSplineCouplingLayer blocks stacked.',
        long: 'Each coupling layer sees the full condition vector (x: pressure, flow, temperature) and transforms exactly half the theta dimensions. Reduced from 6 → 4 to reduce total model capacity — the condition is only 3-dimensional (pressure, temperature, fuel gas LHV), which cannot provide enough signal to justify 6 deep coupling layers. Fewer layers also means fewer parameters to regularise against the chronological train/val distribution shift.',
        tradeoff: { up: 'More layers → richer, more accurate posterior', down: 'Too many layers with a weak condition → overfitting to training time period' },
        range: 'Typical: 4 – 8 for 3-dim conditioning',
      },
      {
        key: 'hidden_dim',
        value: '128',
        type: 'int',
        short: 'Width of every linear layer inside each ResidualMLP.',
        long: 'Each coupling layer contains a ResidualMLP that maps (half_theta + condition) → spline knot parameters. Reduced from 256 → 128 because the input condition is only 3-dimensional — a 256-wide MLP receiving 3 numbers has an enormous capacity mismatch that encourages overfitting. 128 neurons is sufficient to capture non-linear interactions between pressure, temperature, and fuel LHV.',
        tradeoff: { up: 'Wider MLP → richer conditional dependencies captured', down: 'Oversized relative to a 3-dim condition → overfitting' },
        range: '64–256 for 3-dim conditioning',
      },
      {
        key: 'num_bins',
        value: '8',
        type: 'int',
        short: 'Knot count for the Rational-Quadratic spline.',
        long: 'The spline divides [-bound, +bound] into num_bins intervals. Each bin has a learned width, height, and slope — giving the flow fine-grained control over how it warps each dimension. More bins = smoother, more precise transforms, but each bin adds 3 learned parameters per output dimension per coupling layer, increasing the total parameter count substantially.',
        tradeoff: { up: 'More bins → smoother density, better tail modelling', down: 'Each extra bin adds 3K params per layer (K = output dims)' },
        range: 'Typical: 4 – 16',
      },
      {
        key: 'bound',
        value: '9.0',
        type: 'float',
        short: 'The spline only acts inside [−bound, +bound].',
        long: 'Outside this range the flow applies the identity transform — z = θ unchanged. A diagnostic run revealed that with bound=5.0, 6.6% of training samples and 7.6% of validation samples had at least one PCA coefficient outside [−5, 5] (max observed: PC10 val = −7.82). These samples received identity treatment, and when the model scored them under the MultivariateNormal, the raw z-values of ±7 produced losses of +150 to +300 per sample — the catastrophic val spikes. bound=9.0 covers 99.9%+ of all samples.',
        tradeoff: { up: 'Covers outlier coefficients — eliminates catastrophic val loss spikes', down: 'Same 8 bins spread over [−9, +9] vs [−5, +5] = coarser per-bin resolution' },
        range: 'Must exceed max(|PCA coefficient|) across all splits — currently 7.82',
      },
      {
        key: 'dropout_rate',
        value: '0.2',
        type: 'float',
        short: 'Fraction of neurons zeroed per ResidualMLP forward pass.',
        long: 'During training only, each neuron is independently zeroed with probability dropout_rate. Increased from 0.1 → 0.2 to strengthen regularisation against the chronological distribution shift. Because the val period represents a different operating regime (PC02 val std = 0.14 vs train std = 1.0), dropout forces the MLP to learn regime-agnostic features rather than memorising the specific patterns of the training period.',
        tradeoff: { up: 'Stronger regularisation narrows train/val gap', down: 'Too high (> 0.3) adds noise to spline parameter predictions, destabilising knots' },
        range: 'Typical: 0.1 – 0.3 when chronological overfitting is present',
      },
      {
        key: 'mlp_layers',
        value: '2',
        type: 'int',
        short: 'Depth of residual blocks inside each coupling layer\'s MLP.',
        long: 'Each ResidualMLP consists of mlp_layers sequential residual blocks: Linear → GELU → Dropout → Linear + skip-connection. Reduced from 4 → 2 to match the reduced hidden_dim=128. A 3-dimensional input condition (x) does not require 4 deep blocks to extract useful features — 2 blocks is sufficient to capture non-linear interactions, and fewer blocks means fewer parameters to overfit.',
        tradeoff: { up: 'Deeper MLP → richer conditional parameter functions', down: 'Overkill for a 3-dim condition — extra blocks memorise training patterns' },
        range: 'Typical: 2 – 4 for low-dimensional conditioning',
      },
    ],
  },
  {
    id: 'dataloader',
    label: 'Dataloader',
    icon: Database,
    color: 'sky',
    bg: 'bg-sky-50',
    border: 'border-sky-200',
    badge: 'bg-sky-100 text-sky-700',
    iconColor: 'text-sky-500',
    params: [
      {
        key: 'num_workers',
        value: '12',
        type: 'int',
        short: 'Parallel CPU processes that prefetch batches.',
        long: 'PyTorch\'s DataLoader spawns this many subprocesses, each independently reading parquet rows and assembling tensors. While the GPU is running a forward/backward pass on batch N, workers are already loading batch N+1 (and beyond). The M4 Max has 14 performance cores — 12 workers leaves 2 free for the OS and MPS runtime, achieving near-zero GPU idle time.',
        tradeoff: { up: 'More workers = less GPU idle time between batches', down: 'Each worker consumes ~200 MB RAM; too many can exhaust memory' },
        range: '4 on quad-core; 12 on M4 Max',
      },
      {
        key: 'prefetch_factor',
        value: '4',
        type: 'int',
        short: 'Batches each worker queues ahead of time.',
        long: 'Each of the num_workers processes will pre-load prefetch_factor batches into a shared queue. With 12 workers × 4 prefetch = 48 batches always ready in RAM, the GPU never waits for a batch regardless of parquet read latency spikes. Safe with 128 GB unified memory — total prefetch RAM = 48 × 4096 × (12 + 3) floats × 4 bytes ≈ 12 GB.',
        tradeoff: { up: 'Higher prefetch = more I/O pipeline depth', down: 'Multiplies the warm-up memory footprint; reduce to 2 on RAM-limited machines' },
        range: 'Typical: 2 – 8',
      },
      {
        key: 'persistent_workers',
        value: 'true',
        type: 'bool',
        short: 'Keep worker processes alive between epochs.',
        long: 'Without this, Python re-forks all num_workers processes at the start of every single epoch. Each fork has OS overhead of ~150–300 ms and must re-import libraries and re-open parquet files. With persistent_workers=true, workers stay alive and simply block on an empty queue between epochs. On a 20-epoch run with 12 workers, this saves ~60 seconds of pure overhead.',
        tradeoff: { up: 'Eliminates ~2s per-epoch process spawn overhead', down: 'Workers hold file handles and RAM even when idle; not an issue on 128 GB' },
        range: 'Always true unless debugging worker crashes',
      },
    ],
  },
  {
    id: 'logging',
    label: 'Logging & Hardware',
    icon: BarChart2,
    color: 'emerald',
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    badge: 'bg-emerald-100 text-emerald-700',
    iconColor: 'text-emerald-500',
    params: [
      {
        key: 'wandb',
        value: 'true',
        type: 'bool',
        short: 'Master switch for Weights & Biases experiment tracking.',
        long: 'When true, train.py calls wandb.init() before the training loop, and the trainer logs train/val loss, learning rate, and the model graph every 50 steps. All runs are grouped under wandb_project on the W&B dashboard, enabling side-by-side comparison of hyperparameter sweeps. Set to false for quick offline debug runs — all wandb calls are conditionally skipped.',
        tradeoff: { up: 'Persistent loss curves, run comparison, model graph', down: 'Adds ~10 MB/epoch network overhead; requires wandb login' },
        range: 'true / false',
      },
      {
        key: 'wandb_project',
        value: '"smpc-spline-flow"',
        type: 'string',
        short: 'W&B project name that groups all runs.',
        long: 'Runs logged with the same project name appear together at https://wandb.ai/<username>/smpc-spline-flow. You can compare loss curves, LR schedules, and hyperparameter configurations across all runs in one view. Change this if you want to isolate a new experiment family (e.g. "smpc-ablation-num_bins") from previous runs.',
        tradeoff: { up: 'Logical grouping of related experiment runs', down: 'No runtime effect — purely an organisational label' },
        range: 'Any string — no spaces recommended',
      },
      {
        key: 'use_bf16',
        value: 'false',
        type: 'bool',
        short: 'Enable bfloat16 autocast during training.',
        long: 'BF16 has the same 8-bit exponent range as FP32 (so no overflow risk unlike FP16) but only 7 mantissa bits. On CUDA A100/H100, this gives ~1.5–2× speedup with negligible accuracy loss. On M4 Max MPS, torch.autocast with bfloat16 is currently unstable for normalising flow log-det computations — the jacobian log-determinant accumulates relative rounding errors that manifest as NaN losses around epoch 5–10. Leave false until PyTorch MPS autocast matures.',
        tradeoff: { up: '~1.5-2× faster on CUDA hardware', down: 'Numerically unstable for flow log-det on MPS — keep false on Apple Silicon' },
        range: 'true only on CUDA A100/H100',
      },
    ],
  },
];

// ─────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────

const TypeBadge = ({ type }) => {
  const styles = {
    int:    'bg-amber-100 text-amber-700',
    float:  'bg-purple-100 text-purple-700',
    bool:   'bg-green-100 text-green-700',
    string: 'bg-blue-100 text-blue-700',
  };
  return (
    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded uppercase tracking-wide ${styles[type] ?? 'bg-slate-100 text-slate-600'}`}>
      {type}
    </span>
  );
};

const ParamCard = ({ param, badge, bg, border }) => {
  const [open, setOpen] = useState(false);

  return (
    <div className={`rounded-xl border ${border} ${open ? bg : 'bg-white'} transition-colors duration-200`}>
      {/* Header row — always visible */}
      <button
        className="w-full text-left px-4 py-3 flex items-start gap-3"
        onClick={() => setOpen(v => !v)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-0.5">
            <code className="text-sm font-bold text-slate-800 font-mono">{param.key}</code>
            <TypeBadge type={param.type} />
            <span className={`text-xs font-mono font-semibold px-2 py-0.5 rounded-full ${badge}`}>
              {param.value}
            </span>
          </div>
          <p className="text-xs text-slate-500 leading-snug">{param.short}</p>
        </div>
        <div className="flex-shrink-0 mt-0.5 text-slate-400">
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="px-4 pb-4 space-y-3 border-t border-slate-100 pt-3">
          <p className="text-sm text-slate-700 leading-relaxed">{param.long}</p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className="bg-green-50 border border-green-100 rounded-lg px-3 py-2 text-xs">
              <span className="font-semibold text-green-700 block mb-0.5">↑ Increase</span>
              <span className="text-green-800">{param.tradeoff.up}</span>
            </div>
            <div className="bg-red-50 border border-red-100 rounded-lg px-3 py-2 text-xs">
              <span className="font-semibold text-red-700 block mb-0.5">↓ Decrease / Risk</span>
              <span className="text-red-800">{param.tradeoff.down}</span>
            </div>
          </div>

          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <Sliders size={12} />
            <span>{param.range}</span>
          </div>
        </div>
      )}
    </div>
  );
};

const SectionBlock = ({ section }) => {
  const Icon = section.icon;
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`rounded-2xl border ${section.border} overflow-hidden`}>
      {/* Section header */}
      <button
        className={`w-full flex items-center gap-3 px-5 py-4 ${section.bg} text-left`}
        onClick={() => setCollapsed(v => !v)}
      >
        <Icon size={18} className={section.iconColor} />
        <span className="font-bold text-slate-800 text-base flex-1">{section.label}</span>
        <span className="text-xs text-slate-400 font-mono mr-2">
          {section.params.length} param{section.params.length !== 1 ? 's' : ''}
        </span>
        {collapsed ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronUp size={16} className="text-slate-400" />}
      </button>

      {/* Param cards */}
      {!collapsed && (
        <div className="p-4 space-y-2 bg-white">
          {section.params.map(p => (
            <ParamCard
              key={p.key}
              param={p}
              badge={section.badge}
              bg={section.bg}
              border={section.border}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// MAIN EXPORT
// ─────────────────────────────────────────────
export default function HyperparamSection() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <Settings size={22} className="text-slate-600" />
          <h2 className="text-2xl font-bold text-slate-800">Hyperparameter Reference</h2>
        </div>
        <p className="text-slate-500 text-sm leading-relaxed">
          Every value in <code className="bg-slate-100 px-1.5 py-0.5 rounded font-mono text-xs">configs/train.yaml</code> is documented here.
          Click any parameter to expand its full explanation, trade-off analysis, and safe tuning range.
        </p>
      </div>

      {/* Info banner */}
      <div className="flex items-start gap-3 bg-blue-50 border border-blue-200 rounded-xl px-4 py-3">
        <Info size={16} className="text-blue-500 mt-0.5 flex-shrink-0" />
        <p className="text-xs text-blue-800 leading-relaxed">
          <strong>Single source of truth.</strong> Nothing is hardcoded in <code className="font-mono">train.py</code>, <code className="font-mono">trainer.py</code>, or any model file.
          Change a value here and it propagates automatically through the entire pipeline on the next run.
        </p>
      </div>

      {/* YAML at a glance */}
      <div className="rounded-xl border border-slate-200 overflow-hidden">
        <div className="bg-slate-800 px-4 py-2 flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-400" />
          <div className="w-3 h-3 rounded-full bg-yellow-400" />
          <div className="w-3 h-3 rounded-full bg-green-400" />
          <span className="ml-2 text-slate-400 text-xs font-mono">configs/train.yaml — quick view</span>
        </div>
        <pre className="bg-slate-900 text-slate-300 text-xs font-mono p-4 overflow-x-auto leading-relaxed">
{`training:
  epochs: 60                    batch_size: 4096
  learning_rate: 5e-5           grad_clip: 1.0
  weight_decay: 5e-3            lr_scheduler_factor: 0.5
  lr_scheduler_patience: 3      lr_scheduler_min: 1e-6
  early_stopping_patience: 15

model:
  num_layers: 4    hidden_dim: 128
  num_bins: 8      bound: 9.0
  dropout_rate: 0.2   mlp_layers: 2

dataloader:
  num_workers: 12  prefetch_factor: 4
  persistent_workers: true

logging:
  wandb: true   wandb_project: "smpc-spline-flow"
  use_bf16: false`}
        </pre>
      </div>

      {/* Sections */}
      <div className="space-y-4">
        {SECTIONS.map(s => <SectionBlock key={s.id} section={s} />)}
      </div>
    </div>
  );
}
