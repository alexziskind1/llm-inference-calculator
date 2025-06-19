import { useState } from 'react';
import './App.css';
import { MemoryMode, ModelQuantization, KvCacheQuantization } from './types';
import {
  calculateHardwareRecommendation,
  calculateOnDiskSize,
} from './calculations';
import { Tooltip } from './components/Tooltip';
import { ThemeToggle } from './components/ThemeToggle';

function App() {
  // -----------------------------------
  // 1. STATE
  // -----------------------------------

  // Model config
  const [params, setParams] = useState<number>(65); // Billions of parameters
  const [modelQuant, setModelQuant] = useState<ModelQuantization>('Q4');

  // KV Cache
  const [useKvCache, setUseKvCache] = useState<boolean>(false); // Changed from true to false
  const [kvCacheQuant, setKvCacheQuant] = useState<KvCacheQuantization>('Q4'); // Changed from 'F16' to 'Q4'

  // Inference mode
  const [inferenceMode, setInferenceMode] = useState<'incremental' | 'bulk'>(
    'incremental'
  );

  // Misc
  const [contextLength, setContextLength] = useState<number>(4096);
  const [memoryMode, setMemoryMode] = useState<MemoryMode>('DISCRETE_GPU');
  const [systemMemory, setSystemMemory] = useState<number>(128); // in GB
  const [gpuVram, setGpuVram] = useState<number>(24); // in GB, default 24GB

  // -----------------------------------
  // 2. HELPER FUNCTIONS
  // -----------------------------------

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement>,
    setter: React.Dispatch<React.SetStateAction<number>>
  ) => {
    const newValue = Number(event.target.value);
    if (!isNaN(newValue)) {
      setter(newValue);
    }
  };

  // -----------------------------------
  // 3. CALCULATE & RENDER
  // -----------------------------------
  const recommendation = calculateHardwareRecommendation(
    params,
    modelQuant,
    contextLength,
    useKvCache,
    kvCacheQuant,
    memoryMode,
    systemMemory,
    gpuVram,
    inferenceMode
  );

  const onDiskSize = calculateOnDiskSize(params, modelQuant);

  return (
    <div className="App">
      <ThemeToggle />
      <h1>LLM Inference Hardware Calculator</h1>
      <p className="intro-text">
        Estimate VRAM & System RAM for single-user inference (Batch=1).
        <br />
        Model quant & KV cache quant are configured separately.
      </p>

      <div className="layout">
        {/* Left Panel: Inputs */}
        <div className="input-panel">
          <h2 className="section-title">Model Configuration</h2>

          <label className="label-range">
            Number of Parameters (Billions):
            <input
              className="text-input-group"
              type="number"
              min={1}
              max={1000}
              value={params}
              onChange={(e) => handleInputChange(e, setParams)}
            />
            <Tooltip text="The total number of model parameters in billions. For example, '13' means a 13B model.">
              i
            </Tooltip>
          </label>
          <div className="slider-input-group">
            <input
              type="range"
              min={1}
              max={1000}
              value={params}
              onChange={(e) => setParams(Number(e.target.value))}
            />
          </div>
          <label className="label-range">
            Model Quantization:
            <Tooltip text="The data format used to store model weights in GPU memory. For instance, F16 uses ~2GB per 1B params, Q4 ~0.5GB, etc.">
              i
            </Tooltip>
          </label>
          <select
            value={modelQuant}
            onChange={(e) => setModelQuant(e.target.value as ModelQuantization)}
          >
            {/* F32, F16, Q8, Q6, Q5, Q4, Q3, Q2, GPTQ, AWQ */}
            <option value="F32">F32</option>
            <option value="F16">F16</option>
            <option value="Q8">Q8</option>
            <option value="Q6">Q6</option>
            <option value="Q5">Q5</option>
            <option value="Q4">Q4</option>
            <option value="Q3">Q3</option>
            <option value="Q2">Q2</option>
            <option value="GPTQ">GPTQ</option>
            <option value="AWQ">AWQ</option>
          </select>

          <label className="label-range">
            Context Length (Tokens):
            <input
              className="text-input-group"
              type="number"
              min={128}
              max={32768}
              step={128}
              value={contextLength}
              onChange={(e) => handleInputChange(e, setContextLength)}
            />
            <Tooltip text="Maximum tokens (including prompt and history) available at once. Larger context = more memory usage.">
              i
            </Tooltip>
          </label>
          <div className="slider-input-group">
            <input
              type="range"
              min={128}
              max={32768}
              step={128}
              value={contextLength}
              onChange={(e) => setContextLength(Number(e.target.value))}
            />
          </div>

          {/* Inference Mode */}
          <label className="label-range">
            Inference Mode:
            <Tooltip text="'Incremental' is streaming token-by-token generation, 'Bulk' processes the entire context in one pass.">
              i
            </Tooltip>
          </label>
          <select
            value={inferenceMode}
            onChange={(e) =>
              setInferenceMode(e.target.value as 'incremental' | 'bulk')
            }
          >
            <option value="incremental">Incremental (streaming)</option>
            <option value="bulk">Bulk (all at once)</option>
          </select>

          {/* KV Cache Toggle */}
          <div className="checkbox-row">
            <input
              type="checkbox"
              checked={useKvCache}
              onChange={() => setUseKvCache(!useKvCache)}
              id="kvCache"
            />
            <label htmlFor="kvCache">
              Enable KV Cache
              <Tooltip text="Reuses key/value attention states to accelerate decoding, at the cost of additional VRAM.">
                i
              </Tooltip>
            </label>
          </div>

          {/* 
             (Animated) KV Cache Quant Section:
             We'll wrap it in a div that transitions "max-height"
             so the UI doesn't jump abruptly.
          */}
          <div className={`kvCacheAnimate ${useKvCache ? 'open' : 'closed'}`}>
            <label className="label-range">
              KV Cache Quantization:
              <Tooltip text="Data format for KV cache memory usage. Lower precision reduces memory but may affect performance/quality.">
                i
              </Tooltip>
            </label>
            <select
              value={kvCacheQuant}
              onChange={(e) =>
                setKvCacheQuant(e.target.value as KvCacheQuantization)
              }
            >
              <option value="F32">F32</option>
              <option value="F16">F16</option>
              <option value="Q8">Q8</option>
              <option value="Q5">Q5</option>
              <option value="Q4">Q4</option>
            </select>
          </div>

          <hr style={{ margin: '1rem 0' }} />

          <h2 className="section-title">System Configuration</h2>

          <label className="label-range">System Type:</label>
          <select
            value={memoryMode}
            onChange={(e) => setMemoryMode(e.target.value as MemoryMode)}
          >
            <option value="DISCRETE_GPU">Discrete GPU</option>
            <option value="UNIFIED_MEMORY">
              Unified memory (ex: Apple silicon, AMD Ryzen™ Al Max+ 395)
            </option>
          </select>

          {memoryMode === 'DISCRETE_GPU' && (
            <>
              <label className="label-range">GPU VRAM (GB):</label>
              <select
                value={gpuVram}
                onChange={(e) => setGpuVram(Number(e.target.value))}
              >
                <option value={8}>8</option>
                <option value={12}>12</option>
                <option value={16}>16</option>
                <option value={20}>20</option>
                <option value={24}>24</option>
                <option value={32}>32</option>
                <option value={40}>40</option>
                <option value={48}>48</option>
                <option value={80}>80</option>
              </select>
            </>
          )}

          <label className="label-range">
            System Memory (GB):
            <input
              className="text-input-group"
              type="number"
              min={8}
              max={512}
              step={8}
              value={systemMemory}
              onChange={(e) => handleInputChange(e, setSystemMemory)}
            />
          </label>
          <div className="slider-input-group">
            <input
              type="range"
              min={8}
              max={512}
              step={8}
              value={systemMemory}
              onChange={(e) => setSystemMemory(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Right Panel: Results */}
        <div className="results-panel">
          <h2 className="section-title">Hardware Requirements</h2>

          <p>
            <strong>VRAM Needed:</strong>{' '}
            <span className="result-highlight">
              {recommendation.vramNeeded} GB
            </span>
          </p>
          <p>
            <strong>On-Disk Size:</strong>{' '}
            <span className="result-highlight">{onDiskSize.toFixed(2)} GB</span>
          </p>
          <p>
            <strong>GPU Config:</strong> {recommendation.gpuType}
          </p>

          {recommendation.gpusRequired > 1 && (
            <p>
              <strong>Number of GPUs Required:</strong>{' '}
              {recommendation.gpusRequired}
            </p>
          )}
          {recommendation.gpusRequired === 1 && (
            <p>
              <strong>Number of GPUs Required:</strong> 1 (Fits on a single GPU)
            </p>
          )}

          {memoryMode === 'DISCRETE_GPU' && (
            <p>
              <strong>System RAM:</strong>{' '}
              {recommendation.systemRamNeeded.toFixed(1)} GB
            </p>
          )}

          {memoryMode === 'UNIFIED_MEMORY' && recommendation.fitsUnified && (
            <p style={{ color: 'green' }}>✅ Fits in unified memory!</p>
          )}
          {memoryMode === 'UNIFIED_MEMORY' && !recommendation.fitsUnified && (
            <p style={{ color: 'red' }}>
              ⚠️ Exceeds unified memory. Increase system RAM or reduce model
              size.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
