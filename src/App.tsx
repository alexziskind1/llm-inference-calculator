import { useState, useMemo } from 'react';
import './App.css';
import { MemoryMode, ModelQuantization, KvCacheQuantization } from './types';
import {
  calculateHardwareRecommendation,
  calculateOnDiskSize,
} from './calculations';

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

  // Misc
  const [contextLength, setContextLength] = useState<number>(4096);
  const [memoryMode, setMemoryMode] = useState<MemoryMode>('DISCRETE_GPU');
  const [systemMemory, setSystemMemory] = useState<number>(128); // in GB
  const [gpuVram, setGpuVram] = useState<number>(24); // in GB, default 24GB
  const [ramUsage, setRamUsage] = useState<number>(75); // Percentage of system memory usable for LLM (max 90%)

  // -----------------------------------
  // 2. HELPER FUNCTIONS
  // -----------------------------------

  const handleInputChange = (
    { target: { value } }: React.ChangeEvent<HTMLInputElement>,
    setter: React.Dispatch<React.SetStateAction<number>>
  ) => {
    const newValue = Number(value);
    if (!isNaN(newValue)) {
      setter(newValue);
    }
  };

  // -----------------------------------
  // 3. CALCULATE & RENDER
  // -----------------------------------
  const recommendation = useMemo(
    () =>
      calculateHardwareRecommendation(
        params,
        modelQuant,
        contextLength,
        useKvCache,
        kvCacheQuant,
        memoryMode,
        systemMemory,
        gpuVram,
        ramUsage
      ),
    [
      params,
      modelQuant,
      contextLength,
      useKvCache,
      kvCacheQuant,
      memoryMode,
      systemMemory,
      gpuVram,
      ramUsage,
    ]
  );

  const onDiskSize = useMemo(
    () => calculateOnDiskSize(params, modelQuant),
    [params, modelQuant]
  );

  return (
    <div className="App">
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
          <label className="label-range">Model Quantization:</label>
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

          {/* KV Cache Toggle */}
          <div className="checkbox-row">
            <input
              type="checkbox"
              checked={useKvCache}
              onChange={(e) => setUseKvCache(e.target.checked)}
              id="kvCache"
            />
            <label htmlFor="kvCache">Enable KV Cache</label>
          </div>

          {/* 
             (Animated) KV Cache Quant Section:
             We'll wrap it in a div that transitions "max-height"
             so the UI doesn't jump abruptly.
          */}
          <div className={`kvCacheAnimate ${useKvCache ? "open" : "closed"}`}>
            <label className="label-range">KV Cache Quantization:</label>
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

          <hr style={{ margin: "1rem 0" }} />

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

          {memoryMode === "DISCRETE_GPU" && (
            <>
              <label className="label-range">GPU VRAM (GB):</label>
              <select
                value={gpuVram}
                onChange={(e) => setGpuVram(Number(e.target.value))}
              >
                <option value={8}>8</option>
                <option value={12}>12</option>
                <option value={16}>16</option>
                <option value={24}>24</option>
                <option value={32}>32</option>
                <option value={40}>40</option>
                <option value={48}>48</option>
                <option value={80}>80</option>
              </select>
            </>
          )}

          {memoryMode === "UNIFIED_MEMORY" && (
            <>
              <label className="label-range">
                RAM Usage (%):
                <input
                  className="text-input-group"
                  type="number"
                  min={1}
                  max={90}
                  value={ramUsage}
                  onChange={(e) =>
                    setRamUsage(Math.min(Number(e.target.value), 90))
                  }
                />
              </label>
              <div className="slider-input-group">
                <input
                  type="range"
                  min={1}
                  max={90}
                  value={ramUsage}
                  onChange={(e) =>
                    setRamUsage(Math.min(Number(e.target.value), 90))
                  }
                />
              </div>
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
            </>
          )}
        </div>

        {/* Right Panel: Results */}
        <div className="results-panel">
          <h2 className="section-title">Hardware Requirements</h2>

          <p>
            <strong>VRAM Needed:</strong>{" "}
            <span className="result-highlight">
              {recommendation.vramNeeded} GB{" "}
              <span className="result-standard"></span>
              {memoryMode === "UNIFIED_MEMORY" &&
                `(with ${ramUsage}% RAM Usage)`}
            </span>
          </p>
          <p>
            <strong>On-Disk Size:</strong>{" "}
            <span className="result-highlight">{onDiskSize.toFixed(2)} GB</span>
          </p>
          <p>
            <strong>GPU Config:</strong> {recommendation.gpuType}
          </p>

          {recommendation.gpusRequired > 1 && (
            <p>
              <strong>Number of GPUs Required:</strong>{" "}
              {recommendation.gpusRequired}
            </p>
          )}
          {recommendation.gpusRequired === 1 && (
            <p>
              <strong>Number of GPUs Required:</strong> 1 (Fits on a single GPU)
            </p>
          )}

          {memoryMode === "UNIFIED_MEMORY" && (
            <>
              <p>
                <strong>System RAM:</strong>{" "}
                {recommendation.systemRamNeeded.toFixed(1)} GB
              </p>
            </>
          )}

          {memoryMode === "UNIFIED_MEMORY" && recommendation.fitsUnified && (
            <p style={{ color: "green" }}>✅ Fits in unified memory!</p>
          )}
          {memoryMode === "UNIFIED_MEMORY" && !recommendation.fitsUnified && (
            <p style={{ color: "red" }}>
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
