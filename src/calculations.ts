import { MemoryMode, ModelQuantization, KvCacheQuantization, Recommendation } from './types';

const BASE_CONTEXT_LENGTH = 2048;
const KV_CACHE_ALPHA = 0.2; // fraction representing typical KV cache overhead
const DISK_SIZE_OVERHEAD = 1.1; // ~10% overhead

/**
 * Returns bits-based multiplier for the main model based on quantization
 */
export const getModelQuantFactor = (q: ModelQuantization): number => {
  switch (q) {
    case "F32":
      return 4.0;
    case "F16":
      return 2.0;
    case "Q8":
      return 1.0;
    case "Q6":
      return 0.75;
    case "Q5":
      return 0.625;
    case "Q4":
      return 0.5;
    case "Q3":
      return 0.375;
    case "Q2":
      return 0.25;
    case "GPTQ":
      return 0.4;
    case "AWQ":
      return 0.35;
    default:
      return 1.0; // fallback
  }
};

/**
 * Returns bits-based multiplier for KV cache based on quantization
 */
export const getKvCacheQuantFactor = (k: KvCacheQuantization): number => {
  switch (k) {
    case "F32":
      return 4.0;
    case "F16":
      return 2.0;
    case "Q8":
      return 1.0;
    case "Q5":
      return 0.625;
    case "Q4":
      return 0.5;
    default:
      return 1.0; // fallback
  }
};

/**
 * Calculate VRAM for single-user inference.
 * Split into Model Memory + KV Cache Memory.
 */
export const calculateRequiredVram = (
  params: number,
  modelQuant: ModelQuantization,
  contextLength: number,
  useKvCache: boolean,
  kvCacheQuant: KvCacheQuantization
): number => {
  // 1) Model memory
  const modelFactor = getModelQuantFactor(modelQuant);
  const baseModelMem = params * modelFactor; // GB if 1B params

  // 2) Context scaling
  let contextScale = contextLength / BASE_CONTEXT_LENGTH;
  if (contextScale < 1) contextScale = 1;
  const modelMem = baseModelMem * contextScale;

  // 3) KV cache memory (if enabled)
  let kvCacheMem = 0;
  if (useKvCache) {
    const kvFactor = getKvCacheQuantFactor(kvCacheQuant);
    kvCacheMem = params * kvFactor * contextScale * KV_CACHE_ALPHA;
  }

  // 4) total
  return modelMem + kvCacheMem;
};

/**
 * For unified memory, up to 75% of system RAM can be used as VRAM
 */
export const getMaxUnifiedVram = (memGB: number, ramUsage: number): number =>
  memGB * (ramUsage / 100);

/**
 * Calculate the hardware recommendation based on the model and system configuration
 */
export const calculateHardwareRecommendation = (
  params: number,
  modelQuant: ModelQuantization,
  contextLength: number,
  useKvCache: boolean,
  kvCacheQuant: KvCacheQuantization,
  memoryMode: MemoryMode,
  systemMemory: number,
  gpuVram: number,
  ramUsage: number
): Recommendation => {
  const requiredVram = calculateRequiredVram(
    params,
    modelQuant,
    contextLength,
    useKvCache,
    kvCacheQuant
  );
  const recSystemMemory = systemMemory;

  if (memoryMode === "UNIFIED_MEMORY") {
    const unifiedLimit = getMaxUnifiedVram(recSystemMemory, ramUsage);
    if (requiredVram <= unifiedLimit) {
      return {
        gpuType: "Unified memory (ex: Apple silicon, AMD Ryzen™ Al Max 395)",
        vramNeeded: requiredVram.toFixed(1),
        fitsUnified: true,
        systemRamNeeded: recSystemMemory,
        gpusRequired: 1,
      };
    } else {
      return {
        gpuType: "Unified memory (insufficient)",
        vramNeeded: requiredVram.toFixed(1),
        fitsUnified: false,
        systemRamNeeded: recSystemMemory,
        gpusRequired: 0,
      };
    }
  }

  // Discrete GPU
  const singleGpuVram = gpuVram;
  if (requiredVram <= singleGpuVram) {
    return {
      gpuType: `Single ${singleGpuVram}GB GPU`,
      vramNeeded: requiredVram.toFixed(1),
      fitsUnified: false,
      systemRamNeeded: Math.max(recSystemMemory, requiredVram),
      gpusRequired: 1,
    };
  } else {
    // multiple GPUs
    const count = Math.ceil(requiredVram / singleGpuVram);
    return {
      gpuType: `Discrete GPUs (${singleGpuVram}GB each)`,
      vramNeeded: requiredVram.toFixed(1),
      fitsUnified: false,
      systemRamNeeded: Math.max(recSystemMemory, requiredVram),
      gpusRequired: count,
    };
  }
};

/**
 * Estimate on-disk model size (GB). We do NOT factor in KV here.
 */
export const calculateOnDiskSize = (
  params: number,
  modelQuant: ModelQuantization
): number => {
  let bitsPerParam: number;
  switch (modelQuant) {
    case "F32":
      bitsPerParam = 32;
      break;
    case "F16":
      bitsPerParam = 16;
      break;
    case "Q8":
      bitsPerParam = 8;
      break;
    case "Q6":
      bitsPerParam = 6;
      break;
    case "Q5":
      bitsPerParam = 5;
      break;
    case "Q4":
      bitsPerParam = 4;
      break;
    case "Q3":
      bitsPerParam = 3;
      break;
    case "Q2":
      bitsPerParam = 2;
      break;
    case "GPTQ":
      bitsPerParam = 4;
      break;
    case "AWQ":
      bitsPerParam = 4;
      break;
    default:
      bitsPerParam = 8;
      break;
  }

  const totalBits = params * 1e9 * bitsPerParam;
  const bytes = totalBits / 8;
  const gigabytes = bytes / 1e9;
  return gigabytes * DISK_SIZE_OVERHEAD;
};