import { MemoryMode, ModelQuantization, KvCacheQuantization, Recommendation } from './types';

type InferenceMode = 'incremental' | 'bulk';

/**
 * Returns GB factor for a 1B param model based on quantization.
 * This function converts the quantization setting into an estimate of how many
 * gigabytes are required per 1B parameters. For example, FP16 (F16) uses 2GB per 1B.
 */
export const getModelQuantFactor = (q: ModelQuantization): number => {
  switch (q) {
    case 'F32':  return 4.0;
    case 'F16':  return 2.0;
    case 'Q8':   return 1.0;
    case 'Q6':   return 0.75;
    case 'Q5':   return 0.625;
    case 'Q4':   return 0.5;
    case 'Q3':   return 0.375;
    case 'Q2':   return 0.25;
    case 'GPTQ': return 0.4;  // approximate factor for GPTQ quantization
    case 'AWQ':  return 0.35; // approximate factor for AWQ quantization
    default:     return 1.0;
  }
};

/**
 * Returns GB factor for KV cache usage (per 1B params), depending on quantization.
 * This is used to adjust the additional memory required when KV caching is enabled.
 */
export const getKvCacheQuantFactor = (k: KvCacheQuantization): number => {
  switch (k) {
    case 'F32': return 4.0;
    case 'F16': return 2.0;
    case 'Q8':  return 1.0;
    case 'Q5':  return 0.625;
    case 'Q4':  return 0.5;
    default:    return 1.0;
  }
};

/**
 * Calculate VRAM requirement (GB) for single-user inference,
 * distinguishing between incremental and bulk forward pass.
 *
 * Changes made:
 *  - Removed context scaling from the base model memory since model weights are fixed.
 *  - Added context-based memory estimation separately:
 *      * Incremental mode only adds KV cache overhead.
 *      * Bulk mode adds a larger overhead for storing full activations.
 *  - Included a further addition if KV cache is enabled in bulk mode.
 *  - Finally, an overhead factor of ~10% is applied for fragmentation and extra buffers.
 */
export const calculateRequiredVram = (
  params: number,              // number of model parameters in *billions*
  modelQuant: ModelQuantization,
  contextLength: number,
  useKvCache: boolean,
  kvCacheQuant: KvCacheQuantization,
  inferenceMode: InferenceMode
): number => {

  // 1) Base model memory is independent of context length.
  const modelFactor = getModelQuantFactor(modelQuant);
  const baseModelMem = params * modelFactor; // e.g. 8B params * 2.0 = 16GB for an 8B FP16 model

  // 2) Estimate additional memory needed for context processing.
  let contextMem = 0;

  if (inferenceMode === 'incremental') {
    // Start with basic context scaling for all cases
    const contextScale = Math.max(contextLength / 2048, 1);
    contextMem = baseModelMem * contextScale;

    // If KV cache is enabled, adjust the memory calculation
    if (useKvCache) {
        // alphaAt2048 represents the fraction of base model memory used by the KV cache at 2048 tokens.
        const alphaAt2048 = 0.2;
        const kvFactor = getKvCacheQuantFactor(kvCacheQuant);
        contextMem = baseModelMem * alphaAt2048 * contextScale * kvFactor;
    }
  } else {
    // Bulk forward pass:
    // In bulk mode, the entire context is processed at once, leading to a larger memory footprint.
    // We use a larger alpha value (bulkAlphaAt2048) to represent the higher overhead.
    const bulkAlphaAt2048 = 0.5;  // Rough estimate for full activation storage at 2048 tokens.
    const bulkScale = contextLength / 2048;
    contextMem = baseModelMem * bulkAlphaAt2048 * bulkScale;

    // If KV cache is enabled, add extra overhead (though in a full forward pass, it might be less effective).
    if (useKvCache) {
      const kvFactor = getKvCacheQuantFactor(kvCacheQuant);
      contextMem += baseModelMem * 0.1 * kvFactor * bulkScale;
    }
  }

  // 3) Total VRAM = base model memory + context-dependent memory.
  let totalVram = baseModelMem + contextMem;

  // 4) Add an overhead factor (~10%) to account for fragmentation and extra buffers.
  const overheadFactor = 1.1;
  totalVram *= overheadFactor;

  return totalVram;
};

/**
 * Calculate hardware recommendation based on model parameters and system configuration.
 *
 * Changes made:
 *  - The system RAM calculation now includes an estimate for extra activation memory in bulk mode.
 *  - GPU requirement is calculated with a buffer multiplier (1.2) to account for fragmentation.
 *  - The logic now differentiates between unified memory and discrete GPU setups.
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
  inferenceMode: InferenceMode
): Recommendation => {
  const requiredVram = calculateRequiredVram(
    params,
    modelQuant,
    contextLength,
    useKvCache,
    kvCacheQuant,
    inferenceMode
  );

  // Adjust system RAM calculation to include additional bulk activation memory.
  // For incremental mode, assume half of base memory is needed.
  // For bulk mode, add extra memory proportional to the context length.
  const baseSystemRamNeeded = params * getModelQuantFactor(modelQuant) * 0.5 +
                              (inferenceMode === 'bulk' ? contextLength / 1024 : 0);
  const systemRamNeeded = Math.max(8, baseSystemRamNeeded); // Ensure at least 8GB is recommended.

  // Determine if the required VRAM fits within unified memory mode.
  const fitsUnified = memoryMode === 'UNIFIED_MEMORY' && systemMemory >= requiredVram;

  // Calculate discrete GPU requirements.
  let gpusRequired = 0;
  let gpuType = '';

  if (memoryMode === 'DISCRETE_GPU') {
    // Use a buffer multiplier (1.2) to account for real-world fragmentation.
    gpusRequired = Math.ceil((requiredVram * 1.2) / gpuVram);
    if (gpusRequired === 1) {
      gpuType = `Single ${gpuVram}GB GPU`;
    } else if (gpusRequired <= 8) {
      gpuType = `${gpusRequired}x ${gpuVram}GB GPUs`;
    } else {
      // If more than 8 GPUs are needed, mark it as not feasible.
      gpuType = `Exceeds 8x ${gpuVram}GB GPUs`;
      gpusRequired = 0;
    }
  } else {
    // For unified memory mode, no discrete GPUs are needed.
    gpuType = `Unified memory (${systemMemory}GB)`;
    gpusRequired = 0;
  }

  return {
    gpuType,
    vramNeeded: requiredVram.toFixed(2),
    fitsUnified,
    systemRamNeeded,
    gpusRequired
  };
};

/**
 * Estimate on-disk model size.
 *
 * Changes made:
 *  - Instead of directly returning the product of parameters and the quantization factor,
 *    we now convert the effective factor (GB per 1B params) to bits per parameter,
 *    compute total bits, then convert to bytes and finally to gigabytes.
 */
export const calculateOnDiskSize = (
  params: number,
  modelQuant: ModelQuantization
): number => {
  const modelFactor = getModelQuantFactor(modelQuant);
  // Convert GB per 1B params to bits per parameter.
  const bitsPerParam = modelFactor * 8;
  const totalBits = params * 1e9 * bitsPerParam;
  return totalBits / 8 / 1e9;  // Convert bytes to GB
};
