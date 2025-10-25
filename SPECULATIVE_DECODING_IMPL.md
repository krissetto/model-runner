# Speculative Decoding Implementation Plan

## Goal
Add speculative decoding support for both llama.cpp and vLLM backends with unified API.

## Data Structures

### BackendConfiguration Extension
```go
type SpeculativeDecodingConfig struct {
    DraftModel        string  `json:"draft_model"`
    NumTokens         int     `json:"num_tokens,omitempty"`
    MinAcceptanceRate float64 `json:"min_acceptance_rate,omitempty"`
}
```

### Runner Key Update
Change from `{backend, model, mode}` to `{backend, model, draftModel, mode}`

## Implementation Tasks

### Core Changes
- [ ] Update `pkg/inference/backend.go` - Add SpeculativeDecodingConfig
- [ ] Update `pkg/inference/scheduling/loader.go` - Extend runnerKey with draftModel
- [ ] Update `pkg/inference/scheduling/api.go` - Add speculative config to ConfigureRequest

### Backend Implementations
- [ ] Update `pkg/inference/backends/llamacpp/llamacpp.go` - Memory estimation for draft model
- [ ] Update `pkg/inference/backends/llamacpp/llamacpp_config.go` - CLI args generation
- [ ] Update `pkg/inference/backends/vllm/vllm.go` - Memory estimation for draft model
- [ ] Update `pkg/inference/backends/vllm/vllm_config.go` - CLI args generation

### Validation
- [ ] Verify draft model exists before starting runner
- [ ] Memory reservation for both models
- [ ] Handle draft model unavailability gracefully

## Backend-Specific Flags

### llama.cpp
- `--draft <path>` - draft model path
- `--draft-n <int>` - num tokens
- `--draft-p-min <float>` - min acceptance

### vLLM
- `--speculative-model <path>` - draft model path
- `--num-speculative-tokens <int>` - num tokens
- `--use-v2-block-manager` - required flag

## Notes
- Draft model becomes part of runner identity
- Both models managed through same distribution system
- Empty draft model = no speculative decoding (backward compatible)