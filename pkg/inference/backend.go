package inference

import (
	"context"
	"net/http"
)

// BackendMode encodes the mode in which a backend should operate.
type BackendMode uint8

const (
	// BackendModeCompletion indicates that the backend should run in chat
	// completion mode.
	BackendModeCompletion BackendMode = iota
	// BackendModeEmbedding indicates that the backend should run in embedding
	// mode.
	BackendModeEmbedding
)

type ErrGGUFParse struct {
	Err error
}

func (e *ErrGGUFParse) Error() string {
	return "failed to parse GGUF: " + e.Err.Error()
}

// String implements Stringer.String for BackendMode.
func (m BackendMode) String() string {
	switch m {
	case BackendModeCompletion:
		return "completion"
	case BackendModeEmbedding:
		return "embedding"
	default:
		return "unknown"
	}
}

type SpeculativeDecodingConfig struct {
	DraftModel        string  `json:"draft_model,omitempty"`
	NumTokens         int     `json:"num_tokens,omitempty"`
	MinAcceptanceRate float64 `json:"min_acceptance_rate,omitempty"`
}

type BackendConfiguration struct {
	ContextSize  int64                       `json:"context-size,omitempty"`
	RuntimeFlags []string                    `json:"runtime-flags,omitempty"`
	Speculative  *SpeculativeDecodingConfig `json:"speculative,omitempty"`
}

type RequiredMemory struct {
	RAM  uint64
	VRAM uint64 // TODO(p1-0tr): for now assume we are working with single GPU set-ups
}

// Backend is the interface implemented by inference engine backends. Backend
// implementations need not be safe for concurrent invocation of the following
// methods, though their underlying server implementations do need to support
// concurrent API requests.
type Backend interface {
	// Name returns the backend name. It must be all lowercase and usable as a
	// path component in an HTTP request path and a Unix domain socket path. It
	// should also be suitable for presenting to users (at least in logs). The
	// package providing the backend implementation should also expose a
	// constant called Name which matches the value returned by this method.
	Name() string
	// UsesExternalModelManagement should return true if the backend uses an
	// external model management system and false if the backend uses the shared
	// model manager.
	UsesExternalModelManagement() bool
	// Install ensures that the backend is installed. It should return a nil
	// error if installation succeeds or if the backend is already installed.
	// The provided HTTP client should be used for any HTTP operations.
	Install(ctx context.Context, httpClient *http.Client) error
	// Run runs an OpenAI API web server on the specified Unix domain socket
	// socket for the specified model using the backend. It should start any
	// process(es) necessary for the backend to function for the model. It
	// should not return until either the process(es) fail or the provided
	// context is cancelled. By the time Run returns, any process(es) it has
	// spawned must terminate.
	//
	// Backend implementations should be "one-shot" (i.e. returning from Run
	// after the failure of an underlying process). Backends should not attempt
	// to perform restarts on failure. Backends should only return a nil error
	// in the case of context cancellation, otherwise they should return the
	// error that caused them to fail.
	//
	// Run will be provided with the path to a Unix domain socket on which the
	// backend should listen for incoming OpenAI API requests and a model name
	// to be loaded. Backends should not load multiple models at once and should
	// instead load only the specified model. Backends should still respond to
	// OpenAI API requests for other models with a 421 error code.
	Run(ctx context.Context, socket, model string, modelRef string, mode BackendMode, config *BackendConfiguration) error
	// Status returns a description of the backend's state.
	Status() string
	// GetDiskUsage returns the disk usage of the backend.
	GetDiskUsage() (int64, error)
	// GetRequiredMemoryForModel returns the required working memory for a given
	// model.
	GetRequiredMemoryForModel(ctx context.Context, model string, config *BackendConfiguration) (RequiredMemory, error)
}
