package bundle

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/docker/model-runner/pkg/distribution/modelpack"
	"github.com/docker/model-runner/pkg/distribution/oci"
	"github.com/docker/model-runner/pkg/distribution/types"
)

func TestValidatePathWithinDirectory(t *testing.T) {
	// Create a temporary directory for testing
	baseDir := t.TempDir()

	tests := []struct {
		name        string
		targetPath  string
		expectError bool
		description string
	}{
		// Valid paths - should pass
		{
			name:        "simple filename",
			targetPath:  "model.safetensors",
			expectError: false,
			description: "Simple filename should be valid",
		},
		{
			name:        "nested directory",
			targetPath:  "text_encoder/model.safetensors",
			expectError: false,
			description: "Nested path should be valid",
		},
		{
			name:        "deeply nested",
			targetPath:  "a/b/c/d/model.safetensors",
			expectError: false,
			description: "Deeply nested path should be valid",
		},

		// Directory traversal attacks - should fail
		{
			name:        "parent directory escape",
			targetPath:  "../etc/passwd",
			expectError: true,
			description: "Parent directory escape should be blocked",
		},
		{
			name:        "multiple parent escape",
			targetPath:  "../../../etc/passwd",
			expectError: true,
			description: "Multiple parent directory escape should be blocked",
		},
		{
			name:        "mixed path with escape",
			targetPath:  "text_encoder/../../../etc/passwd",
			expectError: true,
			description: "Path that starts valid but escapes should be blocked",
		},
		{
			name:        "absolute path unix",
			targetPath:  "/etc/passwd",
			expectError: true,
			description: "Absolute Unix path should be blocked",
		},
		// Note: Windows absolute path test is platform-specific.
		// On Unix, "C:\..." is treated as a relative path (it doesn't start with /),
		// so it would create a file/directory with that name, which is allowed.
		// On Windows, filepath.IsAbs() correctly identifies "C:\" as absolute.

		// Edge cases
		{
			name:        "empty path",
			targetPath:  "",
			expectError: true,
			description: "Empty path should be blocked (filepath.IsLocal returns false for empty)",
		},
		{
			name:        "dot only",
			targetPath:  ".",
			expectError: true,
			description: "Dot path should be blocked",
		},
		{
			name:        "double dot only",
			targetPath:  "..",
			expectError: true,
			description: "Double dot path should be blocked",
		},
		{
			name:        "path with null byte",
			targetPath:  "model\x00.safetensors",
			expectError: true,
			description: "Path with null byte should be blocked (invalid in most filesystems)",
		},

		// Tricky paths that might bypass naive checks
		{
			name:        ".. in middle",
			targetPath:  "foo/../bar/model.safetensors",
			expectError: false,
			description: "Path with .. that stays within directory should be valid",
		},
		{
			name:        "trailing slash",
			targetPath:  "text_encoder/",
			expectError: false,
			description: "Directory path with trailing slash should be valid",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validatePathWithinDirectory(baseDir, tt.targetPath)

			if tt.expectError && err == nil {
				t.Errorf("Expected error for path %q (%s), but got nil", tt.targetPath, tt.description)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error for path %q (%s), but got: %v", tt.targetPath, tt.description, err)
			}
		})
	}
}

func TestUpdateBundleFieldsFromLayer_CNCFMediaTypes(t *testing.T) {
	tests := []struct {
		name              string
		mediaType         oci.MediaType
		relPath           string
		modelFormat       string
		expectGGUF        string
		expectSafetensors string
	}{
		{
			name:              "Docker safetensors media type",
			mediaType:         types.MediaTypeSafetensors,
			relPath:           "model/model.safetensors",
			modelFormat:       "",
			expectSafetensors: "model/model.safetensors",
		},
		{
			name:              "CNCF format-specific safetensors media type",
			mediaType:         oci.MediaType(modelpack.MediaTypeWeightSafetensors),
			relPath:           "model/model.safetensors",
			modelFormat:       "",
			expectSafetensors: "model/model.safetensors",
		},
		{
			name:        "CNCF format-specific GGUF media type",
			mediaType:   oci.MediaType(modelpack.MediaTypeWeightGGUF),
			relPath:     "model/model.gguf",
			modelFormat: "",
			expectGGUF:  "model/model.gguf",
		},
		{
			name:              "CNCF generic weight raw with safetensors format",
			mediaType:         oci.MediaType(modelpack.MediaTypeWeightRaw),
			relPath:           "model/model.safetensors",
			modelFormat:       string(types.FormatSafetensors),
			expectSafetensors: "model/model.safetensors",
		},
		{
			name:        "CNCF generic weight raw with GGUF format",
			mediaType:   oci.MediaType(modelpack.MediaTypeWeightRaw),
			relPath:     "model/model.gguf",
			modelFormat: string(types.FormatGGUF),
			expectGGUF:  "model/model.gguf",
		},
		{
			name:              "CNCF generic weight raw without format does nothing",
			mediaType:         oci.MediaType(modelpack.MediaTypeWeightRaw),
			relPath:           "model/model.safetensors",
			modelFormat:       "",
			expectSafetensors: "",
			expectGGUF:        "",
		},
		{
			name:              "unknown media type does nothing",
			mediaType:         "application/vnd.cncf.model.weight.config.v1.raw",
			relPath:           "model/config.json",
			modelFormat:       string(types.FormatSafetensors),
			expectSafetensors: "",
			expectGGUF:        "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bundle := &Bundle{}
			updateBundleFieldsFromLayer(bundle, tt.mediaType, tt.relPath, tt.modelFormat)

			if bundle.safetensorsFile != tt.expectSafetensors {
				t.Errorf("safetensorsFile = %q, want %q", bundle.safetensorsFile, tt.expectSafetensors)
			}
			if bundle.ggufFile != tt.expectGGUF {
				t.Errorf("ggufFile = %q, want %q", bundle.ggufFile, tt.expectGGUF)
			}
		})
	}
}

func TestIsCNCFModel(t *testing.T) {
	tests := []struct {
		name            string
		configMediaType oci.MediaType
		expected        bool
	}{
		{
			name:            "CNCF ModelPack config V1",
			configMediaType: modelpack.MediaTypeModelConfigV1,
			expected:        true,
		},
		{
			name:            "Docker V0.1 config",
			configMediaType: types.MediaTypeModelConfigV01,
			expected:        false,
		},
		{
			name:            "Docker V0.2 config",
			configMediaType: types.MediaTypeModelConfigV02,
			expected:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a minimal artifact with the given config media type
			artifact := &testArtifactWithConfigMediaType{
				configMediaType: tt.configMediaType,
			}
			result := isCNCFModel(artifact)
			if result != tt.expected {
				t.Errorf("isCNCFModel() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// testArtifactWithConfigMediaType is a minimal ModelArtifact for testing isCNCFModel/isV02Model.
type testArtifactWithConfigMediaType struct {
	configMediaType oci.MediaType
}

func (a *testArtifactWithConfigMediaType) Manifest() (*oci.Manifest, error) {
	return &oci.Manifest{
		Config: oci.Descriptor{
			MediaType: a.configMediaType,
		},
	}, nil
}

// Stubs to satisfy types.ModelArtifact interface (not used in these tests).
func (a *testArtifactWithConfigMediaType) ID() (string, error) { return "", nil }
func (a *testArtifactWithConfigMediaType) Config() (types.ModelConfig, error) {
	return nil, nil
}
func (a *testArtifactWithConfigMediaType) Tags() []string { return nil }
func (a *testArtifactWithConfigMediaType) Descriptor() (types.Descriptor, error) {
	return types.Descriptor{}, nil
}
func (a *testArtifactWithConfigMediaType) GGUFPaths() ([]string, error) { return nil, nil }
func (a *testArtifactWithConfigMediaType) SafetensorsPaths() ([]string, error) {
	return nil, nil
}
func (a *testArtifactWithConfigMediaType) Layers() ([]oci.Layer, error)         { return nil, nil }
func (a *testArtifactWithConfigMediaType) RawConfigFile() ([]byte, error)       { return nil, nil }
func (a *testArtifactWithConfigMediaType) RawManifest() ([]byte, error)         { return nil, nil }
func (a *testArtifactWithConfigMediaType) MediaType() (oci.MediaType, error)    { return "", nil }
func (a *testArtifactWithConfigMediaType) Size() (int64, error)                 { return 0, nil }
func (a *testArtifactWithConfigMediaType) ConfigName() (oci.Hash, error)        { return oci.Hash{}, nil }
func (a *testArtifactWithConfigMediaType) ConfigFile() (*oci.ConfigFile, error) { return nil, nil }
func (a *testArtifactWithConfigMediaType) Digest() (oci.Hash, error)            { return oci.Hash{}, nil }
func (a *testArtifactWithConfigMediaType) LayerByDigest(oci.Hash) (oci.Layer, error) {
	return nil, nil
}
func (a *testArtifactWithConfigMediaType) LayerByDiffID(oci.Hash) (oci.Layer, error) {
	return nil, nil
}

func TestValidatePathWithinDirectory_RealFilesystem(t *testing.T) {
	// Create a temporary directory structure
	baseDir := t.TempDir()

	// Create a sibling directory that attacker might try to access
	siblingDir := filepath.Join(filepath.Dir(baseDir), "sibling-secret")
	if err := os.MkdirAll(siblingDir, 0755); err != nil {
		t.Fatalf("Failed to create sibling dir: %v", err)
	}
	defer os.RemoveAll(siblingDir)

	// Create a secret file in the sibling directory
	secretFile := filepath.Join(siblingDir, "secret.txt")
	if err := os.WriteFile(secretFile, []byte("secret data"), 0644); err != nil {
		t.Fatalf("Failed to create secret file: %v", err)
	}

	// Try to escape to the sibling directory
	escapePath := "../sibling-secret/secret.txt"
	err := validatePathWithinDirectory(baseDir, escapePath)
	if err == nil {
		t.Errorf("Expected error when attempting to escape to sibling directory, but validation passed")
	}
}
