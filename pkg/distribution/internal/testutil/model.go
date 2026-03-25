package testutil

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/docker/model-runner/pkg/distribution/builder"
	"github.com/docker/model-runner/pkg/distribution/internal/partial"
	"github.com/docker/model-runner/pkg/distribution/oci"
	"github.com/docker/model-runner/pkg/distribution/types"
)

// BuildModelFromPath constructs a model artifact from a file path and fails the test on error.
func BuildModelFromPath(t *testing.T, path string) types.ModelArtifact {
	t.Helper()

	b, err := builder.FromPath(path)
	if err != nil {
		t.Fatalf("Failed to create model from path %q: %v", path, err)
	}
	return b.Model()
}

// Artifact is a generic, configurable OCI model artifact for testing.
// It accepts raw config bytes, a config media type, and an arbitrary set of layers,
// allowing tests to construct Docker model-spec, ModelPack, or any custom OCI artifact
// without re-implementing the oci.Image interface in every test file.
type Artifact struct {
	rawConfig       []byte
	configMediaType oci.MediaType
	layers          []oci.Layer
}

// NewArtifact creates a new generic test artifact with the given raw config bytes,
// config media type, and layers.
func NewArtifact(rawConfig []byte, configMediaType oci.MediaType, layers ...oci.Layer) *Artifact {
	return &Artifact{
		rawConfig:       rawConfig,
		configMediaType: configMediaType,
		layers:          layers,
	}
}

// GetConfigMediaType implements partial.WithConfigMediaType so that ManifestForLayers
// uses the correct config media type when building the OCI manifest.
func (a *Artifact) GetConfigMediaType() oci.MediaType {
	return a.configMediaType
}

// RawConfigFile implements partial.WithRawConfigFile.
func (a *Artifact) RawConfigFile() ([]byte, error) {
	return a.rawConfig, nil
}

// Layers implements oci.Image.
func (a *Artifact) Layers() ([]oci.Layer, error) {
	return a.layers, nil
}

// MediaType implements oci.Image.
func (a *Artifact) MediaType() (oci.MediaType, error) {
	m, err := a.Manifest()
	if err != nil {
		return "", err
	}
	return m.MediaType, nil
}

// Size implements oci.Image.
func (a *Artifact) Size() (int64, error) {
	rawManifest, err := a.RawManifest()
	if err != nil {
		return 0, err
	}
	size := int64(len(rawManifest) + len(a.rawConfig))
	for _, l := range a.layers {
		ls, err := l.Size()
		if err != nil {
			return 0, err
		}
		size += ls
	}
	return size, nil
}

// ConfigName implements oci.Image.
func (a *Artifact) ConfigName() (oci.Hash, error) {
	hash, _, err := oci.SHA256(bytes.NewReader(a.rawConfig))
	return hash, err
}

// ConfigFile implements oci.Image. Model artifacts do not have a standard OCI config file.
func (a *Artifact) ConfigFile() (*oci.ConfigFile, error) {
	return nil, errors.New("not supported for model artifacts")
}

// Digest implements oci.Image.
func (a *Artifact) Digest() (oci.Hash, error) {
	raw, err := a.RawManifest()
	if err != nil {
		return oci.Hash{}, err
	}
	hash, _, err := oci.SHA256(bytes.NewReader(raw))
	return hash, err
}

// Manifest implements oci.Image.
func (a *Artifact) Manifest() (*oci.Manifest, error) {
	return partial.ManifestForLayers(a)
}

// RawManifest implements oci.Image.
func (a *Artifact) RawManifest() ([]byte, error) {
	m, err := a.Manifest()
	if err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

// LayerByDigest implements oci.Image.
func (a *Artifact) LayerByDigest(hash oci.Hash) (oci.Layer, error) {
	for _, l := range a.layers {
		d, err := l.Digest()
		if err != nil {
			return nil, err
		}
		if d == hash {
			return l, nil
		}
	}
	return nil, fmt.Errorf("layer with digest %s not found", hash)
}

// LayerByDiffID implements oci.Image.
func (a *Artifact) LayerByDiffID(hash oci.Hash) (oci.Layer, error) {
	for _, l := range a.layers {
		d, err := l.DiffID()
		if err != nil {
			return nil, err
		}
		if d == hash {
			return l, nil
		}
	}
	return nil, fmt.Errorf("layer with diffID %s not found", hash)
}
