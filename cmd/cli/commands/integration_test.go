//go:build integration

package commands

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/docker/model-runner/cmd/cli/desktop"
	"github.com/docker/model-runner/pkg/distribution/builder"
	"github.com/docker/model-runner/pkg/distribution/registry"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/network"
	"github.com/testcontainers/testcontainers-go/wait"
)

// testEnv holds the test environment components
type testEnv struct {
	ctx         context.Context
	registryURL string
	client      *desktop.Client
}

// setupTestEnv creates the complete test environment with registry and DMR
func setupTestEnv(t *testing.T) *testEnv {
	ctx := context.Background()

	// Create a custom network for container communication
	net, err := network.New(ctx)
	require.NoError(t, err)
	testcontainers.CleanupNetwork(t, net)

	registryURL := ociRegistry(t, ctx, net)
	dmrURL := dockerModelRunner(t, ctx, net)

	modelRunnerCtx, err := desktop.NewContextForTest(dmrURL, nil)
	require.NoError(t, err, "Failed to create model runner context")

	client := desktop.New(modelRunnerCtx)
	if !client.Status().Running {
		t.Fatal("DMR is not running")
	}

	return &testEnv{
		ctx:         ctx,
		registryURL: registryURL,
		client:      client,
	}
}

func ociRegistry(t *testing.T, ctx context.Context, net *testcontainers.DockerNetwork) string {
	t.Log("Starting OCI registry container...")
	ctr, err := testcontainers.Run(
		ctx, "registry:3",
		testcontainers.WithExposedPorts("5000/tcp"),
		testcontainers.WithWaitStrategy(wait.ForHTTP("/v2/").WithPort("5000/tcp").WithStartupTimeout(30*time.Second)),
		network.WithNetwork([]string{"registry.local"}, net),
	)
	require.NoError(t, err)
	testcontainers.CleanupContainer(t, ctr)

	registryEndpoint, err := ctr.Endpoint(ctx, "")
	require.NoError(t, err)
	registryURL := fmt.Sprintf("http://%s", registryEndpoint)
	t.Logf("Registry available at: %s", registryURL)
	return registryURL
}

func dockerModelRunner(t *testing.T, ctx context.Context, net *testcontainers.DockerNetwork) string {
	t.Log("Starting DMR container...")
	ctr, err := testcontainers.Run(
		ctx, "docker/model-runner:latest",
		testcontainers.WithExposedPorts("12434/tcp"),
		testcontainers.WithWaitStrategy(wait.ForHTTP("/engines/status").WithPort("12434/tcp").WithStartupTimeout(10*time.Second)),
		testcontainers.WithEnv(map[string]string{
			"DEFAULT_REGISTRY":  "registry.local:5000",
			"INSECURE_REGISTRY": "true",
		}),
		network.WithNetwork([]string{"dmr"}, net),
	)
	require.NoError(t, err)
	testcontainers.CleanupContainer(t, ctr)

	dmrEndpoint, err := ctr.Endpoint(ctx, "")
	require.NoError(t, err)

	dmrURL := fmt.Sprintf("http://%s", dmrEndpoint)
	t.Logf("DMR available at: %s", dmrURL)
	return dmrURL
}

// removeModel removes a model from the local store
func removeModel(client *desktop.Client, modelID string) error {
	_, err := client.Remove([]string{modelID}, true)
	return err
}

// verifyModelInspect inspects a model using the given reference and verifies it matches expectations
func verifyModelInspect(t *testing.T, client *desktop.Client, ref, expectedID, expectedDigest string) {
	t.Helper()

	model, err := client.Inspect(ref, false)
	require.NoError(t, err, "Failed to inspect model with reference: %s", ref)

	// Verify model ID matches
	require.Equal(t, expectedID, model.ID,
		"Model ID mismatch when inspecting with reference: %s. Expected: %s, Got: %s",
		ref, expectedID, model.ID)

	// Verify digest matches if provided
	if expectedDigest != "" {
		require.Equal(t, expectedDigest, model.ID,
			"Model digest mismatch when inspecting with reference: %s. Expected: %s, Got: %s",
			ref, expectedDigest, model.ID)
	}

	// Verify model has tags
	require.NotEmpty(t, model.Tags, "Model should have at least one tag")

	t.Logf("✓ Successfully inspected model with reference: %s (ID: %s, Tags: %v)",
		ref, model.ID[7:19], model.Tags)
}

// createAndPushTestModel creates a minimal test model and pushes it to the local registry.
// Returns the model ID, FQDNs for host and network access, and the manifest digest.
func createAndPushTestModel(t *testing.T, registryURL, modelRef string, contextSize uint64) (modelID, hostFQDN, networkFQDN, digest string) {
	ctx := context.Background()

	// Use the dummy GGUF file from assets
	dummyGGUFPath := filepath.Join("../../../assets/dummy.gguf")
	absPath, err := filepath.Abs(dummyGGUFPath)
	require.NoError(t, err)

	// Check if the file exists
	_, err = os.Stat(absPath)
	require.NoError(t, err, "dummy.gguf not found at %s", absPath)

	// Create a builder from the GGUF file
	t.Logf("Creating test model %s from %s", modelRef, absPath)
	pkg, err := builder.FromGGUF(absPath)
	require.NoError(t, err)

	// Set context size if specified
	if contextSize > 0 {
		pkg = pkg.WithContextSize(contextSize)
	}

	// Construct the full reference with the local registry host for pushing from test host
	uri, err := url.Parse(registryURL)
	require.NoError(t, err)

	hostFQDN = fmt.Sprintf("%s/%s", uri.Host, modelRef)
	t.Logf("Pushing to local registry: %s", hostFQDN)

	// Create registry client
	client := registry.NewClient(registry.WithUserAgent("integration-test/1.0"))
	target, err := client.NewTarget(hostFQDN)
	require.NoError(t, err)

	// Push the model
	err = pkg.Build(ctx, target, io.Discard)
	require.NoError(t, err)
	t.Logf("Successfully pushed test model: %s", hostFQDN)

	// For pulling from DMR, use the network alias "registry.local:5000"
	// go-containerregistry will automatically use HTTP for .local hostnames
	networkFQDN = fmt.Sprintf("registry.local:5000/%s", modelRef)

	id, err := pkg.Model().ID()
	require.NoError(t, err)
	t.Logf("Model ID: %s", id)

	// Get the manifest digest
	manifestDigest, err := pkg.Model().Digest()
	require.NoError(t, err)
	digest = manifestDigest.String()
	t.Logf("Model digest: %s", digest)

	return id, hostFQDN, networkFQDN, digest
}

// TestIntegration_PullModel tests pulling a model from the local OCI registry via DMR
// with various model reference formats to ensure proper normalization.
func TestIntegration_PullModel(t *testing.T) {
	env := setupTestEnv(t)

	models, err := listModels(false, env.client, true, false, "")
	require.NoError(t, err)

	if len(models) != 0 {
		t.Fatal("Expected no initial models, but found some")
	}

	// Create and push two test models with different organizations
	// Model 1: custom org (test/test-model:latest)
	modelRef1 := "test/test-model:latest"
	modelID1, hostFQDN1, networkFQDN1, digest1 := createAndPushTestModel(t, env.registryURL, modelRef1, 2048)
	t.Logf("Test model 1 pushed: %s (ID: %s) FQDN: %s Digest: %s", hostFQDN1, modelID1, networkFQDN1, digest1)

	// Model 2: default org (ai/test-model:latest)
	modelRef2 := "ai/test-model:latest"
	modelID2, hostFQDN2, networkFQDN2, digest2 := createAndPushTestModel(t, env.registryURL, modelRef2, 2048)
	t.Logf("Test model 2 pushed: %s (ID: %s) FQDN: %s Digest: %s", hostFQDN2, modelID2, networkFQDN2, digest2)

	// Test cases for different model reference formats
	testCases := []struct {
		name              string
		pullRef           string // Reference to use when pulling
		expectedModelID   string // Expected model ID after pull
		expectedModelName string // Expected model name for logging
	}{
		{
			name:              "explicit custom org and tag",
			pullRef:           "registry.local:5000/test/test-model:latest",
			expectedModelID:   modelID1,
			expectedModelName: "test/test-model:latest",
		},
		{
			name:              "custom org without tag (should default to :latest)",
			pullRef:           "registry.local:5000/test/test-model",
			expectedModelID:   modelID1,
			expectedModelName: "test/test-model:latest",
		},
		{
			name:              "explicit default org and tag",
			pullRef:           "registry.local:5000/ai/test-model:latest",
			expectedModelID:   modelID2,
			expectedModelName: "ai/test-model:latest",
		},
		{
			name:              "default org without tag (should default to :latest)",
			pullRef:           "registry.local:5000/ai/test-model",
			expectedModelID:   modelID2,
			expectedModelName: "ai/test-model:latest",
		},
		{
			name:              "no org with tag (should default to ai/)",
			pullRef:           "test-model:latest",
			expectedModelID:   modelID2,
			expectedModelName: "ai/test-model:latest",
		},
		{
			name:              "no org and no tag (should default to ai/:latest)",
			pullRef:           "test-model",
			expectedModelID:   modelID2,
			expectedModelName: "ai/test-model:latest",
		},
		{
			name:              "pull by digest with full registry path",
			pullRef:           fmt.Sprintf("registry.local:5000/test/test-model@%s", digest1),
			expectedModelID:   modelID1,
			expectedModelName: "test/test-model",
		},
		{
			name:              "pull by digest with default registry",
			pullRef:           fmt.Sprintf("ai/test-model@%s", digest2),
			expectedModelID:   modelID2,
			expectedModelName: "ai/test-model",
		},

		{
			name:              "pull by digest with default registry and default org",
			pullRef:           fmt.Sprintf("test-model@%s", digest2),
			expectedModelID:   modelID2,
			expectedModelName: "ai/test-model",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Pull the model using the test case reference
			t.Logf("Pulling model with reference: %s", tc.pullRef)
			err := pullModel(newPullCmd(), env.client, tc.pullRef, true)
			require.NoError(t, err, "Failed to pull model with reference: %s", tc.pullRef)

			// List models and verify the expected model is present
			models, err := listModels(false, env.client, true, false, "")
			require.NoError(t, err)

			if len(models) == 0 {
				t.Fatalf("No models found after pulling %s", tc.pullRef)
			}

			models = strings.TrimSpace(models)

			// Extract truncated ID format (sha256:xxx... -> xxx where xxx is 12 chars)
			// listModels with quiet=true returns modelID[7:19]
			truncatedID := tc.expectedModelID[7:19]
			if models != truncatedID {
				t.Errorf("Expected model ID %s (truncated: %s) not found in model list after pulling %s.\nExpected model: %s\nModel list:\n%s",
					tc.expectedModelID, truncatedID, tc.pullRef, tc.expectedModelName, models)
			} else {
				t.Logf("✓ Successfully verified model %s (ID: %s) after pulling with reference: %s",
					tc.expectedModelName, truncatedID, tc.pullRef)
			}

			// Verify inspect works with the same reference used for pulling
			// Determine expected digest based on which model was pulled
			expectedDigest := ""
			if tc.expectedModelID == modelID1 {
				expectedDigest = digest1
			} else if tc.expectedModelID == modelID2 {
				expectedDigest = digest2
			}
			verifyModelInspect(t, env.client, tc.pullRef, tc.expectedModelID, expectedDigest)

			// Clean up: remove the model for the next test iteration
			// Note: We use the full model ID for removal to ensure we remove the correct model
			t.Logf("Removing model %s", truncatedID)
			err = removeModel(env.client, tc.expectedModelID)
			require.NoError(t, err, "Failed to remove model")
		})
	}
}

// TestIntegration_InspectModel tests inspecting a model with various reference formats
// to ensure proper reference normalization and consistent output.
func TestIntegration_InspectModel(t *testing.T) {
	env := setupTestEnv(t)

	// Ensure no models exist initially
	models, err := listModels(false, env.client, true, false, "")
	require.NoError(t, err)
	if len(models) != 0 {
		t.Fatal("Expected no initial models, but found some")
	}

	// Create and push a test model with default org (ai/inspect-test:latest)
	modelRef := "ai/inspect-test:latest"
	modelID, hostFQDN, networkFQDN, digest := createAndPushTestModel(t, env.registryURL, modelRef, 2048)
	t.Logf("Test model pushed: %s (ID: %s) FQDN: %s Digest: %s", hostFQDN, modelID, networkFQDN, digest)

	// Pull the model using a short reference
	pullRef := "inspect-test"
	t.Logf("Pulling model with reference: %s", pullRef)
	err = pullModel(newPullCmd(), env.client, pullRef, true)
	require.NoError(t, err, "Failed to pull model")

	// Verify the model was pulled
	models, err = listModels(false, env.client, true, false, "")
	require.NoError(t, err)
	truncatedID := modelID[7:19]
	require.Equal(t, truncatedID, strings.TrimSpace(models), "Model not found after pull")

	// Test cases: different ways to reference the same model
	testCases := []struct {
		name string
		ref  string
	}{
		{
			name: "short form (no org, no tag)",
			ref:  "inspect-test",
		},
		{
			name: "with tag (no org)",
			ref:  "inspect-test:latest",
		},
		{
			name: "with org (no tag)",
			ref:  "ai/inspect-test",
		},
		{
			name: "fully qualified (org and tag)",
			ref:  "ai/inspect-test:latest",
		},
		{
			name: "with registry (fully qualified)",
			ref:  "registry.local:5000/ai/inspect-test:latest",
		},
		{
			name: "with registry (no tag)",
			ref:  "registry.local:5000/ai/inspect-test",
		},
		{
			name: "full model ID",
			ref:  modelID,
		},
		{
			name: "truncated model ID (12 chars)",
			ref:  truncatedID,
		},
		{
			name: "model ID without sha256 prefix",
			ref:  strings.TrimPrefix(modelID, "sha256:"),
		},
	}

	// Verify inspect works with all reference formats
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			verifyModelInspect(t, env.client, tc.ref, modelID, digest)
		})
	}

	// Cleanup: remove the model
	t.Logf("Removing model %s", truncatedID)
	err = removeModel(env.client, modelID)
	require.NoError(t, err, "Failed to remove model")

	// Verify model was removed
	models, err = listModels(false, env.client, true, false, "")
	require.NoError(t, err)
	require.Empty(t, strings.TrimSpace(models), "Model should be removed")
}
