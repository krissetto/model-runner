//go:build integration

package commands

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
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

// modelInfo contains all the information needed to generate model references
type modelInfo struct {
	name         string // e.g., "test-model"
	org          string // e.g., "ai"
	tag          string // e.g., "latest"
	registry     string // e.g., "registry.local:5000"
	modelID      string // Full ID: sha256:...
	digest       string // sha256:...
	expectedName string // What we expect to see: "ai/test-model:latest"
}

// referenceTestCase represents a test case for a specific reference format
type referenceTestCase struct {
	name string
	ref  string
}

// generateReferenceTestCases generates a comprehensive list of reference formats for testing
func generateReferenceTestCases(info modelInfo) []referenceTestCase {
	cases := []referenceTestCase{
		{
			name: "short form (no registry, no org, no tag)",
			ref:  info.name,
		},
		{
			name: "with tag",
			ref:  fmt.Sprintf("%s:%s", info.name, info.tag),
		},
		{
			name: "with org",
			ref:  fmt.Sprintf("%s/%s", info.org, info.name),
		},
		{
			name: "with org and tag",
			ref:  fmt.Sprintf("%s/%s:%s", info.org, info.name, info.tag),
		},
		{
			name: "fqdn",
			ref:  fmt.Sprintf("%s/%s/%s:%s", info.registry, info.org, info.name, info.tag),
		},
		{
			name: "with registry and org",
			ref:  fmt.Sprintf("%s/%s/%s", info.registry, info.org, info.name),
		},
		{
			name: "by digest",
			ref:  fmt.Sprintf("%s@%s", info.name, info.digest),
		},
		{
			name: "by digest with org",
			ref:  fmt.Sprintf("%s/%s@%s", info.org, info.name, info.digest),
		},
		{
			name: "by digest with registry and org",
			ref:  fmt.Sprintf("%s/%s/%s@%s", info.registry, info.org, info.name, info.digest),
		},
		{
			name: "full model ID",
			ref:  info.modelID,
		},
		{
			name: "truncated model ID (12 chars)",
			ref:  info.modelID[7:19],
		},
		{
			name: "model ID without sha256 prefix",
			ref:  strings.TrimPrefix(info.modelID, "sha256:"),
		},
	}

	return cases
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
	if os.Getenv("BUILD_DMR") == "1" {
		t.Log("Building DMR container...")
		out, err := exec.CommandContext(ctx, "make", "-C", "../../..", "docker-build").CombinedOutput()
		if err != nil {
			t.Fatalf("Failed to build DMR container: %v\n%s", err, out)
		}
	}
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

	// Verify digest matches
	require.Equal(t, expectedDigest, model.ID,
		"Model digest mismatch when inspecting with reference: %s. Expected: %s, Got: %s",
		ref, expectedDigest, model.ID)

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

	// Generate test cases for custom org model (test/test-model)
	customOrgInfo := modelInfo{
		name:         "test-model",
		org:          "test",
		tag:          "latest",
		registry:     "registry.local:5000",
		modelID:      modelID1,
		digest:       digest1,
		expectedName: "test/test-model:latest",
	}
	customOrgCases := generateReferenceTestCases(customOrgInfo)

	// Model 2: default org (ai/test-model:latest)
	modelRef2 := "ai/test-model:latest"
	modelID2, hostFQDN2, networkFQDN2, digest2 := createAndPushTestModel(t, env.registryURL, modelRef2, 2048)
	t.Logf("Test model 2 pushed: %s (ID: %s) FQDN: %s Digest: %s", hostFQDN2, modelID2, networkFQDN2, digest2)

	// Generate test cases for default org model (ai/test-model)
	defaultOrgInfo := modelInfo{
		name:         "test-model",
		org:          "ai",
		tag:          "latest",
		registry:     "registry.local:5000",
		modelID:      modelID2,
		digest:       digest2,
		expectedName: "ai/test-model:latest",
	}
	defaultOrgCases := generateReferenceTestCases(defaultOrgInfo)

	// Combine test cases with expected model IDs
	// References without explicit org should resolve to ai/ (default org)
	// References with explicit "test" org should resolve to test/test-model
	type pullTestCase struct {
		referenceTestCase
		expectedModelID   string
		expectedModelName string
		expectedDigest    string
	}

	var testCases []pullTestCase

	// Add custom org cases (with explicit "test" org in reference)
	// Only include cases where the reference explicitly contains the "test" org
	// Cases without explicit org will normalize to "ai/" (default org)
	for _, tc := range customOrgCases {
		// Skip ID-based references for pull tests (can't pull by ID)
		if strings.Contains(tc.name, "model ID") {
			continue
		}
		// Skip cases that don't have explicit org in the reference
		// These will normalize to the default org (ai/) and should only be in defaultOrgCases
		if !strings.Contains(tc.ref, "test/") && !strings.Contains(tc.ref, "registry.local:5000/test/") {
			continue
		}
		testCases = append(testCases, pullTestCase{
			referenceTestCase: tc,
			expectedModelID:   modelID1,
			expectedModelName: "test/test-model:latest",
			expectedDigest:    digest1,
		})
	}

	// Add default org cases (references that should default to ai/)
	for _, tc := range defaultOrgCases {
		// Skip ID-based references for pull tests (can't pull by ID)
		if strings.Contains(tc.name, "model ID") {
			continue
		}
		testCases = append(testCases, pullTestCase{
			referenceTestCase: tc,
			expectedModelID:   modelID2,
			expectedModelName: "ai/test-model:latest",
			expectedDigest:    digest2,
		})
	}

	// Run all test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Pull the model using the test case reference
			t.Logf("Pulling model with reference: %s", tc.ref)
			err := pullModel(newPullCmd(), env.client, tc.ref, true)
			require.NoError(t, err, "Failed to pull model with reference: %s", tc.ref)

			// List models and verify the expected model is present
			models, err := listModels(false, env.client, true, false, "")
			require.NoError(t, err)

			if len(models) == 0 {
				t.Fatalf("No models found after pulling %s", tc.ref)
			}

			models = strings.TrimSpace(models)

			// Extract truncated ID format (sha256:xxx... -> xxx where xxx is 12 chars)
			// listModels with quiet=true returns modelID[7:19]
			truncatedID := tc.expectedModelID[7:19]
			if models != truncatedID {
				t.Errorf("Expected model ID %s (truncated: %s) not found in model list after pulling %s.\nExpected model: %s\nModel list:\n%s",
					tc.expectedModelID, truncatedID, tc.ref, tc.expectedModelName, models)
			} else {
				t.Logf("✓ Successfully verified model %s (ID: %s) after pulling with reference: %s",
					tc.expectedModelName, truncatedID, tc.ref)
			}

			// Verify inspect works with the same reference used for pulling
			verifyModelInspect(t, env.client, tc.ref, tc.expectedModelID, tc.expectedDigest)

			// Clean up: remove the model for the next test iteration
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

	// Generate all reference test cases using the unified system
	info := modelInfo{
		name:         "inspect-test",
		org:          "ai",
		tag:          "latest",
		registry:     "registry.local:5000",
		modelID:      modelID,
		digest:       digest,
		expectedName: "ai/inspect-test:latest",
	}
	testCases := generateReferenceTestCases(info)

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
