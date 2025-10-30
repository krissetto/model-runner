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
		ctx, "registry:2",
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

// createAndPushTestModel creates a minimal test model and pushes it to the local registry.
// Returns the model ID and FQDNs for host and network access.
func createAndPushTestModel(t *testing.T, registryURL, modelRef string, contextSize uint64) (modelID, hostFQDN, networkFQDN string) {
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

	return id, hostFQDN, networkFQDN
}

// TestIntegration_PullModel tests pulling a model from the local OCI registry via DMR
func TestIntegration_PullModel(t *testing.T) {
	env := setupTestEnv(t)

	models, err := listModels(false, env.client, true, false, "")
	require.NoError(t, err)

	if len(models) != 0 {
		t.Fatal("No models found after pull")
	}

	// Create and push a test model
	modelRef := "test/test-model:latest"
	modelID, hostFQDN, networkFQDN := createAndPushTestModel(t, env.registryURL, modelRef, 2048)
	t.Logf("Test model pushed: %s", hostFQDN)

	// Pull the model using the network alias (for inter-container communication)
	// go-containerregistry automatically uses HTTP for single-word hostnames without dots
	t.Logf("Pulling model %s", networkFQDN)
	err = pullModel(newPullCmd(), env.client, networkFQDN, false)
	require.NoError(t, err)

	models, err = listModels(false, env.client, true, false, "")
	require.NoError(t, err)

	if len(models) == 0 {
		t.Fatal("No models found after pull")
	}

	if strings.Contains(models, modelID) == false {
		t.Fatalf("Pulled model ID %s not found in model list", modelID)
	}
}
