//go:build e2e

// Package e2e contains end-to-end tests that build and run the full
// model-runner stack (server + llama.cpp backend + CLI) from source.
//
// These tests require:
//   - The llamacpp submodule to be initialised and built (make build-llamacpp)
//   - A successful `make build` so that model-runner, model-cli, and dmr exist
//
// Run with:
//
//	go test -v -count=1 -tags=e2e -timeout=15m ./e2e/
package e2e

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

const (
	// testModel is small enough to pull quickly in CI.
	testModel = "ai/smollm2:135M-Q4_0"

	serverStartTimeout = 60 * time.Second
)

var (
	// serverURL is the base URL of the running model-runner instance.
	serverURL string
	// cliBin is the absolute path to the model-cli binary.
	cliBin string
)

// TestMain builds the binaries, starts the server (same pattern as dmr),
// and tears it down after all tests complete.
func TestMain(m *testing.M) {
	code := run(m)
	os.Exit(code)
}

func run(m *testing.M) int {
	// go test sets cwd to the package directory (e2e/), so the repo root is ../
	root, err := filepath.Abs("..")
	if err != nil {
		fmt.Fprintf(os.Stderr, "e2e: %v\n", err)
		return 1
	}

	// ── 1. Build binaries ──────────────────────────────────────────────
	fmt.Fprintln(os.Stderr, "e2e: building server and CLI...")
	if err := makeTarget(root, "build"); err != nil {
		fmt.Fprintf(os.Stderr, "e2e: make build failed: %v\n", err)
		return 1
	}

	serverBin := filepath.Join(root, "model-runner")
	cliBin = filepath.Join(root, "cmd", "cli", "model-cli")
	llamaBin := filepath.Join(root, "llamacpp", "install", "bin")

	for _, path := range []string{serverBin, cliBin, llamaBin} {
		if _, err := os.Stat(path); err != nil {
			fmt.Fprintf(os.Stderr, "e2e: not found: %s\n", path)
			return 1
		}
	}

	// ── 2. Start model-runner (same pattern as cmd/dmr) ────────────────
	port, err := freePort()
	if err != nil {
		fmt.Fprintf(os.Stderr, "e2e: %v\n", err)
		return 1
	}
	serverURL = "http://localhost:" + strconv.Itoa(port)
	fmt.Fprintf(os.Stderr, "e2e: starting model-runner on port %d\n", port)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server := exec.CommandContext(ctx, serverBin)
	server.Dir = root
	server.Env = append(os.Environ(),
		"MODEL_RUNNER_PORT="+strconv.Itoa(port),
		"LLAMA_SERVER_PATH="+llamaBin,
	)
	server.Stdout = os.Stderr
	server.Stderr = os.Stderr

	if err := server.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "e2e: failed to start server: %v\n", err)
		return 1
	}
	defer func() {
		cancel()
		_ = server.Wait()
	}()

	// ── 3. Wait for health ─────────────────────────────────────────────
	if err := waitForServer(serverURL+"/models", serverStartTimeout); err != nil {
		fmt.Fprintf(os.Stderr, "e2e: %v\n", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "e2e: server ready at %s\n", serverURL)

	// ── 4. Run tests ───────────────────────────────────────────────────
	return m.Run()
}

func makeTarget(dir, target string) error {
	cmd := exec.Command("make", target)
	cmd.Dir = dir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("finding free port: %w", err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func waitForServer(url string, timeout time.Duration) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("server not ready after %s", timeout)
}

// runCLI executes the model-cli binary with the given arguments and
// MODEL_RUNNER_HOST pointing to the test server. The subprocess is
// cancelled if the test's context expires.
func runCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	cmd := exec.CommandContext(t.Context(), cliBin, args...)
	cmd.Env = append(os.Environ(), "MODEL_RUNNER_HOST="+serverURL)
	out, err := cmd.CombinedOutput()
	return string(out), err
}
