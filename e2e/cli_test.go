//go:build e2e

package e2e

import (
	"strings"
	"testing"
)

func TestE2E_CLI(t *testing.T) {
	for _, bc := range backends {
		bc := bc
		t.Run(bc.name, func(t *testing.T) {
			t.Run("Pull", func(t *testing.T) {
				out, err := runCLI(t, "pull", bc.model)
				if err != nil {
					t.Fatalf("cli pull failed: %v\noutput: %s", err, out)
				}
				t.Logf("pull output: %s", out)
			})

			t.Run("List", func(t *testing.T) {
				out, err := runCLI(t, "ls")
				if err != nil {
					t.Fatalf("cli ls failed: %v\noutput: %s", err, out)
				}
				if !strings.Contains(out, "smollm2") {
					t.Errorf("expected smollm2 in list output, got:\n%s", out)
				}
			})

			t.Run("Run", func(t *testing.T) {
				out, err := runCLI(t, "run", bc.model, "Say hi in one word.")
				if err != nil {
					t.Fatalf("cli run failed: %v\noutput: %s", err, out)
				}
				if strings.TrimSpace(out) == "" {
					t.Fatal("cli run produced empty output")
				}
				t.Logf("run output: %s", out)
			})

			t.Run("Remove", func(t *testing.T) {
				out, err := runCLI(t, "rm", "-f", bc.model)
				if err != nil {
					t.Fatalf("cli rm failed: %v\noutput: %s", err, out)
				}
			})
		})
	}
}
