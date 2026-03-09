package commands

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMergedLogEqualTimestamps(t *testing.T) {
	f1 := filepath.Join(t.TempDir(), "a.log")
	_ = os.WriteFile(f1, []byte("[2026-03-09T12:00:00.000000000Z] line a\n"), 0644)
	f2 := filepath.Join(t.TempDir(), "b.log")
	_ = os.WriteFile(f2, []byte("[2026-03-09T12:00:00.000000000Z] line b\n"), 0644)

	done := make(chan error, 1)
	go func() { done <- printMergedLog(f1, f2) }()

	select {
	case <-done:
		// ok
	case <-time.After(3 * time.Second):
		t.Fatal("printMergedLog hung — equal timestamp deadlock")
	}
}
