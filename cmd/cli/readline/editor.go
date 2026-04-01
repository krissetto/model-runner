package readline

import (
	"os"
	"os/exec"
	"strings"
)

func runEditor(content string, defaultEditor string) (string, error) {
	tmpFile, err := os.CreateTemp("", "docker-model-prompt-*.txt")
	if err != nil {
		return content, err
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write([]byte(content)); err != nil {
		tmpFile.Close()
		return content, err
	}
	tmpFile.Close()

	editor := strings.TrimSpace(os.Getenv("EDITOR"))
	if editor == "" {
		editor = defaultEditor
	}

	// handle for env varibles set with args
	parts := strings.Fields(editor)
	args := append(parts[1:], tmpFile.Name())
	cmd := exec.Command(parts[0], args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return content, err
	}

	edited, err := os.ReadFile(tmpFile.Name())
	if err != nil {
		return content, err
	}

	return string(edited), nil
}
