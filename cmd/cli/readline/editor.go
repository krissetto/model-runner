package readline

import (
	"os"
	"os/exec"
	"runtime"
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

	cmd := buildEditorCmd(defaultEditor, tmpFile.Name())
	if err := cmd.Run(); err != nil {
		return content, err
	}

	edited, err := os.ReadFile(tmpFile.Name())
	if err != nil {
		return content, err
	}

	result := strings.TrimRight(string(edited), "\r\n")

	return result, nil
}

func buildEditorCmd(defaultEditor string, filePath string) *exec.Cmd {
	editor := strings.TrimSpace(os.Getenv("EDITOR"))
	if editor == "" {
		editor = defaultEditor
	}

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", editor+" \""+filePath+"\"")
	} else {
		cmd = exec.Command("sh", "-c", editor+" \"$1\"", "--", filePath)
	}
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd
}
