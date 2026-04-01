package readline

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

const (
	defaultEditor = "vi"
	defaultShell  = "/bin/bash"
	windowsEditor = "notepad"
	windowsShell  = "cmd"
)

func platformize(linux, windows string) string {
	if runtime.GOOS == "windows" {
		return windows
	}
	return linux
}

func defaultEnvShell() []string {
	shell := os.Getenv("SHELL")
	if len(shell) == 0 {
		shell = platformize(defaultShell, windowsShell)
	}
	flag := "-c"
	if shell == windowsShell {
		flag = "/C"
	}
	return []string{shell, flag}
}

func resolveEditor() ([]string, bool) {
	editor := strings.TrimSpace(os.Getenv("EDITOR"))
	if len(editor) == 0 {
		editor = platformize(defaultEditor, windowsEditor)
	}

	if !strings.Contains(editor, " ") {
		return []string{editor}, false
	}

	if !strings.ContainsAny(editor, "\"'\\") {
		return strings.Split(editor, " "), false
	}

	shell := defaultEnvShell()
	return append(shell, editor), true
}

func buildEditorCmd(filePath string) *exec.Cmd {
	args, shell := resolveEditor()

	if shell {
		// The editor string is the last element — append the file path to it
		// so the shell interprets the full command.
		args[len(args)-1] = fmt.Sprintf("%s %s", args[len(args)-1], filePath)
	} else {
		args = append(args, filePath)
	}

	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd
}

func runEditor(content string) (string, error) {
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

	cmd := buildEditorCmd(tmpFile.Name())
	if err := cmd.Run(); err != nil {
		return content, err
	}

	edited, err := os.ReadFile(tmpFile.Name())
	if err != nil {
		return content, err
	}

	return strings.TrimRight(string(edited), "\r\n"), nil
}
