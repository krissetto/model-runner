//go:build !windows

package readline

import (
	"syscall"
)

func handleCharCtrlZ(fd uintptr, termios any) (string, error) {
	t := termios.(*Termios)
	if err := UnsetRawMode(fd, t); err != nil {
		return "", err
	}

	_ = syscall.Kill(0, syscall.SIGSTOP)

	// on resume...
	return "", nil
}

func openInEditor(fd uintptr, termios any, content string) (string, error) {
	t := termios.(*Termios)

	if err := UnsetRawMode(fd, t); err != nil {
		return content, err
	}

	edited, err := runEditor(content, "vi")
	if err != nil {
		SetRawMode(fd)
		return content, err
	}

	if _, err := SetRawMode(fd); err != nil {
		return edited, err
	}

	return edited, nil
}
