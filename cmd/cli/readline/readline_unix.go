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

	edited, err := runEditor(content)

	if _, restoreErr := SetRawMode(fd); restoreErr != nil {
		return content, restoreErr
	}

	if err != nil {
		return content, err
	}

	return edited, nil
}
