package readline

import "errors"

func handleCharCtrlZ(fd uintptr, state any) (string, error) {
	// not supported
	return "", nil
}

func openInEditor(fd uintptr, termios any, content string) (string, error) {
	s := termios.(*State)

	if err := UnsetRawMode(fd, s); err != nil {
		return content, err
	}

	edited, err := runEditor(content)

	if _, restoreErr := SetRawMode(fd); restoreErr != nil {
		return content, errors.Join(err, restoreErr)
	}

	if err != nil {
		return content, err
	}

	return edited, nil
}
