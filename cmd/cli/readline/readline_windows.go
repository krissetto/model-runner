package readline

func handleCharCtrlZ(fd uintptr, state any) (string, error) {
	// not supported
	return "", nil
}

func openInEditor(fd uintptr, termios any, content string) (string, error) {
	s := termios.(*State)

	if err := UnsetRawMode(fd, s); err != nil {
		return content, err
	}

	edited, err := runEditor(content, "notepad")

	// Always restore raw mode using the original state, whether the editor
	// succeeded or failed, so the terminal returns to its previous configuration.
	if _, restoreErr := SetRawMode(fd); restoreErr != nil {
		return content, restoreErr
	}

	if err != nil {
		return content, err
	}

	return edited, nil
}
