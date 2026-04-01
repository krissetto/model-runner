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
	if err != nil {
		SetRawMode(fd)
		return content, err
	}

	if _, err := SetRawMode(fd); err != nil {
		return edited, err
	}

	return edited, nil
}
