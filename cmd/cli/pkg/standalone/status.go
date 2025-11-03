package standalone

// StatusPrinter is the interface used to print status updates.
type StatusPrinter interface {
	// Printf should perform formatted printing.
	Printf(format string, args ...any)
	// Println should perform line-based printing.
	Println(args ...any)
	// Write implements io.Writer for stream-based output.
	Write(p []byte) (n int, err error)
	// GetFdInfo returns the file descriptor and terminal status for the output.
	GetFdInfo() (fd uintptr, isTerminal bool)
}

// noopPrinter is used to silence auto-install progress if desired.
type noopPrinter struct{}

// Printf implements StatusPrinter.Printf.
func (*noopPrinter) Printf(format string, args ...any) {}

// Println implements StatusPrinter.Println.
func (*noopPrinter) Println(args ...any) {}

// Write implements StatusPrinter.Write.
func (*noopPrinter) Write(p []byte) (n int, err error) {
	return len(p), nil
}

// GetFdInfo implements StatusPrinter.GetFdInfo.
func (*noopPrinter) GetFdInfo() (fd uintptr, isTerminal bool) {
	return 0, false
}

// NoopPrinter returns a StatusPrinter that does nothing.
func NoopPrinter() StatusPrinter {
	return &noopPrinter{}
}
