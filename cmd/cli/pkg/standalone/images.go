package standalone

import (
	"context"
	"fmt"

	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/jsonmessage"
	gpupkg "github.com/docker/model-runner/cmd/cli/pkg/gpu"
)

// EnsureControllerImage ensures that the controller container image is pulled.
func EnsureControllerImage(ctx context.Context, dockerClient client.ImageAPIClient, gpu gpupkg.GPUSupport, backend string, printer StatusPrinter) error {
	imageName := controllerImageName(gpu, backend)

	// Perform the pull.
	out, err := dockerClient.ImagePull(ctx, imageName, image.PullOptions{})
	if err != nil {
		return fmt.Errorf("failed to pull image %s: %w", imageName, err)
	}
	defer out.Close()

	// Display pull progress using Docker's built-in display handler
	fd, isTerminal := printer.GetFdInfo()
	if err := jsonmessage.DisplayJSONMessagesStream(out, printer, fd, isTerminal, nil); err != nil {
		return fmt.Errorf("failed to pull image %s: %w", imageName, err)
	}

	printer.Println("Successfully pulled", imageName)
	return nil
}

// PruneControllerImages removes any unused controller container images.
func PruneControllerImages(ctx context.Context, dockerClient client.ImageAPIClient, printer StatusPrinter) error {
	// Remove the standard image, if present.
	imageNameCPU := fmtControllerImageName(ControllerImage, controllerImageVersion(), "")
	if _, err := dockerClient.ImageRemove(ctx, imageNameCPU, image.RemoveOptions{}); err == nil {
		printer.Println("Removed image", imageNameCPU)
	}

	// Remove the CUDA GPU image, if present.
	imageNameCUDA := fmtControllerImageName(ControllerImage, controllerImageVersion(), "cuda")
	if _, err := dockerClient.ImageRemove(ctx, imageNameCUDA, image.RemoveOptions{}); err == nil {
		printer.Println("Removed image", imageNameCUDA)
	}
	return nil
}
