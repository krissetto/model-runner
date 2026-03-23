//go:build e2e

package e2e

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/model-runner/cmd/cli/desktop"
)

// TestE2E_Inference runs all inference API tests sequentially as subtests
// to ensure correct ordering (pull → list → chat → streaming → remove).
func TestE2E_Inference(t *testing.T) {
	t.Run("PullModel", func(t *testing.T) {
		t.Logf("pulling model %s via API...", testModel)

		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, serverURL+"/models/create", strings.NewReader(`{"from":"`+testModel+`"}`))
		if err != nil {
			t.Fatalf("creating request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("pull request failed: %v", err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
			t.Fatalf("pull failed: status=%d body=%s", resp.StatusCode, body)
		}
		t.Logf("pull completed (status %d)", resp.StatusCode)
	})

	t.Run("ListModels", func(t *testing.T) {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, serverURL+"/engines/v1/models", nil)
		if err != nil {
			t.Fatalf("creating request: %v", err)
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("list models failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("list models: status=%d body=%s", resp.StatusCode, body)
		}

		var result struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatalf("decoding list response: %v", err)
		}

		found := false
		for _, m := range result.Data {
			t.Logf("  model: %s", m.ID)
			if strings.Contains(m.ID, "smollm2") {
				found = true
			}
		}
		if !found {
			t.Fatalf("expected %s in model list", testModel)
		}
	})

	t.Run("ChatCompletionNonStreaming", func(t *testing.T) {
		reqBody := desktop.OpenAIChatRequest{
			Model: testModel,
			Messages: []desktop.OpenAIChatMessage{
				{Role: "user", Content: "Say hello in exactly one word."},
			},
			Stream: false,
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}

		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, serverURL+"/engines/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("creating request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()

		respBody, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("chat completion failed: status=%d body=%s", resp.StatusCode, respBody)
		}

		var chatResp desktop.OpenAIChatResponse
		if err := json.Unmarshal(respBody, &chatResp); err != nil {
			t.Fatalf("decoding response: %v (body=%s)", err, respBody)
		}

		if len(chatResp.Choices) == 0 {
			t.Fatal("no choices in response")
		}

		content := chatResp.Choices[0].Message.Content
		t.Logf("model response: %q", content)

		if content == "" {
			t.Fatal("empty response content")
		}
		if chatResp.Choices[0].Message.Role != "assistant" {
			t.Errorf("expected role=assistant, got %q", chatResp.Choices[0].Message.Role)
		}
	})

	t.Run("ChatCompletionStreaming", func(t *testing.T) {
		reqBody := desktop.OpenAIChatRequest{
			Model: testModel,
			Messages: []desktop.OpenAIChatMessage{
				{Role: "user", Content: "Count from 1 to 3."},
			},
			Stream: true,
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}

		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, serverURL+"/engines/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("creating request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			t.Fatalf("streaming request failed: status=%d body=%s", resp.StatusCode, respBody)
		}

		var accumulated strings.Builder
		var chunkCount int
		gotDone := false

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			if line == "" {
				continue
			}

			if !strings.HasPrefix(line, "data: ") {
				if strings.HasPrefix(line, ":") {
					continue
				}
				t.Fatalf("unexpected SSE line: %q", line)
			}

			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				gotDone = true
				break
			}

			var chunk desktop.OpenAIChatResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				t.Fatalf("parsing SSE chunk: %v (data=%q)", err, data)
			}

			chunkCount++
			if len(chunk.Choices) > 0 {
				accumulated.WriteString(chunk.Choices[0].Delta.Content)
			}
		}

		if err := scanner.Err(); err != nil {
			t.Fatalf("reading stream: %v", err)
		}

		t.Logf("received %d chunks, accumulated: %q", chunkCount, accumulated.String())

		if !gotDone {
			t.Error("stream did not end with [DONE]")
		}
		if chunkCount == 0 {
			t.Error("received no SSE chunks")
		}
		if accumulated.Len() == 0 {
			t.Error("accumulated content is empty")
		}
	})

	t.Run("RemoveModel", func(t *testing.T) {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodDelete, fmt.Sprintf("%s/models/%s", serverURL, testModel), nil)
		if err != nil {
			t.Fatalf("creating request: %v", err)
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("delete request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("delete failed: status=%d body=%s", resp.StatusCode, body)
		}
		t.Logf("model %s removed", testModel)
	})
}
