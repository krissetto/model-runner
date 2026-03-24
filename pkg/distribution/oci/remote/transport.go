package remote

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"

	"github.com/docker/model-runner/pkg/distribution/oci/authn"
	"github.com/docker/model-runner/pkg/distribution/oci/reference"
)

// PullScope is the scope for pulling from a registry.
const PullScope = "pull"

// PushScope is the scope for pushing to a registry.
const PushScope = "push,pull"

// PingResponse contains information from a registry ping.
type PingResponse struct {
	WWWAuthenticate WWWAuthenticate
}

// WWWAuthenticate contains parsed WWW-Authenticate header information.
type WWWAuthenticate struct {
	Realm   string
	Service string
	Scope   string
}

// Token represents an authentication token.
type Token struct {
	Token       string `json:"token"`
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
}

// Ping pings a registry and returns authentication information.
func Ping(ctx context.Context, reg reference.Registry, transport http.RoundTripper) (*PingResponse, error) {
	if transport == nil {
		transport = http.DefaultTransport
	}

	client := &http.Client{Transport: transport}
	scheme := reg.Scheme()

	pingURL := fmt.Sprintf("%s://%s/v2/", scheme, reg.RegistryStr())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, pingURL, http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("creating ping request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("pinging registry: %w", err)
	}
	defer resp.Body.Close()

	// Parse WWW-Authenticate header
	wwwAuth := resp.Header.Get("WWW-Authenticate")
	if wwwAuth == "" {
		// No auth required or already authenticated
		return &PingResponse{}, nil
	}

	pr := &PingResponse{
		WWWAuthenticate: parseWWWAuthenticate(wwwAuth),
	}

	return pr, nil
}

// parseWWWAuthenticate parses a WWW-Authenticate header.
func parseWWWAuthenticate(header string) WWWAuthenticate {
	result := WWWAuthenticate{}

	// Remove "Bearer " prefix
	header = strings.TrimPrefix(header, "Bearer ")

	// Parse key=value pairs
	for _, part := range strings.Split(header, ",") {
		part = strings.TrimSpace(part)
		kv := strings.SplitN(part, "=", 2)
		if len(kv) != 2 {
			continue
		}
		key := strings.TrimSpace(kv[0])
		value := strings.Trim(strings.TrimSpace(kv[1]), "\"")

		switch key {
		case "realm":
			result.Realm = value
		case "service":
			result.Service = value
		case "scope":
			result.Scope = value
		}
	}

	return result
}

// privateOrLoopbackCIDRs lists IP ranges that must never be contacted as a
// token-exchange realm. Allowing requests to these addresses would let a
// malicious registry pivot Model Runner into an internal-service proxy
// (SSRF), reaching endpoints that are not accessible from the public internet.
var privateOrLoopbackCIDRs = func() []*net.IPNet {
	cidrs := []string{
		"127.0.0.0/8",    // loopback IPv4
		"::1/128",        // loopback IPv6
		"169.254.0.0/16", // link-local IPv4 / AWS EC2 instance-metadata
		"fe80::/10",      // link-local IPv6
		"10.0.0.0/8",     // RFC-1918 private
		"172.16.0.0/12",  // RFC-1918 private
		"192.168.0.0/16", // RFC-1918 private
		"fc00::/7",       // IPv6 ULA
	}
	var nets []*net.IPNet
	for _, cidr := range cidrs {
		_, n, err := net.ParseCIDR(cidr)
		if err == nil {
			nets = append(nets, n)
		}
	}
	return nets
}()

// internalHostnames lists hostnames that must never be used as a realm, regardless
// of what IP address they resolve to.
var internalHostnames = []string{
	"localhost",
	"host.docker.internal",
	"model-runner.docker.internal",
	"gateway.docker.internal",
}

// validateRealmURL rejects realm URLs that point to private, loopback, or
// link-local addresses and known internal hostnames. This prevents SSRF: a
// malicious registry that returns a crafted WWW-Authenticate header could
// otherwise cause Model Runner to send (potentially credentialed) HTTP
// requests to any host-reachable service — including the AWS EC2
// instance-metadata endpoint, Kubernetes API servers, or other internal APIs.
func validateRealmURL(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid realm URL: %w", err)
	}

	host := u.Hostname()

	// Block well-known internal hostnames regardless of DNS resolution.
	for _, internal := range internalHostnames {
		if strings.EqualFold(host, internal) {
			return fmt.Errorf("realm URL hostname %q is not allowed", host)
		}
	}

	// Resolve the hostname and reject any address in a private/loopback range.
	ips, err := net.LookupHost(host)
	if err != nil {
		return fmt.Errorf("resolving realm hostname %q: %w", host, err)
	}
	for _, ipStr := range ips {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			continue
		}
		for _, cidr := range privateOrLoopbackCIDRs {
			if cidr.Contains(ip) {
				return fmt.Errorf("realm URL resolves to a disallowed address %s", ipStr)
			}
		}
	}
	return nil
}

// Exchange exchanges credentials for a bearer token.
func Exchange(ctx context.Context, reg reference.Registry, auth authn.Authenticator, transport http.RoundTripper, scopes []string, pr *PingResponse) (*Token, error) {
	if transport == nil {
		transport = http.DefaultTransport
	}

	client := &http.Client{Transport: transport}

	// Validate the realm URL before making any request. The realm value comes
	// from the registry's WWW-Authenticate header and is therefore attacker-
	// controlled. Without this check a malicious registry could use Model
	// Runner as an SSRF proxy to reach internal services.
	if err := validateRealmURL(pr.WWWAuthenticate.Realm); err != nil {
		return nil, fmt.Errorf("realm URL rejected: %w", err)
	}

	// Build token request URL
	tokenURL, err := url.Parse(pr.WWWAuthenticate.Realm)
	if err != nil {
		return nil, fmt.Errorf("parsing realm URL: %w", err)
	}

	q := tokenURL.Query()
	if pr.WWWAuthenticate.Service != "" {
		q.Set("service", pr.WWWAuthenticate.Service)
	}
	for _, scope := range scopes {
		q.Add("scope", scope)
	}
	tokenURL.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, tokenURL.String(), http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("creating token request: %w", err)
	}

	// Add authentication if provided
	if auth != nil {
		cfg, err := auth.Authorization()
		if err != nil {
			return nil, fmt.Errorf("getting auth config: %w", err)
		}
		if cfg.Username != "" && cfg.Password != "" {
			req.SetBasicAuth(cfg.Username, cfg.Password)
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching token: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		slog.DebugContext(ctx, "token request failed",
			"status", resp.StatusCode,
			"body", string(body),
		)
		return nil, fmt.Errorf("token request failed: unexpected status %d from token endpoint", resp.StatusCode)
	}

	var token Token
	if err := json.NewDecoder(resp.Body).Decode(&token); err != nil {
		return nil, fmt.Errorf("decoding token response: %w", err)
	}

	// Some registries return access_token instead of token
	if token.Token == "" && token.AccessToken != "" {
		token.Token = token.AccessToken
	}

	return &token, nil
}

// BearerTransport wraps an http.RoundTripper with bearer token authentication.
type BearerTransport struct {
	Transport http.RoundTripper
	Token     string
}

// RoundTrip implements http.RoundTripper.
func (t *BearerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req2 := req.Clone(req.Context())
	if t.Token != "" {
		req2.Header.Set("Authorization", "Bearer "+t.Token)
	}
	if t.Transport == nil {
		return http.DefaultTransport.RoundTrip(req2)
	}
	return t.Transport.RoundTrip(req2)
}
