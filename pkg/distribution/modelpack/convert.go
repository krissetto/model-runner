package modelpack

import (
	"encoding/json"
)

// IsModelPackConfig detects if raw config bytes are in ModelPack format.
// It parses the JSON structure for precise detection, avoiding false positives from string matching.
// ModelPack format characteristics: config.paramSize or descriptor.createdAt
// Docker format uses: config.parameters and descriptor.created
func IsModelPackConfig(raw []byte) bool {
	if len(raw) == 0 {
		return false
	}

	// Parse as map to check actual JSON structure
	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return false
	}

	// Check for config.paramSize (ModelPack-specific field)
	if configRaw, ok := parsed["config"]; ok {
		var config map[string]json.RawMessage
		if err := json.Unmarshal(configRaw, &config); err == nil {
			if _, hasParamSize := config["paramSize"]; hasParamSize {
				return true
			}
		}
	}

	// Check for descriptor.createdAt (ModelPack uses camelCase)
	if descRaw, ok := parsed["descriptor"]; ok {
		var desc map[string]json.RawMessage
		if err := json.Unmarshal(descRaw, &desc); err == nil {
			if _, hasCreatedAt := desc["createdAt"]; hasCreatedAt {
				return true
			}
		}
	}

	// Check for modelfs (ModelPack-specific field name)
	if _, hasModelFS := parsed["modelfs"]; hasModelFS {
		return true
	}

	return false
}
