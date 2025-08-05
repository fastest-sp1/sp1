//debug
#include "json.hpp" 
// Use the nlohmann json namespace
using json = nlohmann::json;

// --- JSON Serialization for bb31_t ---
// This tells nlohmann::json how to convert bb31_t to/from JSON.
// We'll just use its underlying integer value.
void to_json(json& j, const bb31_t& p) {
    j = p.val;
}

void from_json(const json& j, bb31_t& p) {
    p.val = j.get<unsigned int>();
}

// --- JSON Serialization for bb31_quartic_extension_t ---
// This tells nlohmann::json how to convert our extension field element.
// We'll represent it as a JSON object like the one in Rust.
void to_json(json& j, const bb31_quartic_extension_t& p) {
    j = json{
        {"value", {p.coeffs[0], p.coeffs[1], p.coeffs[2], p.coeffs[3]}},
        {"_phantom", nullptr} // Using nullptr for JSON null
    };
}

// from_json is not strictly needed for this debug task, but it's good practice.
void from_json(const json& j, bb31_quartic_extension_t& p) {
    j.at("value").get_to(p.coeffs);
}
//end debug