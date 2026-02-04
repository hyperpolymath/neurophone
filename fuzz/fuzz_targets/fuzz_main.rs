#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz data processing with arbitrary binary input
    if data.is_empty() || data.len() > 100000 {
        return;
    }

    // Test UTF-8 handling
    let _ = std::str::from_utf8(data);

    // Test as potential structured data
    if data.len() >= 4 {
        let _ = serde_json::from_slice::<serde_json::Value>(data);
    }
});
