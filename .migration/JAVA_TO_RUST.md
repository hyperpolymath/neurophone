# Java/Kotlin → Rust Migration Guide

## Policy
Java/Kotlin code should be migrated to Rust for:
- Memory safety without GC
- Better performance
- Smaller binaries

## Migration Steps
1. Create `Cargo.toml` project
2. Convert Java classes to Rust structs/enums
3. Replace JVM concurrency with Rust async/threads
4. Remove pom.xml/build.gradle after migration

## Status: PENDING MIGRATION
