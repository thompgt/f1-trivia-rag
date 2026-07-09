---
name: f1-data-conventions
description: F1 dataset schema, era boundaries, and scoring rule changes. Use when writing ingestion, chunking, embedding, or retrieval filtering code.
allowed-tools: Read, Bash(python scripts/*)
---
- Points systems: pre-2003 (10-6-4-3-2-1), 2003-2009 (10-8-6-5-4-3-2-1), 2010+ (25-18-15...), +1 fastest lap 2019-2020 & 2025+
- Sprint races exist only 2021+, award separate points, don't conflate with GP results
- Disqualifications: store both "as-raced" and "final classified" result — trivia questions target both
- Driver/constructor name changes (e.g. team renames) need canonical + alias fields for entity resolution
