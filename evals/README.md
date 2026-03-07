# RAG Evaluation

Two ways to run the evaluation against an indexed collection.

## Option A — Automated script (recommended)

Runs all questions directly against ChromaDB, checks must_contain terms
programmatically, and exits non-zero if any question fails.

```bash
# Run against the default samsung_tv collection
uv run python evals/run_eval.py

# Override collection or retrieval depth
uv run python evals/run_eval.py --collection samsung_tv --n-results 8

# Run a different eval file
uv run python evals/run_eval.py --eval evals/my_other_eval.xml
```

Example output:

```
Loading embedding model (first run will take ~30s)...
Collection : samsung_tv
Questions  : 10

  Q 1 [PASS] input          remote control key events Samsung TV web app
  Q 2 [PASS] configuration  config.xml required fields Samsung Smart TV
  Q 3 [FAIL] lifecycle      Samsung TV web app lifecycle visibility events
            missing: resume
  ...

=================================================================
  Score: 8/10  (80%)
=================================================================

|  # | Topic          | Result | Missing terms
|----|----------------|--------|------------------------------
|  1 | input          | PASS   | -
|  2 | configuration  | PASS   | -
|  3 | lifecycle      | FAIL   | resume
  ...
```

The script exits 0 if all questions pass, 1 otherwise — usable in CI.

## Option B — Interactive Claude session

Paste this prompt into a Claude Code session that has the `devdocs-rag`
MCP server connected:

---

Please run the RAG evaluation defined in evals/samsung_tv_eval.xml.

For each of the 10 questions:
1. Call search_docs with the suggested_query and collection="samsung_tv"
2. Use the results to answer the question
3. Check whether all must_contain terms appear in your answer
4. Record PASS or FAIL

At the end, print a summary table:

| # | Topic | Result | Missing terms (if any) |
|---|-------|--------|------------------------|
...

Then give a final score: X/10 passed.

---

## Adding new eval files

Create a new XML file following the schema in `samsung_tv_eval.xml`:

```xml
<eval name="my_docs" collection="my_collection" version="1">
  <question id="1" topic="my_topic">
    <prompt>Natural language question here.</prompt>
    <tools>
      <tool name="search_docs">
        <suggested_query>search query to use</suggested_query>
      </tool>
    </tools>
    <must_contain>
      <term>keyword1</term>
      <term>keyword2</term>
    </must_contain>
    <scoring>pass if all must_contain terms appear in the final answer</scoring>
  </question>
</eval>
```

Then run: `uv run python evals/run_eval.py --eval evals/my_docs_eval.xml`
