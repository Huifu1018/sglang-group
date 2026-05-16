# Method Notes

`sglang-group` exposes one SGLang algorithm name, `SGLANG_GROUP`, with multiple
runtime methods.

## `itl`

TokenTiming-style route:

```text
target ids -> target text
target text -> draft ids
draft model proposes draft ids
draft ids -> draft text
draft text -> target ids
DTW alignment diagnostics
target model verifies proxy ids
```

This method is useful for high-temperature sampling in the current MiniMax
benchmarks.

## `itl-base-slem`

First-paper SLEM/UAG-style route:

```text
target ids -> target text
target text -> draft ids
draft model proposes draft ids
draft ids + lookbehind -> draft text
draft text -> target ids
suffix alignment -> new target proxy ids
target model verifies proxy ids
```

This method is greedy-only and is the default `auto` route for
`temperature=0`.

## `itl-base-tli`

First-paper TLI route:

```text
assistant vocab token string == target vocab token string
assistant id -> target id
draft logits restricted to shared token strings
draft probabilities mapped into target vocabulary rows
target verifier runs speculative rejection sampling
```

This method is the default `auto` route for mid-temperature sampling.

## `auto`

Default routing:

```text
temperature == 0       -> itl-base-slem
0 < temperature < 0.9  -> itl-base-tli
temperature >= 0.9     -> itl
```

The threshold and route methods are configurable from the launch wrapper.
