# Distributed PJRT and NCCL

RXLA separates collective execution from distributed client rendezvous:

1. StableHLO/SPMD describes collectives in the compiled program. On CUDA,
   XLA:GPU normally implements these operations with NCCL.
2. PJRT client creation discovers the global device topology and exchanges
   bootstrap data through a process-shared key-value store.

NCCL communicators are therefore not application-owned handles in the normal
RXLA execution path. XLA owns their creation, scheduling, stream dependencies,
and failure handling.

## Creating distributed clients

Implement `KeyValueStore` using the deployment's coordination service. The
implementation must be thread-safe because PJRT may call it from background
threads:

```rust,ignore
use rxla_pjrt::{
    ClientOptions, DistributedClientConfig, KeyValueStore, Plugin,
};
use std::sync::Arc;

let store: Arc<dyn KeyValueStore> = Arc::new(MyEtcdStore::connect(endpoint)?);
let distributed = DistributedClientConfig::new(node_id, node_count, store)?;
let options = ClientOptions::new()
    .set("visible_devices", local_cuda_ordinals);

// Every node loads the same plugin and joins with a distinct node_id.
let client = plugin.create_distributed_client(&options, distributed)?;
```

`DistributedClientConfig` injects the CUDA PJRT `node_id` and `num_nodes`
options. Explicit values with those names in `ClientOptions` are replaced so
the validated topology remains authoritative. The client retains the callback
state and store until after `PJRT_Client_Destroy`.

`InMemoryKeyValueStore` is provided for tests with multiple clients in one
process. It cannot coordinate separate processes or machines.

## Callback guarantees

The Rust bridge:

- preserves binary keys and values without UTF-8 conversion;
- implements blocking `get`, nonblocking `try_get`, and `put`;
- maps store failures to PJRT status codes;
- copies returned values into storage released by PJRT's value deleter;
- catches Rust panics before they can cross the C ABI boundary.

Production stores should namespace keys per job. PJRT only prevents collisions
inside one plugin/user; the application remains responsible for separating
unrelated jobs that share a store.

## Direct collectives extension

`Plugin::supports_collectives_extension()` reports whether a plugin advertises
the experimental PJRT direct-collectives extension. This is only capability
discovery: the extension currently obtains its collective implementation
through other experimental APIs and is not the portable CUDA execution path.

The tested JAX CUDA 13 plugin does not advertise this extension, while it does
support StableHLO collectives and distributed client rendezvous. Absence of the
extension therefore does **not** mean that NCCL is unavailable.

## Validation

The ignored native test creates two logical CUDA PJRT nodes against one visible
GPU and verifies that both clients discover the distributed process topology:

```sh
PJRT_PLUGIN_PATH=/trusted/xla_cuda_plugin.so \
  cargo test -p rxla-pjrt --test distributed \
  cuda_plugin_creates_two_distributed_nodes_through_rendezvous_callbacks \
  -- --ignored --exact
```

This validates PJRT rendezvous and topology creation. A true NCCL data-path
test still requires at least two CUDA devices (or two hosts), a compiled
StableHLO collective, and per-node execution launched concurrently.
