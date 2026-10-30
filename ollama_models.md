# Ollama judge models

All LLM inference was local via ollama (judging runs: ollama 0.32.1; digests read from the same host under 0.35.0, default 4-bit
quantizations of the library tags below). Generation options per model are in
`ml/llm_features.py` (`MODEL_CFG`, `MODEL_CFG_THINK`); `num_ctx` is pinned to 8192 and
seeds are fixed (`seed_base + attempt`, resampling on parse failure).

Digests: `manifest` is the ollama model ID (`ollama list` shows its first 12 hex
digits; full value = sha256 of the tag's manifest). `weights` is the digest of the
GGUF model layer and is the stronger identity check (`ollama show --modelfile <tag>`
prints the blob path that contains it).

| Paper name | ollama tag | Role in paper | manifest (model ID) | weights layer | Status |
|---|---|---|---|---|---|
| gemma3-12B | `gemma3:12b` | generic-dialogue, BGR+OSCE, phase+technique, anti-band judges | `sha256:f4031aab637d1ffa37b42570452ae0e4fad0314754d17ded67322e4b95836f8a` | `sha256:e8ad13eff07a78d89926e9e8b882317d082ef5bf9768ad7b50fcdbbcd63748de` | read from the judging host |
| qwen3-14B | `qwen3:14b` | generic-dialogue judge, think mode | `sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8` | `sha256:a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e` | read from the judging host |
| qwen3.6-27B | `qwen3.6:27b` | BGR+OSCE judge, screened, not in the primary ensemble | `sha256:a50eda8ed977ab48a12431878896b27ffd5cef552c17af3317d9623b939a7f1e` | `sha256:83c54730a5fea8a0958598c01617c1419c431e93b33bacf980b49a420c798926` | read from the judging host |
| granite4.1-30B | `granite4.1:30b` | BGR+OSCE judge | `sha256:3f3e5df8a021439fd6f867a0e526bdc303cac79c811201cb6bac193298cb9fcd` | `sha256:b33e4376e3581d11236ea53ced6b38399f6e91c0a391488486dc0827972f23f6` | read from the judging host |
| phi4-14B | `phi4:14b` | BGR+OSCE judge, screened, not in the primary ensemble | `sha256:ac896e5b8b34a1f4efa7b14d7520725140d5512484457fab45d2a4ea14c69dba` | `sha256:fd7b6731c33c57f61767612f56517460ec2d1e2e5a3f0163e0eb3d8d8cb5df20` | read from the judging host |
| granite4.1-8B | `granite4.1:8b` | screening candidate (rejected) | `sha256:444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852` | `sha256:ed902ac9eb6adce5a90c6a08c8ea201b50e23fdc5976d1cd0362006afac5309e` | read from the judging host |
| qwen3.5-27B | `qwen3.5:27b` | think-mode screening (negative) | `sha256:7653528ba5cba4dd8e19da24aaddc7f4d0b5ecd93571c0825dfd4137958ec06e` | `sha256:d4b8b4f4c350f5d322dc8235175eeae02d32c6f3fd70bdb9ea481e3abb7d7fc4` | read from a development host (screening only) |
| gemma4-12B | `gemma4:12b` | screening (negative, all-5s collapse) | `sha256:4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c` | `sha256:1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606` | read from the judging host |

Library tags are mutable. If a tag now resolves to a different digest, results may
differ from the paper; pull by the digest above where the registry still serves it.
