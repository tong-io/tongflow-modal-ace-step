# tongflow-modal-ace-step

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text-to-music generation with **ACE-Step** (`ACE-Step/acestep-v15-xl-base` and `ACE-Step/acestep-5Hz-lm-4B`), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Music generation** (`gen-music`) — generate music from a text prompt.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The ACE-Step weights are public — no Hugging Face token required.
