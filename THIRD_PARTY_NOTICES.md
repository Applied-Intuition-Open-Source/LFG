# Third-Party Notices

This repository bundles the minimal model code needed to run LFG checkpoints. The bundled
components keep their own licenses, which apply in addition to the repository's Apache-2.0
license in `LICENSE`.

- **Pi3** (https://github.com/yyfz/Pi3), bundled under `Pi3/`, is BSD-3-Clause. The upstream
  copyright notice and license text are retained in `Pi3/LICENSE`.
- **DINOv2** (https://github.com/facebookresearch/dinov2, Meta Platforms), whose derived files
  sit under `Pi3/pi3/models/dinov2/`, is Apache-2.0 and retains its headers in source.
- Model checkpoints are separate artifacts and carry their own terms; the LFG weights are
  CC BY-NC 4.0.

Evaluation baselines are downloaded at runtime and are not redistributed here: VGGT, Depth
Anything 3, SegFormer and MaskFormer, each under its own license.
