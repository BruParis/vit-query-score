import click
import cv2
import numpy as np
import einops
import itertools
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Set
from mmengine.config import Config
from mmengine.registry import MODELS

# import to trigger model registration
import vit_query_score.vit
import vit_query_score.vit_adapter

from vit_query_score.encoding import reencode_and_replace


def write_query_scores(
    video_fp, set_frame_idx: Set[int], array_query_score, H, W, output_path
):
    """
    Args:
        video_fp: Path to the input video file.
        list_query_score: List of query scores for each frame, shape [Nframes, H, W].
        H: Number of patches in height.
        W: Number of patches in width.
        output_path: Path to save the output video.
    """
    cap = cv2.VideoCapture(video_fp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    patch_h = height // H
    patch_w = width // W
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    progress_bar = tqdm(total=num_frames, desc="Writing query scores")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #if frame_idx >= len(array_query_score):
        #    print(f"frame_idx {frame_idx} >= len(array_query_score) {len(array_query_score)}")
        #    break

        if frame_idx not in set_frame_idx:
            out.write(frame)
            frame_idx += 1
            progress_bar.update(1)
            continue

        # Get the query scores for this frame
        query_scores = array_query_score[frame_idx]
        # make 1 - exp(-8 * score) to enhance the contrast
        qs_color = 1 - np.exp(-30 * query_scores)
        qs_color_255 = (qs_color * 255).astype(np.uint8)

        query_score_resize = cv2.resize(
            qs_color_255,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        heatmap = cv2.applyColorMap(query_score_resize, cv2.COLORMAP_JET)

        overlayed_frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        frame_h, frame_w, _ = frame.shape

        # Overlay patch scores on the frame
        for i, j in itertools.product(range(H), range(W)):
            # Get the patch's top-left corner
            y1, x1 = i * patch_h, j * patch_w
            # y2, x2 = y1 + patch_h, x1 + patch_w

            # Get the score for this patch (normalized for visualization)
            score = query_scores[i, j]

            # Optionally, write the score on the patch
            cv2.putText(
                overlayed_frame,
                f"{score:.3f}",
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # write lines to form a grid
            cv2.line(overlayed_frame, (x1, 0), (x1, frame_h), (128, 128, 128), 1)
            cv2.line(overlayed_frame, (0, y1), (frame_w, y1), (128, 128, 128), 1)

        # write the right and bottom border lines
        cv2.line(
            overlayed_frame,
            (frame_w - 1, 0),
            (frame_w - 1, frame_h),
            (128, 128, 128),
            1,
        )
        cv2.line(
            overlayed_frame,
            (0, frame_h - 1),
            (frame_w, frame_h - 1),
            (128, 128, 128),
            1,
        )

        # write two frames to slow down the video
        out.write(overlayed_frame)
        out.write(overlayed_frame)

        frame_idx += 1
        progress_bar.update(1)

    cap.release()
    out.release()


def generate_frame_tensors(cap, chunk_size=16, target_size=(160, 160), stride=4):
    frames = []
    frames_indices = []

    fr_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if fr_idx % stride != 0:
            frames_indices.append(fr_idx)
            fr_idx += 1
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        # Resize with aspect ratio, center crop
        # and normalize frame
        scale = min(target_size[0] / h, target_size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create blank canvas of target size
        canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

        # Calculate coordinates to center the resized image
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2

        # Place the resized image in the center of the canvas
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = frame

        # Now center crop if needed (only if resized image is larger than target)
        if new_h > target_size[0] or new_w > target_size[1]:
            h, w = frame.shape[:2]
            startx = w // 2 - (target_size[1] // 2)
            starty = h // 2 - (target_size[0] // 2)
            resized = frame[
                starty : starty + target_size[0], startx : startx + target_size[1]
            ]
        else:
            resized = canvas  # Use the padded version if no cropping needed

        # Normalize frame (ImageNet stats)
        frame = resized.astype(np.float32) / 255.0  # Scale to [0,1]
        mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
        std = np.array([0.229, 0.224, 0.225])  # ImageNet std
        frame = (frame - mean) / std  # Normalize

        frames.append(frame)
        frames_indices.append(fr_idx)

        if len(frames) == chunk_size:
            chunk = np.stack(frames, axis=0)
            chunk_tensor = torch.from_numpy(chunk).float()
            chunk_tensor = einops.rearrange(chunk_tensor, "t h w c -> 1 c t h w")

            yield frames_indices, chunk_tensor
            frames = []
            frames_indices = []

        fr_idx += 1

    cap.release()


@click.command()
@click.argument(
    "config_fp", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.argument(
    "weights_fp", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.argument("video_fp", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument(
    "output_dir",
    type=click.Path(exists=True, dir_okay=True, writable=True),
    default=".",
)
@click.option(
    "--threshold", "-t", type=float, default=0.7, help="Threshold for class logits."
)
@click.option("--stride", "-s", type=int, default=4, help="Frame stride.")
def main(
    config_fp: str,
    weights_fp: str,
    video_fp: str,
    output_dir: str,
    threshold: float,
    stride: int,
):

    video_name = Path(video_fp).stem

    cfg = Config.fromfile(config_fp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    cap = cv2.VideoCapture(video_fp)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = MODELS.build(cfg.model)
    # model.to(device)
    model.eval()

    # print("Before loading")
    # print(
    #     "   --> patch_embed",
    #     model.state_dict()["backbone.patch_embed.projection.weight"][0, 0, 0, :5],
    # )
    # print(
    #     "   --> attn qkv", model.state_dict()["backbone.blocks.0.attn.qkv.weight"][:5]
    # )
    # Load weights
    state_dict = torch.load(weights_fp, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # for k, v in state_dict.items():
    #     print(f"{k}: {v.shape}")
    # remove all  prefix before the last backbone in state_dict keys
    # state_dict = {
    #     "backbone." + k.split("backbone.")[-1]: v for k, v in state_dict.items()
    # }

    model.load_state_dict(state_dict)

    # print("After loading")
    # print(
    #     "   --> patch_embed",
    #     model.state_dict()["backbone.patch_embed.projection.weight"][0, 0, 0, :5],
    # )
    # print(
    #     "   --> attn qkv", model.state_dict()["backbone.blocks.0.attn.qkv.weight"][:5]
    # )
    chunk_size = 16
    # num chunk taking stride into account
    num_chunks = (num_frames + stride - 1) // stride // chunk_size

    list_query_score = []
    set_frame_idx: Set[int] = set()
    frame_generator = generate_frame_tensors(
        cap, chunk_size=16, target_size=(224, 224), stride=stride
    )
    for chunk_fr_idx, chunk_frames in tqdm(
        frame_generator,
        total=num_chunks,
        desc="Processing frames",
    ):
        with torch.inference_mode():
            # chunk_frames = chunk_frames.to(device)
            cls_logits, query_score = model(chunk_frames, query_score_block_idx=11)
            # get max logits and its index
            max_logits, max_indices = cls_logits.max(dim=1)
            print(f"max_logits: {max_logits} - max_indices: {max_indices}")

        query_score = einops.repeat(query_score, "b f h w -> b (f s) h w", s=stride)
        query_score = einops.rearrange(query_score, "b f h w -> (b f) h w")
        query_score = query_score.cpu().numpy()
        list_query_score.append(query_score)

        if max_logits.item() >= threshold:
            print(
                f" -> max logits {max_logits.item():.3f} >= threshold {threshold}"
            )
            set_frame_idx.update(chunk_fr_idx)

    # List[Nframes, H, W] -> Ntotal, H, W
    array_query_score = np.concatenate(list_query_score, axis=0)
    print(f"Total query scores shape: {array_query_score.shape}")

    N, H, W = array_query_score.shape

    output_path = Path(output_dir) / f"{video_name}_query_scores.mp4"
    write_query_scores(video_fp, set_frame_idx, array_query_score, H, W, output_path)

    tmp_dir = Path("/tmp/vit_query_score")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # reencode_and_replace(
    #     tmp_dir=tmp_dir,
    #     video_fp=output_path,
    #     hwaccel=True,
    # )


if __name__ == "__main__":
    main()
