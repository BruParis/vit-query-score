import click
import cv2
import numpy as np
import einops
import itertools
import torch
from pathlib import Path
from tqdm import tqdm
from mmengine.config import Config
from mmengine.registry import MODELS

# import to trigger model registration
import vit_query_score.vit


def write_query_scores(video_fp, array_query_score, H, W, output_path):
    """
    Args:
        video_fp: Path to the input video file.
        list_query_score: List of query scores for each frame, shape [Nframes, H, W].
        H: Number of patches in height.
        W: Number of patches in width.
        output_path: Path to save the output video.
    """
    print("array_query_score shape:", array_query_score.shape)
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

        if frame_idx >= len(array_query_score):
            break

        # Get the query scores for this frame
        query_scores = array_query_score[frame_idx]
        query_scores_255 = (query_scores * 255).astype(np.uint8)

        query_score_resize = cv2.resize(
            query_scores_255,
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
                0.4,
                (255, 255, 255),
                1,
            )

            # write lines to form a grid
            cv2.line(overlayed_frame, (x1, 0), (x1, frame_h), (128, 128, 128), 1)
            cv2.line(overlayed_frame, (0, y1), (frame_w, y1), (128, 128, 128), 1)

        out.write(overlayed_frame)
        frame_idx += 1
        progress_bar.update(1)

    cap.release()
    out.release()


def generate_frame_tensors(cap, chunk_size=16, target_size=(224, 224)):
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize and normalize frame
        frame = cv2.resize(frame, target_size)
        frame = frame / 255.0  # Normalize to [0, 1]
        frames.append(frame)

        if len(frames) == chunk_size:
            chunk = np.stack(frames, axis=0)
            chunk_tensor = torch.from_numpy(chunk).float()
            chunk_tensor = einops.rearrange(chunk_tensor, "t h w c -> 1 c t h w")
            yield chunk_tensor
            frames = []

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
def main(config_fp: str, weights_fp: str, video_fp: str, output_dir: str):

    video_name = Path(video_fp).stem

    cfg = Config.fromfile(config_fp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(video_fp)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = MODELS.build(cfg.model)
    # model.to(device)
    model.eval()

    # Load weights
    state_dict = torch.load(weights_fp, map_location=device)
    model.load_state_dict(state_dict)

    list_query_score = []
    for i, chunk_frames in tqdm(
        enumerate(generate_frame_tensors(cap, chunk_size=16, target_size=(224, 224))),
        total=num_frames // 16,
        desc="Processing frames",
    ):
        with torch.inference_mode():
            # chunk_frames = chunk_frames.to(device)
            cls_logits, query_score = model(chunk_frames, query_score_block_idx=11)
            # get max logits and its index
            max_logits, max_indices = cls_logits.max(dim=1)
            print(f"max_logits: {max_logits} - max_indices: {max_indices}")

        query_score = einops.rearrange(query_score, "b f h w -> (b f) h w")
        query_score = query_score.cpu().numpy()
        list_query_score.append(query_score)

    # List[Nframes, H, W] -> Ntotal, H, W
    array_query_score = np.concatenate(list_query_score, axis=0)
    print(f"Total query scores shape: {array_query_score.shape}")

    N, H, W = array_query_score.shape

    output_path = Path(output_dir) / f"{video_name}_query_scores.mp4"
    write_query_scores(video_fp, array_query_score, H, W, output_path)


if __name__ == "__main__":
    main()
